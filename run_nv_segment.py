#!/usr/bin/env python3
"""Offline automatic NV-Segment runner. Uses immutable sources and a job-local bundle."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from nv_segment_logic import MODES, check_modality, generate

CHILD=None

def stop(*_):
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        # Children remain in our process group so the live worker can cancel the
        # whole inference tree. A direct invocation also reaps its immediate child.
        CHILD.terminate()
        try: CHILD.wait(timeout=15)
        except subprocess.TimeoutExpired:
            CHILD.kill(); CHILD.wait()
    raise SystemExit(143)

def run(command, *, cwd, env):
    global CHILD
    print(json.dumps({'command':[str(v) for v in command]}),flush=True)
    CHILD=subprocess.Popen([str(v) for v in command],cwd=cwd,env=env)
    code=CHILD.wait();CHILD=None
    if code: raise RuntimeError(f'Child exited {code}: {command[0]}')

def sha256(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def job_environment(work, device, scratch=None):
    """Confine publisher caches and home files to this job's writable copy."""
    env = os.environ.copy()
    scratch = scratch or work/'tmp'
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HOME=str(work/'home'),
               HF_HOME=str(work/'home/hf'), TORCH_HOME=str(work/'home/torch'),
               XDG_CACHE_HOME=str(work/'home/cache'), MPLCONFIGDIR=str(work/'home/matplotlib'),
               TRITON_CACHE_DIR=str(work/'home/triton'),
               TORCHINDUCTOR_CACHE_DIR=str(work/'home/torchinductor'),
               NUMBA_CACHE_DIR=str(work/'home/numba'),
               TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
               OMP_NUM_THREADS=env.get('NV_SEGMENT_THREADS','4'),
               ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=env.get('NV_SEGMENT_THREADS','4'))
    env.pop('PYTHONPATH', None)
    env['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else env.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    return env


def check_write_paths(paths, readonly_roots):
    for path in paths:
        if any(path.resolve().is_relative_to(root.resolve()) for root in readonly_roots):
            raise ValueError(f'Job output/scratch must be outside the model bundle and runtimes: {path}')

def prepare_bundle(source, checkpoint, work, input_path, mode, device, manifest):
    spec=MODES[mode]
    for rel,expected in manifest['source_files'].items():
        if sha256(source/rel)!=expected: raise ValueError(f'Source pin mismatch: {source/rel}')
    if sha256(checkpoint)!=spec['sha256']: raise ValueError('Checkpoint SHA-256 mismatch')
    shutil.copytree(source/'configs',work/'configs')
    shutil.copytree(source/'scripts',work/'scripts')
    (work/'models').mkdir(); (work/'models/model.pt').symlink_to(checkpoint.resolve())
    evaluator=work/'scripts/evaluator.py';text=evaluator.read_text()
    old='points = torch.from_numpy(points).to(inputs.device)'
    if text.count(old)!=1: raise ValueError('MONAI compatibility patch anchor mismatch')
    evaluator.write_text(text.replace(old,'points = torch.as_tensor(points).to(inputs.device)',1))
    p=work/'configs/inference.json';cfg=json.loads(p.read_text())
    initial=cfg['initialize']
    if len(initial)!=3 or 'prepare_huggingface_checkpoint' not in initial[0]:
        raise ValueError('Unexpected bundle initialization; cannot guarantee offline loading')
    cfg['initialize']=initial[1:]
    cfg['device']="$torch.device('cuda:0')" if device=='cuda' else "$torch.device('cpu')"
    cfg['input_dict']={'image':str(input_path.resolve())}
    cfg['output_dir']=str(work/'output')
    cfg['output_dtype']='$np.uint16'
    cfg['checkpointloader']['strict']=True
    cfg['evaluator']['amp']=(device=='cuda')
    if spec['upstream_mode']: cfg['modality']=spec['upstream_mode']
    # 255 is a real LUMIR foreground ID. Do not turn missing values into it.
    transforms=cfg['postprocessing']['transforms']
    fills=[t for t in transforms if t.get('_target_')=='Lambdad' and t.get('func')=='$lambda x: torch.nan_to_num(x, nan=255)']
    if len(fills)!=1: raise ValueError('NaN fill patch anchor mismatch')
    fills[0]['func']='$lambda x: torch.nan_to_num(x, nan=0, posinf=0, neginf=0)'
    p.write_text(json.dumps(cfg,indent=2)+'\n')
    return cfg

def patch_brain_revert_text(text):
    """Keep nearest-neighbour labels on the exact native grid without uint8 scaling."""
    replacements = {
        'reverted_mask_npy = resampling(inv_mask_npy, tar_pixdim, original_pixdim, order=0)':
            'reverted_mask_npy = zoom(inv_mask_npy.astype(np.uint16), np.array(original_reoriented.shape) / np.array(inv_mask_npy.shape), order=0, prefilter=False, mode="constant")',
        'mask_out_header = out_header.copy()':
            'mask_out_header = out_header.copy()\n        mask_out_header.set_data_dtype(np.uint16)\n        mask_out_header.set_slope_inter(1, 0)',
        'out_mask_nib = nib.Nifti1Image(reverted_mask_nib.get_fdata(), original_affine, header=mask_out_header)':
            'out_mask_nib = nib.Nifti1Image(np.asarray(reverted_mask_nib.dataobj, dtype=np.uint16), original_affine, header=mask_out_header)',
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError('Brain inverse-grid/dtype patch anchor mismatch: ' + old)
        text = text.replace(old, new, 1)
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels-out',type=Path,required=True)
    p.add_argument('--input-modality',choices=['CT','MR'])
    p.add_argument('--model-root',type=Path,default=os.environ.get('RTSEG_MODEL_ROOT','/models'))
    p.add_argument('--runtime-python',type=Path)
    p.add_argument('--brain-python',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--work',type=Path,help='Parent for a new private temporary directory; defaults beside output')
    p.add_argument('--load-only',action='store_true',help='Strict checkpoint/config probe; no prediction is written')
    args=p.parse_args()
    if args.input_modality: check_modality(args.mode,args.input_modality)
    import nibabel as nib
    import numpy as np
    spec=MODES[args.mode]; root=args.model_root.resolve(); here=Path(__file__).resolve().parent
    runtime_root=Path(os.environ.get('RTSEG_RUNTIME_ROOT',root))
    python=args.runtime_python or runtime_root/'envs/lnq/bin/python'
    brain_python=args.brain_python or root/'runtime-envs/nv-segment/bin/python'
    source=root/spec['source']; checkpoint=root/'weights/vista3d'/spec['bundle']/'model.pt'
    manifest=json.loads((here/'nv_segment_manifest.json').read_text())['modes'][args.mode]
    labels,indices=generate(source,args.mode)
    if indices!=manifest['indices']: raise ValueError('Pinned label selection changed')
    label_file=here/'labels'/f'{args.mode}.json'
    if sha256(label_file)!=manifest['labels_sha256'] or json.loads(label_file.read_text())!=labels:
        raise ValueError('Pinned ROI label map changed')
    image=nib.load(args.input)
    if len(image.shape)!=3 or any(n<=1 for n in image.shape): raise ValueError(f'Expected one 3-D volume: {image.shape}')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3]))<1e-10: raise ValueError('Invalid input affine')
    if not np.isfinite(np.asanyarray(image.dataobj)).all(): raise ValueError('Non-finite input intensity')
    if args.input.resolve() in {args.output.resolve(),args.labels_out.resolve()}: raise ValueError('Input/output paths must differ')
    if args.output.exists() or args.labels_out.exists(): raise FileExistsError('Output already exists; use a fresh job path')
    parent=args.work or args.output.parent
    check_write_paths([args.output, args.labels_out, Path(str(args.output)+'.metrics.json'), parent],
                      [root, runtime_root])
    args.output.parent.mkdir(parents=True,exist_ok=True); args.labels_out.parent.mkdir(parents=True,exist_ok=True)
    parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with ExitStack() as cleanup:
        tmp=cleanup.enter_context(tempfile.TemporaryDirectory(prefix='nv-segment-',dir=parent))
        work=Path(tmp);(work/'tmp').mkdir();(work/'home').mkdir()
        scratch=work/'tmp'
        if os.environ.get('RTSEG_SCRATCH_ROOT'):
            scratch_parent=Path(os.environ['RTSEG_SCRATCH_ROOT'])
            check_write_paths([scratch_parent], [root, runtime_root])
            scratch_parent.mkdir(parents=True, exist_ok=True)
            scratch=Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix='nvs-', dir=scratch_parent)))
        env=job_environment(work, args.device, scratch)
        prepared=args.input.resolve()
        # Verify source/checkpoint before running any publisher code.
        prepare_bundle(source,checkpoint,work,prepared,args.mode,args.device,manifest)
        brain=spec['upstream_mode']=='MRI_BRAIN'
        if brain and not args.load_only:
            synth=root/'runtime-homes/nv-segment/synthstrip-1.8'
            sm=json.loads((synth/'manifest.json').read_text())
            for record in sm['files']:
                if sha256(synth/record['image_path'])!=record['sha256']: raise ValueError('SynthStrip pin mismatch')
            scripts=list(synth.rglob('mri_synthstrip'))
            weights=list(synth.rglob('synthstrip.1.pt'))
            if len(scripts)!=1 or len(weights)!=1: raise ValueError('Expected one official SynthStrip script and standard model')
            stripped=work/'stripped.nii.gz';mask=work/'brain_mask.nii.gz'
            run([brain_python,scripts[0],'-i',prepared,'-o',stripped,'-m',mask,'--model',weights[0]],cwd=work,env=env)
            stripped_image=nib.load(stripped)
            if stripped_image.shape!=image.shape or not np.allclose(stripped_image.affine,image.affine,rtol=0,atol=1e-5):
                raise ValueError('SynthStrip changed the native grid')
            bp=work/'brain_t1_preprocess'
            shutil.copytree(source/'brain_t1_preprocess',bp)
            # Zooming by reciprocal rounded spacings can lose/add one voxel.
            # Use the exact original reoriented shape for the inverse label zoom.
            revert=bp/'revert_preprocess.py'
            revert.write_text(patch_brain_revert_text(revert.read_text()))
            prepared=work/'aligned.nii.gz';meta=work/'preprocess.json'
            run([brain_python,bp/'preprocess.py',stripped,bp/'LUMIR_template.nii.gz',prepared,'--save-preprocess',meta],cwd=work,env=env)
            cp=work/'configs/inference.json';cfg=json.loads(cp.read_text());cfg['input_dict']={'image':str(prepared)}
            cp.write_text(json.dumps(cfg,indent=2)+'\n')
        cmd=[python,here/'nv_segment_bundle.py']
        if args.load_only: cmd+=['--load-only']
        run(cmd,cwd=work,env=env)
        metrics=json.loads((work/'bundle_metrics.json').read_text())
        if not args.load_only:
            predictions=list((work/'output').rglob('*_trans.nii.gz'))
            if len(predictions)!=1: raise ValueError(f'Expected one prediction, found {len(predictions)}')
            prediction=predictions[0]
            if brain:
                prediction=work/'native.nii.gz'
                run([brain_python,bp/'revert_preprocess.py',prepared,'--out',work/'reverted_image.nii.gz',
                     '--mask',predictions[0],'--mask-out',prediction,'--meta',meta,'--org',args.input.resolve()],cwd=work,env=env)
            pred=nib.load(prediction);data=np.asanyarray(pred.dataobj)
            if pred.shape!=image.shape or not np.allclose(pred.affine,image.affine,rtol=0,atol=1e-4):
                raise ValueError(f'Native grid mismatch: {pred.shape} versus {image.shape}; no geometry relabelling performed')
            if not np.isfinite(data).all() or not np.array_equal(data,np.rint(data)): raise ValueError('Prediction must contain finite integer labels')
            values,counts=np.unique(data,return_counts=True)
            unknown=set(map(int,values))-{0,*indices}
            if unknown: raise ValueError(f'Unmapped labels: {sorted(unknown)}')
            if not np.any(data): raise ValueError('Empty automatic segmentation')
            header=image.header.copy();header.set_data_dtype(np.uint16);header.set_slope_inter(1,0)
            out=nib.Nifti1Image(data.astype(np.uint16),image.affine,header)
            nib.save(out,args.output)
            saved=nib.load(args.output)
            if saved.shape!=image.shape or not np.array_equal(saved.affine,image.affine):
                args.output.unlink();raise ValueError('Saved output grid is not exactly equal to the input')
            args.labels_out.write_text(json.dumps(labels,indent=2)+'\n')
            metrics.update(shape=list(image.shape),affine=image.affine.tolist(),label_counts={str(int(v)):int(c) for v,c in zip(values,counts)},
                           nonempty_foreground_labels=int(np.count_nonzero(values)),output_dtype=str(saved.get_data_dtype()),exact_input_grid=True)
        metrics.update(mode=args.mode,checkpoint_sha256=spec['sha256'],checkpoint_revision=spec['revision'],source_commit=spec['source_commit'],
                       total_seconds=time.monotonic()-start,device=args.device)
        path=Path(str(args.output)+'.metrics.json');path.write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main()
