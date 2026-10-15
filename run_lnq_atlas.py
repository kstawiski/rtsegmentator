"""Staged LNQ atlas adapter: upstream registration, mean fold probabilities, postprocessing.

Requires the pinned public archive, an nnU-Net 2.8.1 runtime, and a separate
TotalSegmentator 1.5.7 runtime. Does not modify either runtime or upstream source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SOURCE_REVISION = 'e48a74f3223e5d9ef06f79fb08282450c29f7a2d'
MODEL_RELATIVE = Path('nnUNet_results/Dataset017_LNQ2023/trainer_nnUNet_Plans_3d_fullres')
ATLAS_FILES = ('patient2-ct.nii.gz', 'atlas2_total_segm.nii.gz', 'atlas2_reg_mask.nii.gz',
               'atlas2_ct_affine_resample.nii.gz', 'atlas2_segm_affine_resample.nii.gz',
               'average_labels_from_LNQ_train_scaled.nii.gz')


def total_python(runtime_root):
    return Path(os.environ.get('RTSEG_LNQ_TOTAL_PYTHON', str(Path(runtime_root)/'envs/lnq-atlas-total-v1/bin/python')))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def required_files():
    files=[str(MODEL_RELATIVE/name) for name in ('dataset.json','plans.json')]
    files += [str(MODEL_RELATIVE/f'fold_{i}/checkpoint_best.pth') for i in range(5)]
    files += [f'data/Atlas/{name}' for name in ATLAS_FILES]
    return files


def validate_manifest(record):
    """Check provenance without reading gigabytes during a catalog request."""
    if record.get('source_revision') != SOURCE_REVISION:
        raise ValueError('LNQ manifest source revision differs from the reviewed revision')
    if not re.fullmatch('[0-9a-f]{64}',record.get('sha256','')):
        raise ValueError('Missing archive SHA-256 pin')
    hashes=record.get('files_sha256',{})
    for name in required_files():
        expected=hashes.get(name)
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError(f'Missing verified artifact pin: {name}')


def verify_bundle(weights, manifest):
    """Require an explicit SHA-256 pin for every file used for inference."""
    weights=Path(weights).resolve();record=json.loads(Path(manifest).read_text())
    validate_manifest(record)
    for name in required_files():
        expected=record['files_sha256'][name]
        path=weights/name
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'LNQ artifact hash mismatch or file missing: {name}')
    return record


def probability_threshold(probmap):
    import numpy as np
    if not np.isfinite(probmap).all() or probmap.max() <= probmap.min():
        raise ValueError('Degenerate or non-finite registered atlas probability map')
    # Published post_processing_functions.get_probability_threshold_probmap.
    return .3 * (1 - .5 * np.round((probmap-probmap.min())/(probmap.max()-probmap.min()),2))


def average_probabilities(probabilities):
    import numpy as np
    count=0;total=None
    for probability in probabilities:
        if not np.isfinite(probability).all() or np.any(probability<0) or np.any(probability>1):
            raise ValueError('Invalid foreground probability')
        if total is None:total=probability.astype(np.float32,copy=True)
        else:
            if probability.shape!=total.shape:raise ValueError('Fold probability shapes differ')
            total+=probability
        count+=1
    if count!=5:raise ValueError('The released ensemble requires all five folds')
    return total/count


def predict_probabilities(model, channels):
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    plans=PlansManager(json.loads((model/'plans.json').read_text()))
    dataset=json.loads((model/'dataset.json').read_text())
    if dataset['channel_names']!={'0':'CT','1':'noNorm','2':'rgb_to_0_1'} or dataset['labels']!={'background':0,'LN':1}:
        raise ValueError('Unexpected LNQ channel or label ontology')
    config=plans.get_configuration('3d_fullres')
    # The released trainer and both its custom parents inherit the default builder.
    network=nnUNetTrainer.build_network_architecture(plans,config,3,2,enable_deep_supervision=False)
    image,properties=plans.image_reader_writer_class().read_images([str(p) for p in channels])
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,
                             perform_everything_on_device=True,device=torch.device('cuda',0))
    for fold in range(5):
        checkpoint=torch.load(model/f'fold_{fold}/checkpoint_best.pth',map_location='cpu',weights_only=False)
        if checkpoint['init_args']['configuration']!='3d_fullres':
            raise ValueError('Unexpected LNQ configuration')
        parameters=checkpoint['network_weights']
        network.load_state_dict(parameters,strict=True)
        predictor.manual_initialization(network,plans,config,[parameters],dataset,'trainer',
                                         checkpoint.get('inference_allowed_mirroring_axes'))
        _,probabilities=predictor.predict_single_npy_array(image,properties,save_or_return_probabilities=True)
        # Average softmax probabilities AFTER each fold is restored to the image grid,
        # matching the upstream predictor's ensemble_method='avg' (not mean logits).
        yield np.asarray(probabilities[1],dtype=np.float32)
        del checkpoint,parameters,probabilities


def run(args):
    sys.dont_write_bytecode=True
    import importlib.metadata
    import numpy as np
    import SimpleITK as sitk
    if importlib.metadata.version('nnunetv2')!='2.8.1':
        raise ValueError('This adapter targets the installed nnunetv2==2.8.1 API')
    verify_bundle(args.weights,args.manifest)
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!=SOURCE_REVISION:raise ValueError('LNQ source revision differs from the reviewed revision')
    changes=subprocess.check_output(['git','-C',str(args.source),'diff','HEAD','--','Inference'],text=True)
    if changes:raise ValueError('LNQ inference source has local modifications')
    version=subprocess.check_output([str(args.total_python),'-c',
        'from importlib.metadata import version; print(version("TotalSegmentator"))'],text=True).strip()
    if version!='1.5.7':raise ValueError('Atlas registration requires TotalSegmentator==1.5.7 label IDs')
    args.work.mkdir(parents=True,exist_ok=False)
    os.environ['nnUNet_raw']=str(args.work/'raw')
    os.environ['nnUNet_preprocessed']=str(args.work/'preprocessed')
    os.environ['nnUNet_results']=str(args.weights/'nnUNet_results')
    os.environ['nnUNet_compile']='false'
    original=sitk.ReadImage(str(args.input))
    canonical=sitk.DICOMOrient(original,'LPS')
    if not np.allclose(np.array(canonical.GetDirection()).reshape(3,3),np.eye(3),atol=1e-5):
        raise ValueError('Upstream atlas registration requires axis-aligned LPS CT; oblique input needs a separately reviewed resampling policy')
    canonical_path=args.work/'canonical.nii.gz';sitk.WriteImage(canonical,str(canonical_path))
    total=args.work/'total-v1.nii.gz'
    env=os.environ.copy();env['TOTALSEG_HOME_DIR']=str(args.work/'totalseg-home')
    env['PYTHONDONTWRITEBYTECODE']='1'
    # Set TOTALSEG_WEIGHTS_PATH to an existing v1 cache or a new private cache.
    env.setdefault('TOTALSEG_WEIGHTS_PATH',str(args.work/'totalseg-weights'))
    subprocess.run([str(args.total_python),str(Path(__file__).with_name('run_total_v1.py')),
                    '--state-dir',str(args.work/'total-v1-state'),'--input',str(canonical_path),
                    '--output',str(total)],env=env,check=True)
    sys.path.insert(0,str(args.source/'Inference'))
    import preregistration.data as data
    import preregistration.preregistration as registration
    import post_processing_functions as post
    data.BASEPATH=str(args.work);data.nnUNet_raw=str(args.work/'raw')
    data.ATLASPATH=str(args.weights/'data/Atlas');data.RESULTPATH=str(args.work/'registration')
    data.PNGPATH=str(args.work/'registration/PNGs')
    def supplied_total(patient,force=False,debug=False):
        labels=sitk.ReadImage(str(total))
        if (labels.GetSize()!=canonical.GetSize() or not np.allclose(labels.GetOrigin(),canonical.GetOrigin())
            or not np.allclose(labels.GetDirection(),canonical.GetDirection())
            or not np.allclose(labels.GetSpacing(),canonical.GetSpacing())):
            raise ValueError('TotalSegmentator v1 output geometry differs from CT')
        patient.set_image('total_segm',labels)
        patient.set_image('reg_mask',registration.label_image_to_reg_mask(labels))
        patient.set_image('lung_mask',registration.label_image_to_lung_mask(labels))
    registration.total_segmentation=supplied_total
    case=args.work/'raw/case';case.mkdir(parents=True)
    sitk.WriteImage(canonical,str(case/'case_0000.nii.gz'))
    registration.process_file(str(case/'case_0000.nii.gz'),'case')
    channels=[case/f'case_000{i}.nii.gz' for i in range(3)]
    probabilities=average_probabilities(predict_probabilities(args.weights/MODEL_RELATIVE,channels))
    reference=sitk.ReadImage(str(channels[0]));probmap=sitk.ReadImage(str(channels[2]))
    threshold=probability_threshold(sitk.GetArrayFromImage(probmap))
    if probabilities.shape!=threshold.shape:raise ValueError('Atlas and prediction shapes differ')
    binary=sitk.GetImageFromArray((probabilities>=threshold).astype(np.uint8));binary.CopyInformation(reference)
    lung=args.work/'registration/subjects/case/case_lung_mask.nii.gz'
    hull=sitk.ReadImage(post.create_regmask(str(lung)))
    binary=post.pad_to_original_size(binary,hull)
    binary=post.remove_outside_lung(binary,hull)
    binary=post.remove_healthy_lymphnodes_sitk(binary,5)
    binary=sitk.Resample(binary,original,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    args.output.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(binary,str(args.output))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','output','work','weights','manifest','source','total-python'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(argv)
    run(args)


if __name__=='__main__':main()
