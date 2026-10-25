"""Generate DICOM ROI names and source-file pins from the official bundle files."""
import argparse, hashlib, json
from pathlib import Path
from nv_segment_logic import MODES, generate


def _file_sha256(path):
    # hashlib.file_digest needs Python 3.11; the portal image runs 3.10.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'modes':{}}
    for mode,spec in MODES.items():
        source=a.model_root/spec['source'];labels,indices=generate(source,mode)
        (a.output_dir/f'{mode}.json').write_text(json.dumps(labels,indent=2,ensure_ascii=True)+'\n')
        files=[*sorted((source/'configs').glob('*')), *sorted((source/'scripts').glob('*.py'))]
        if spec['upstream_mode']=='MRI_BRAIN':
            files+=sorted((source/'brain_t1_preprocess').rglob('*.py'))
            files+=[source/'brain_t1_preprocess/LUMIR_template.nii.gz']
        hashes={str(f.relative_to(source)):_file_sha256(f) for f in files if f.is_file()}
        manifest['modes'][mode]={**spec,'indices':indices,'source_files':hashes,
             'labels_sha256':_file_sha256(a.output_dir/f'{mode}.json')}
        print(mode, 'foreground labels:',len(labels),'range:',min(indices),max(indices),'label 255:',labels.get('255'))
    (a.output_dir.parent/'nv_segment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
