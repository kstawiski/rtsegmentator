"""Run the official GOUHFI 2.0 pipeline in its portable model/runtime mounts."""
import argparse, os, shutil, subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--input",required=True); p.add_argument("--output",required=True)
p.add_argument("--source",required=True); p.add_argument("--runtime",required=True)
p.add_argument("--mode",choices=("brain","parcellation"),required=True)
p.add_argument("--work",required=True); a=p.parse_args()
source=Path(a.source); runtime=Path(a.runtime); work=Path(a.work)
inputs=work/"inputs"; results=work/"results"; inputs.mkdir(parents=True,exist_ok=True)
shutil.copyfile(a.input,inputs/"case_0000.nii.gz")
env=os.environ.copy()
env.update(GOUHFI_HOME=str(source),PYTHONPATH=str(source),
           nnUNet_raw=str(source/"nnUNet_raw"),nnUNet_preprocessed=str(source/"nnUNet_preprocessed"),
           nnUNet_results=str(source/"trained_model"),
           # The checksum-verified official checkpoints contain NumPy scalar
           # metadata and predate PyTorch 2.6's weights_only=True default.
           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
env["PATH"]=str(runtime/"bin")+os.pathsep+env.get("PATH","")
cmd=[str(runtime/"bin/python"),str(source/"run_inference/gouhfi_inference_postpro_reo.py"),
     "-i",str(inputs),"-o",str(results),"--np","1"]
if a.mode=="brain": cmd.append("--skip_parc")
subprocess.run(cmd,env=env,cwd=source,check=True)
prediction=(results/"outputs_seg_postpro/case.nii.gz" if a.mode=="brain"
            else results/"outputs_parc_postpro/case.nii.gz")
if not prediction.is_file(): raise SystemExit(f"GOUHFI did not create {prediction}")
shutil.copyfile(prediction,a.output)
