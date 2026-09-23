#!/usr/bin/env python3
"""Sequential, resumable post-fix orientation audit on the GPU worker."""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("tasks",nargs="*"); a=p.parse_args()
app=Path(os.environ.get("RTSEG_APP_ROOT","/app"))
root=Path(os.environ.get("RTSEG_MODEL_ROOT","/models"))
runtime=os.environ.get("RTSEG_RUNTIME_ROOT","/runtimes")
fixtures=json.loads((app/"catalog/orientation-audit-fixtures.json").read_text())
tasks=a.tasks or list(fixtures); validation=root/"orientation-validation"; validation.mkdir(exist_ok=True)
runroot=root/"results/orientation-audit-20260812"; runroot.mkdir(parents=True,exist_ok=True)
summary=runroot/"summary.jsonl"
for task in tasks:
    marker=validation/f"{task}.json"
    if marker.exists(): print(f"SKIP {task}",flush=True); continue
    if task not in fixtures: print(f"NO_FIXTURE {task}",flush=True); continue
    job=runroot/task; shutil.rmtree(job,ignore_errors=True); (job/"output").mkdir(parents=True)
    (job/"input").symlink_to(root/"testdata"/fixtures[task],target_is_directory=True)
    env=os.environ.copy(); env.update(RTSEG_APP_ROOT=str(app),RTSEG_MODEL_ROOT=str(root),RTSEG_RUNTIME_ROOT=runtime)
    log=runroot/f"{task}.log"; started=time.time(); status="failed"; error=None
    try:
        with log.open("w") as out:
            subprocess.run([str(app/".venv/bin/python"),str(app/"run_task.py"),str(job),task],env=env,
                           stdout=out,stderr=subprocess.STDOUT,check=True,timeout=12*3600)
        result=job/"output"/f"{task}_RTSTRUCT.dcm"
        subprocess.run([str(app/".venv/bin/python"),str(app/"audit_rtstruct_orientation.py"),
                        "--task",task,"--dicom",str(job/"input"),"--rtstruct",str(result),
                        "--output",str(marker)],check=True,timeout=300)
        status="passed"
    except Exception as exc: error=repr(exc)
    row={"task":task,"status":status,"seconds":round(time.time()-started,1),"error":error}
    with summary.open("a") as f: f.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
