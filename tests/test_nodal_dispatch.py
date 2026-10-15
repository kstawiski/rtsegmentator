"""Exercise new dispatch paths without model inference or real imaging data."""
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile

import pytest
from choi_ontology import REGIONS

ROOT=Path(__file__).parents[1]


@pytest.mark.parametrize('task,runner', [('choi_tmli','run_choi_tmli.py'),('lnq_atlas','run_lnq_atlas.py')])
def test_dispatch_uses_geometry_reference_and_model_labels(tmp_path,monkeypatch,synthetic_ct,task,runner):
    job=tmp_path/'job';(job/'output').mkdir(parents=True);(job/'input').symlink_to(synthetic_ct)
    monkeypatch.setenv('RTSEG_APP_ROOT',str(ROOT));monkeypatch.setenv('RTSEG_MODEL_ROOT',str(tmp_path/'models'))
    monkeypatch.setenv('RTSEG_RUNTIME_ROOT',str(tmp_path/'runtimes'))
    monkeypatch.setattr(sys,'argv',['run_task.py',str(job),task]);monkeypatch.setattr(signal,'signal',lambda *a:None)
    import choi_ontology
    monkeypatch.setattr(choi_ontology,'confirmed_labels',lambda path:dict(zip(map(str,range(1,19)),REGIONS)))
    calls=[];exported_labels=[]
    class Process:
        def __init__(self,command,**kwargs):
            calls.append(command)
            if command[1].endswith('convert_validate_rtstruct.py'):
                exported_labels.append(json.loads(Path(command[command.index('--labels')+1]).read_text()))
                shutil.copyfile(next(synthetic_ct.iterdir()),command[command.index('--output')+1])
        def wait(self):return 0
        def poll(self):return 0
    monkeypatch.setattr(subprocess,'Popen',Process)
    fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        monkeypatch.setattr(tempfile,'tempdir',f'/proc/self/fd/{fd}')
        runpy.run_path(str(ROOT/'run_task.py'),run_name='__main__')
    finally:os.close(fd)
    assert [Path(c[1]).name for c in calls]==['dicom_series_to_nifti.py',runner,'convert_validate_rtstruct.py']
    inference,export=calls[1:]
    assert inference[inference.index('--input')+1]==export[export.index('--model-input')+1]
    assert inference[inference.index('--output')+1]==export[export.index('--prediction')+1]
    assert len(exported_labels[0])==(18 if task=='choi_tmli' else 1)
