"""Run TotalSegmentator 1.5.7 with its legacy home-directory state isolated."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


def run_prediction(input_path,output_path):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(input_path,output_path,ml=True,fast=True,
                     nr_thr_resamp=1,nr_thr_saving=1)


@contextmanager
def private_workspace(state):
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=True)
    bulk=state/'bulk';bulk.mkdir(exist_ok=True)
    config_dir=state/'.totalsegmentator';config_dir.mkdir(exist_ok=True)
    config=config_dir/'config.json'
    config.write_text(json.dumps({'totalseg_id':'rtseg-lnq-local',
                                 'send_usage_stats':False,'prediction_counter':0})+'\n')
    weights=os.environ.get('TOTALSEG_WEIGHTS_PATH',str(state/'weights'))
    original_temporary_directory=tempfile.TemporaryDirectory
    def imaging_temporary_directory(*args,**kwargs):
        if kwargs.get('prefix')=='nnunet_tmp_':kwargs['dir']=str(bulk)
        return original_temporary_directory(*args,**kwargs)
    # v1 ignores TOTALSEG_HOME_DIR. Scope its Path.home() lookups to this child
    # process without changing HOME or editing the installed package.
    # Redirect TotalSegmentator's large imaging workspace, while multiprocessing
    # mkdtemp(prefix='pymp-') still uses the caller's short local TMPDIR.
    with patch.object(Path,'home',return_value=state), \
         patch.object(tempfile,'TemporaryDirectory',new=imaging_temporary_directory), \
         patch.dict(os.environ,{'TOTALSEG_WEIGHTS_PATH':weights}):
        yield


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if version('TotalSegmentator')!='1.5.7':
        raise ValueError('This launcher requires TotalSegmentator==1.5.7')
    with private_workspace(args.state_dir):
        run_prediction(args.input,args.output)


if __name__=='__main__':main()
