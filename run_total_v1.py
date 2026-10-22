"""Run TotalSegmentator 1.5.7 with its legacy home-directory state isolated."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from run_lnq_atlas import (TOTAL_DETERMINISTIC_CONTRACT, apply_total_deterministic_after_nnunet,
                           apply_total_process_environment)

_LOAD_MODEL = None
_DETERMINISTIC_APPLIED = False


def install_total_deterministic_hook():
    """After nnU-Net v1 builds the trainer, force deterministic cuDNN and benchmark off.

    The task-256 pickle restores the trainer with deterministic=False, so
    NetworkTrainer.__init__ enables cudnn.benchmark. mode 'fastest' does not
    clear that flag. predict_cases (mode 'normal') does, after the same load.
    """
    global _LOAD_MODEL
    # nnunet.paths reads RESULTS_FOLDER once, at first import. TotalSegmentator
    # sets it in setup_nnunet(), so that must run before the predictor import.
    from totalsegmentator.config import setup_nnunet
    setup_nnunet()
    import nnunet.inference.predict as predict
    if _LOAD_MODEL is None:
        _LOAD_MODEL = predict.load_model_and_checkpoint_files

    def wrapped(*args, **kwargs):
        global _DETERMINISTIC_APPLIED
        result = _LOAD_MODEL(*args, **kwargs)
        apply_total_deterministic_after_nnunet()
        _DETERMINISTIC_APPLIED = True
        return result

    predict.load_model_and_checkpoint_files = wrapped


def run_prediction(input_path,output_path):
    apply_total_process_environment()
    install_total_deterministic_hook()
    from totalsegmentator.python_api import totalsegmentator
    # Published task-256 call. tta stays off inside TotalSegmentator.
    # nr_thr_resamp is also the nnU-Net preprocessing thread count.
    # fast=True resamples the CT to 3 mm (order 3) and the mask back (order 0).
    totalsegmentator(input_path,output_path,ml=True,fast=True,
                     nr_thr_resamp=1,nr_thr_saving=1)
    if not _DETERMINISTIC_APPLIED:
        raise RuntimeError('nnU-Net v1 loaded no model; refusing an unmarked TotalSegmentator result')
    contract = Path(output_path).with_name('total_deterministic_contract.json')
    contract.write_text(json.dumps(TOTAL_DETERMINISTIC_CONTRACT, indent=2, sort_keys=True) + '\n')


@contextmanager
def private_workspace(state):
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=True)
    bulk=state/'bulk';bulk.mkdir(exist_ok=True)
    config_dir=state/'.totalsegmentator';config_dir.mkdir(exist_ok=True)
    config=config_dir/'config.json'
    config.write_text(json.dumps({'totalseg_id':'rtseg-lnq-local',
                                 'send_usage_stats':False,'prediction_counter':0})+'\n')
    weights=os.environ.get('TOTALSEG_WEIGHTS_PATH')
    if not weights:
        raise SystemExit('TOTALSEG_WEIGHTS_PATH is unset; refusing to download TotalSegmentator weights')
    from run_lnq_atlas import require_task256, require_task256_metadata
    require_task256(weights)
    require_task256_metadata(weights)
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
