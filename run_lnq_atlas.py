"""Staged LNQ atlas adapter: upstream registration, mean fold probabilities, postprocessing.

Requires the pinned public archive, the nnU-Net 2.2.1 runtime at
runtime-envs/lnq-atlas-ref (dynamic-network-architectures 0.2), and a separate
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
PUBLIC_FIXTURE = 'msd-lung/dicom'
TOTAL_TRAINER_PREFIX = 'nnUNetTrainerV2_ep8000_nomirror'
TOTAL_TRAINER = 'nnUNetTrainerV2_ep8000_nomirror__nnUNetPlansv2.1'
TASK256_DIR = Path('nnUNet/3d_fullres/Task256_TotalSegmentator_3mm_1139subj')
TASK256_CHECKPOINT = TASK256_DIR / TOTAL_TRAINER / 'fold_0/model_final_checkpoint.model'
# with_suffix('.pkl') would replace '.model' and miss the released '.model.pkl' name.
TASK256_CHECKPOINT_PKL = TASK256_CHECKPOINT.with_name(TASK256_CHECKPOINT.name + '.pkl')
TASK256_PLANS = TASK256_DIR / TOTAL_TRAINER / 'plans.pkl'
TASK256_SHA256 = '3155cf912468be1536d6cf6e2be76f9e5ada4e7b7e26bfd8098cb677cd71eca3'
TASK256_CHECKPOINT_PKL_SHA256 = '4ca47d9e83cca5549dd5c25c458e8206efc065b80fec4a4473ab2cb781c7ed5f'
TASK256_PLANS_SHA256 = 'd166ffed489513f162e5d0a9809687b075702c87c5681d6911df6c33f3c6ae96'
# Set before the first torch import in this process. nnU-Net 2.2.1 does not
# turn cudnn.benchmark on in nnUNetPredictor.__init__; nnU-Net 2.8.1 did. The
# adapter still sets the flags immediately after that constructor.
# cudnn_benchmark is the string "off", not boolean false: a boolean false anywhere
# in an acceptance marker is rejected as a concealed failure.
REQUIRED_NNUNET = '2.2.1'
REQUIRED_DNA = '0.2'
LNQ_PYTHON_RELATIVE = Path('runtime-envs/lnq-atlas-ref/bin/python')
DETERMINISTIC_CONTRACT = {
    'cudnn_deterministic': True,
    'cudnn_benchmark': 'off',
    'cublas_workspace_config': ':4096:8',
    'nnunet_compile': 'false',
    'applied_after_nnunet_predictor_init': True,
}
# Task 256 fast path (ml=True, fast=True) restores nnUNetTrainerV2_ep8000_nomirror
# with deterministic=False and fp16=True in the checkpoint pickle. That constructor
# sets cudnn.benchmark. predict_cases_fastest, unlike predict_cases, does not turn
# it off. The flags below are applied after load_model_and_checkpoint_files returns.
# mixed precision, fastest mode and 3 mm resampling stay at the published values.
TOTAL_DETERMINISTIC_CONTRACT = {
    'cudnn_deterministic': True,
    'cudnn_benchmark': 'off',
    'cublas_workspace_config': ':4096:8',
    'applied_after_nnunet_v1_model_load': True,
    'tta': 'off',
    'mixed_precision': 'on',
    'autocast': 'on',
    'all_in_gpu': 'on',
    'prediction_mode': 'fastest',
    'threads_preprocessing': 1,
    'threads_resampling': 1,
    'threads_saving': 1,
    'omp_num_threads': 1,
    'resample_spacing_mm': 3.0,
    'resample_image_order': 3,
    'resample_mask_order': 0,
    'task_id': 256,
    'trainer': TOTAL_TRAINER,
    'ml': True,
    'fast': True,
}
LNQ_CHECKS = frozenset({
    'adapter_vs_upstream', 'dice_exactly_1', 'identical_geometry', 'both_nonempty',
    'shared_totalsegmentator_v1_mask', 'gpu_inference', 'portal_rtstruct', 'public_fixture',
    'production_subprocess', 'deterministic_contract', 'reference_environment',
    'inference_artifact_hashes', 'execution_device_binding',
    'totalsegmentator_deterministic_contract', 'production_repeated', 'reference_runtime',
})
ROI_NAMES = ['Mediastinal_Lymph_Node_Candidate']
# Git blob SHA-1 of each tracked Inference file at SOURCE_REVISION.
# sha1("blob {size}\0" + bytes). Mode is not part of the blob hash.
# Tree digest is sha256 of the JSON object with sorted keys and compact separators.
INFERENCE_TREE_SHA256 = 'e038099e7c46dfe5c4ca725c561375339d1efc7aaaddd9a636db7ae736748467'
INFERENCE_BLOBS = {
    'Inference/.dockerignore': '70dad0131dd1c0d97fa3453f0dda27c655719fb0',
    'Inference/Dockerfile': '8195b84651c95dacc56f641981876d5966c2fe54',
    'Inference/README.md': '59d2843c56e7b398e884e85734d4a5503d760687',
    'Inference/download_weights_totalsegm.py': '64f0b830c235345469a1eb940dfc711f9a812945',
    'Inference/nnUnet/adv_bias.py': '2c1948767d61e8be91e7c401a7552fcc74371508',
    'Inference/nnUnet/adv_transformation_base.py': 'd1a20d73d99e088e9d742b72f3da2c30f3267815',
    'Inference/nnUnet/final_trainer.py': 'bb4b3ffa5d1b4e8b0448a3ed244ebaaa62e8c28a',
    'Inference/nnUnet/nnUNetTrainerExtraInfo.py': 'fcb21f17a5956399c1e6cd3a0babb4bb2c08475a',
    'Inference/nnUnet/nnUNetTrainerSmartLoadBougetSampling.py': 'de2f4e4cafacd64c6d67a0559c980c1a284586de',
    'Inference/nnUnet/polylr_refinement.py': '0edfea013c08f0f4d4cb9fc03cae76e592dba8f9',
    'Inference/nnUnet/probmap_weighted_diceloss_utils.py': 'b39e18c094dcfe17fe19171b42e714f277fc44a4',
    'Inference/nnUnet/rand_conv3d.py': 'f48f8a016b7b841426484ffe35d401b363a9e5b1',
    'Inference/nnUnet/smart_cache_nondet_multi_threaded_augmenter.py': 'a5fab36e989a3120d7e437debbac21e12368fb46',
    'Inference/post_processing_functions.py': '4d4ca57872d6c36df62da0ef926ccd9e062ff81b',
    'Inference/predictor.py': '370ed5e3804b8250ec1d1812bf9d482a9ae93fed',
    'Inference/preprocessing_functions.py': '6c68dd6279556b36326721fd72f012389ff8d901',
    'Inference/preregistration/data.py': '15a18603f370a8e2d9a435aab8a4487846025f67',
    'Inference/preregistration/imi_varreg/gaussian_smoothing.py': '270aaca85f04ac977af1570b0882dcb3a5d12c95',
    'Inference/preregistration/imi_varreg/image_gradient.py': 'c5e50d3996b1197962b2819e96349d45926bcbb8',
    'Inference/preregistration/imi_varreg/imi_debug_tools.py': '6c5b0d4b698495bb6304e7c26bac073644386999',
    'Inference/preregistration/imi_varreg/imi_deformation.py': '6b09391261bb5e23f622948a525cdba5eaf86609',
    'Inference/preregistration/imi_varreg/imi_distance_ncc.py': '5ea02eaa74836c43cebdd3c7ed2334f316fc3c2e',
    'Inference/preregistration/imi_varreg/imi_image_pyramid.py': 'bb710ec937cc91a5c6310f669d3b611eed826422',
    'Inference/preregistration/imi_varreg/imi_image_registration.py': '5164f6932d2da1ab82c096c1863036d6b2bd1750',
    'Inference/preregistration/imi_varreg/imi_image_sitk_tools.py': 'b8ef7ff990961b8ce120ecbd277274d2b4694f33',
    'Inference/preregistration/imi_varreg/imi_image_tools.py': '1f7a2ab3874a776a72e0faa96a51f21a2b32fe06',
    'Inference/preregistration/imi_varreg/imi_image_warp.py': 'a75a23d8b12df77f3773a1f78c8a984824404378',
    'Inference/preregistration/imi_varreg/imi_jacobian.py': '84166ecaa093c0b27fd8a7f43a936c184c72d888',
    'Inference/preregistration/imi_varreg/imi_regularize_diffusive.py': 'fccac826f264ce86a1b14b6d44e53bb98569159c',
    'Inference/preregistration/imi_varreg/imi_regularized_warper.py': '768c624ecd2d9b13a653359d7e6f24685001932a',
    'Inference/preregistration/imi_varreg/imi_stop_criterion.py': 'a30def55274379ac4545798d2809a3d68be6ba10',
    'Inference/preregistration/imi_varreg/imi_variational_registration.py': '7179570c82ceaa2017abbcb5532f1699cf35918a',
    'Inference/preregistration/imi_varreg/jacobian.py': '3c83e1119b042cf05ae4ea5ffad2182062bf8444',
    'Inference/preregistration/imi_varreg/loss_thirion.py': 'c53494ca352ee16dde6e0ff487552228638595e4',
    'Inference/preregistration/prereg_utils.py': 'f378b577d66af8543dc3438d6d6cf4af5384ca77',
    'Inference/preregistration/preregistration.py': '1d057683a425e90833718b1d1da89c9dd4c9b9a2',
    'Inference/preregistration/sitk_linear_registration.py': '8486664759c09fb6a7af561a9b6c85445636b766',
    'Inference/process.py': 'edbb96a8a2566abd041c864bf6b87b4b6d95e19f',
    'Inference/requirements.txt': '2316acd947cceb05d66c95e0c45fbc46e8998f38',
}


def total_python(model_root):
    # Bundle-owned environment, like the prostate runtimes: the shared envs/ image is read-only.
    return Path(os.environ.get('RTSEG_LNQ_TOTAL_PYTHON', str(Path(model_root)/'runtime-envs/lnq-atlas-total-v1/bin/python')))


def lnq_python(model_root):
    """Network runtime built from the lnq-upstream-ref lock. Not envs/lnq."""
    return Path(model_root) / LNQ_PYTHON_RELATIVE


def total_weights_root(model_root):
    override = os.environ.get('RTSEG_LNQ_TOTAL_WEIGHTS')
    if override:
        return Path(override)
    return Path(model_root) / 'weights/totalsegmentator-v1.5.7'


def require_task256(weights_root):
    """Stop before TotalSegmentator 1.5.7, which downloads a missing task folder."""
    root = Path(weights_root)
    task_dir = root / TASK256_DIR
    if not task_dir.is_dir():
        raise SystemExit(
            f'TotalSegmentator 1.5.7 task 256 weights are missing: {task_dir}; refusing to download')
    trainers = sorted(path.name for path in task_dir.iterdir()
                      if path.is_dir() and path.name.startswith(TOTAL_TRAINER_PREFIX))
    if trainers != [TOTAL_TRAINER]:
        raise SystemExit(
            'TotalSegmentator 1.5.7 task 256 trainer directory is missing or ambiguous; refusing to download')
    checkpoint = root / TASK256_CHECKPOINT
    if not checkpoint.is_file():
        raise SystemExit(
            f'TotalSegmentator 1.5.7 task 256 checkpoint is missing: {checkpoint}; refusing to download')
    for relative in (TASK256_CHECKPOINT_PKL, TASK256_PLANS):
        path = root / relative
        if not path.is_file():
            raise SystemExit(
                f'TotalSegmentator 1.5.7 task 256 metadata is missing: {path}; refusing to download')
    return checkpoint


def task256_metadata_hashes(weights_root):
    """Hash the checkpoint, its pickle, and plans.pkl. Missing or changed files raise ValueError."""
    root = Path(weights_root)
    expected = {
        TASK256_CHECKPOINT: TASK256_SHA256,
        TASK256_CHECKPOINT_PKL: TASK256_CHECKPOINT_PKL_SHA256,
        TASK256_PLANS: TASK256_PLANS_SHA256,
    }
    found = {}
    for relative, pin in expected.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f'TotalSegmentator 1.5.7 task 256 metadata missing: {relative}')
        digest = sha256(path)
        if digest != pin:
            raise ValueError(
                f'TotalSegmentator 1.5.7 task 256 metadata hash does not match the pin: {relative}')
        found[str(relative)] = digest
    return found


def require_task256_metadata(weights_root):
    """Refuse to launch TotalSegmentator when task 256 metadata is missing or changed."""
    try:
        return task256_metadata_hashes(weights_root)
    except ValueError as exc:
        raise SystemExit(f'{exc}; refusing to launch') from exc


def sha256(path):
    # hashlib.file_digest needs Python 3.11; the portal image runs 3.10.
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _capability(major, minor):
    if type(major) is not int or type(minor) is not int or major < 0 or minor < 0:
        return None
    return major * 10 + minor


def device_supported(cuda_available, major, minor, arch_list):
    """True when CUDA is available and this runtime can execute on that device.

    An ordinary sm_XY cubin covers a device with the same major capability and
    a minor capability greater than or equal to Y. A compute_XY PTX target can
    be JIT-compiled for a device of that capability or newer. An
    architecture-specific token such as sm_90a matches only that exact token.
    The probe reports integer major and minor, so the device token is
    sm_{major*10+minor}. sm_90a is not that token and does not use the minor
    rule, so sm_90a on device 9.0 is false. That refusal is not a statement
    that an sm_90a cubin cannot run on an H100.
    """
    capability = _capability(major, minor)
    if not cuda_available or capability is None:
        return False
    device_major, device_minor = divmod(capability, 10)
    for name in (str(item) for item in (arch_list or ())):
        plain = re.fullmatch(r'sm_(\d+)', name)
        if plain:
            target = int(plain.group(1))
            target_major, target_minor = divmod(target, 10)
            if target_major == device_major and device_minor >= target_minor:
                return True
            continue
        if re.fullmatch(r'sm_\d+[A-Za-z]\w*', name):
            continue
        match = re.fullmatch(r'compute_(\d+)', name)
        if match and int(match.group(1)) <= capability:
            return True
    return False


def device_refusal_message(payload):
    arch = ' '.join(str(item) for item in (payload.get('arch_list') or ()))
    major, minor = payload.get('major'), payload.get('minor')
    capability = _capability(major, minor)
    if not payload.get('cuda_available'):
        shown = 'cuda unavailable'
    elif capability is None:
        shown = 'unknown capability'
    else:
        shown = f'sm_{capability}'
    return (f'LNQ runtime cannot execute on this GPU ({shown}; compiled {arch}); '
            'route lnq_atlas to the H100 worker')


def current_process_device():
    import torch
    available = bool(torch.cuda.is_available())
    arch = [str(item) for item in torch.cuda.get_arch_list()]
    if not available:
        return {'cuda_available': False, 'major': None, 'minor': None, 'arch_list': arch}
    major, minor = torch.cuda.get_device_capability(0)
    return {'cuda_available': True, 'major': int(major), 'minor': int(minor), 'arch_list': arch}


def require_execution_device(payload=None):
    if payload is None:
        payload = current_process_device()
    if not device_supported(payload.get('cuda_available'), payload.get('major'),
                            payload.get('minor'), payload.get('arch_list')):
        raise SystemExit(device_refusal_message(payload))
    return payload


DEVICE_PROBE = (
    'import json, torch\n'
    'info = {"cuda_available": bool(torch.cuda.is_available()),'
    ' "arch_list": list(torch.cuda.get_arch_list()), "name": None}\n'
    'if info["cuda_available"]:\n'
    '    major, minor = torch.cuda.get_device_capability(0)\n'
    '    info["major"] = int(major); info["minor"] = int(minor)\n'
    '    info["name"] = torch.cuda.get_device_name(0)\n'
    'else:\n'
    '    info["major"] = None; info["minor"] = None\n'
    'print(json.dumps(info))\n'
)


def probe_interpreter_device(python):
    """Ask the LNQ interpreter on this host whether its torch build can run here."""
    python = Path(python)
    if not python.is_file():
        raise FileNotFoundError(str(python))
    completed = subprocess.run([str(python), '-c', DEVICE_PROBE],
                               capture_output=True, text=True, timeout=180, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or 'probe failed').strip().splitlines()
        raise RuntimeError(detail[-1][:300] if detail else 'probe failed')
    line = completed.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise RuntimeError('LNQ device probe did not return an object')
    return payload


def apply_deterministic_environment():
    """Set before torch initializes CUDA. CUBLAS reads its workspace config once."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = DETERMINISTIC_CONTRACT['cublas_workspace_config']
    os.environ['nnUNet_compile'] = DETERMINISTIC_CONTRACT['nnunet_compile']


def apply_deterministic_after_nnunet(backends=None):
    """Set the production cuDNN flags after the nnU-Net constructor."""
    if backends is None:
        import torch
        backends = torch.backends
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False


def apply_total_deterministic_after_nnunet(backends=None):
    """Task 256 sets benchmark during trainer construction. Override that."""
    apply_deterministic_after_nnunet(backends)


def apply_total_process_environment():
    """Set before TotalSegmentator imports torch. One thread for the host libraries."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = TOTAL_DETERMINISTIC_CONTRACT['cublas_workspace_config']
    threads = str(TOTAL_DETERMINISTIC_CONTRACT['omp_num_threads'])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'nnUNet_def_n_proc'):
        os.environ[key] = threads


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def inference_tree_sha256(blobs):
    payload = json.dumps(blobs, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def require_installed_source(source):
    """Read the installed revision without git, then check every Inference blob."""
    from prostate_models import installed_source_commit
    commit = installed_source_commit(source)
    if commit != SOURCE_REVISION:
        raise ValueError('Installed LNQ source revision differs from the reviewed revision')
    root = Path(source)
    found = {}
    for relative, expected in INFERENCE_BLOBS.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f'LNQ inference file missing: {relative}')
        digest = git_blob_sha1(path)
        if digest != expected:
            raise ValueError(f'LNQ inference source changed: {relative}')
        found[relative] = digest
    tree = inference_tree_sha256(found)
    if tree != INFERENCE_TREE_SHA256:
        raise ValueError('LNQ inference source integrity mismatch')
    return commit, tree


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


def validate_acceptance(record, weights_dir, provenance_path, total_root, manifest_path, source_path=None,
                        device=None):
    """Accept an LNQ marker only when agreement, hashes and the RTSTRUCT all passed."""
    from cads_variants import marker_has_failure

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(isinstance(record, dict) and record.get('task') == 'lnq_atlas' and record.get('result') == 'passed',
            'Acceptance task/result mismatch')
    require(not marker_has_failure(record), 'Acceptance contains a failure, deviation, or waiver')
    require(record.get('fixture') == PUBLIC_FIXTURE, 'Acceptance fixture is not the confirmed public fixture')
    require(record.get('source_revision') == SOURCE_REVISION, 'Acceptance source revision mismatch')
    dice = record.get('dice')
    require(type(dice) in (int, float) and not isinstance(dice, bool) and dice == 1,
            'Agreement Dice is not exactly 1')
    require(record.get('identical_voxels') is True, 'Adapter and upstream voxels differ')
    require(record.get('both_nonempty') is True, 'Empty masks cannot pass')
    require(record.get('geometry_matches_input') is True, 'Mask geometry does not match the input')
    require(record.get('totalsegmentator_version') == '1.5.7', 'TotalSegmentator version is not 1.5.7')
    require(record.get('validator') == 'convert_validate_rtstruct.py', 'Unexpected RTSTRUCT validator')
    require(type(record.get('nonempty_rois')) is int and record['nonempty_rois'] >= 1, 'Empty RTSTRUCT cannot pass')
    require(record.get('roi_names') == ROI_NAMES, 'Unexpected RTSTRUCT ROI names')
    require(type(record.get('referenced_images_with_contours')) is int
            and record['referenced_images_with_contours'] >= 1,
            'RTSTRUCT does not reference source images')
    require(type(record.get('rtstruct_bytes')) is int and record['rtstruct_bytes'] > 0, 'RTSTRUCT is empty')
    checks = record.get('checks')
    require(isinstance(checks, list) and all(isinstance(item, str) for item in checks) and LNQ_CHECKS <= set(checks),
            'Incomplete acceptance checks')
    require(record.get('deterministic_contract') == DETERMINISTIC_CONTRACT,
            'Deterministic contract mismatch')
    provenance = json.loads(Path(provenance_path).read_text())
    validate_manifest(provenance)
    require(provenance.get('source_revision') == SOURCE_REVISION, 'Provenance source revision mismatch')
    if source_path is None:
        raise ValueError('LNQ source path was not supplied')
    installed_commit, tree = require_installed_source(source_path)
    require(installed_commit == record.get('source_revision') == SOURCE_REVISION,
            'Installed LNQ source revision mismatch')
    require(record.get('inference_tree_sha256') == tree == INFERENCE_TREE_SHA256,
            'LNQ inference source integrity mismatch')
    weights_dir = Path(weights_dir)
    provenance_hashes = provenance['files_sha256']
    installed = {}
    for name in required_files():
        path = weights_dir / name
        if not path.is_file():
            raise ValueError(f'Installed LNQ artifact missing: {name}')
        digest = sha256(path)
        if digest != provenance_hashes.get(name):
            raise ValueError(f'Installed LNQ artifact does not match provenance: {name}')
        installed[name] = digest
    require(record.get('files_sha256') == installed,
            'Acceptance artifact hash map is missing or does not match the installed files')
    checkpoints = {name: installed[name] for name in required_files() if name.endswith('checkpoint_best.pth')}
    require(record.get('checkpoint_hashes') == checkpoints, 'Checkpoint hashes mismatch')
    require(record.get('totalsegmentator_contract') == TOTAL_DETERMINISTIC_CONTRACT,
            'TotalSegmentator deterministic contract mismatch')
    require(record.get('nnunetv2') == REQUIRED_NNUNET, 'Acceptance nnunetv2 is not 2.2.1')
    require(record.get('dynamic_network_architectures') == REQUIRED_DNA,
            'Acceptance dynamic-network-architectures is not 0.2')
    require(record.get('runtime') == 'runtime-envs/lnq-atlas-ref',
            'Acceptance runtime is not lnq-atlas-ref')
    require(isinstance(device, dict), 'Execution device was not supplied')
    require(isinstance(device.get('name'), str) and device.get('name') != ''
            and record.get('execution_device_name') == device['name'],
            'Acceptance device does not match this GPU')
    require(type(device.get('major')) is int and type(device.get('minor')) is int
            and type(record.get('execution_device_major')) is int
            and type(record.get('execution_device_minor')) is int
            and record['execution_device_major'] == device['major']
            and record['execution_device_minor'] == device['minor'],
            'Acceptance compute capability does not match this GPU')
    manifest = json.loads(Path(manifest_path).read_text())
    found = task256_metadata_hashes(total_root)
    digest = found[str(TASK256_CHECKPOINT)]
    require(manifest.get('checkpoint_sha256') == digest == TASK256_SHA256,
            'Task 256 checkpoint does not match the pin')
    require(manifest.get('checkpoint_pkl_sha256') == found[str(TASK256_CHECKPOINT_PKL)] == TASK256_CHECKPOINT_PKL_SHA256,
            'Task 256 checkpoint pickle does not match the pin')
    require(manifest.get('plans_pkl_sha256') == found[str(TASK256_PLANS)] == TASK256_PLANS_SHA256,
            'Task 256 plans pickle does not match the pin')
    require(record.get('task256_checkpoint_sha256') == digest, 'Task 256 hash mismatch')
    require(record.get('task256_checkpoint_pkl_sha256') == TASK256_CHECKPOINT_PKL_SHA256,
            'Task 256 checkpoint pickle hash mismatch')
    require(record.get('task256_plans_pkl_sha256') == TASK256_PLANS_SHA256,
            'Task 256 plans pickle hash mismatch')


def validate_orientation(record, task):
    """Require the geometry payload written by audit_rtstruct_orientation.py."""
    from cads_variants import validate_orientation as validate_shared_orientation
    validate_shared_orientation(record, task)


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
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    plans=PlansManager(json.loads((model/'plans.json').read_text()))
    dataset=json.loads((model/'dataset.json').read_text())
    if dataset['channel_names']!={'0':'CT','1':'noNorm','2':'rgb_to_0_1'} or dataset['labels']!={'background':0,'LN':1}:
        raise ValueError('Unexpected LNQ channel or label ontology')
    config=plans.get_configuration('3d_fullres')
    inputs=determine_num_input_channels(plans, config, dataset)
    if inputs!=3:
        raise ValueError('Unexpected LNQ input channel count')
    # nnU-Net 2.2.1 signature: plans, dataset, configuration, input channels.
    # The released trainer inherits this method and does not replace it.
    # Output heads come from the label manager inside the builder.
    network=nnUNetTrainer.build_network_architecture(
        plans, dataset, config, inputs, enable_deep_supervision=False)
    image,properties=plans.image_reader_writer_class().read_images([str(p) for p in channels])
    predictor=nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                             perform_everything_on_gpu=True, device=torch.device('cuda', 0),
                             verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    # nnU-Net 2.2.1 leaves cuDNN flags alone. The production contract sets them here.
    apply_deterministic_after_nnunet()
    for fold in range(5):
        checkpoint=torch.load(model/f'fold_{fold}/checkpoint_best.pth', map_location='cpu', weights_only=False)
        if checkpoint['init_args']['configuration']!='3d_fullres':
            raise ValueError('Unexpected LNQ configuration')
        parameters=checkpoint['network_weights']
        network.load_state_dict(parameters, strict=True)
        predictor.manual_initialization(network, plans, config, [parameters], dataset, 'trainer',
                                         checkpoint.get('inference_allowed_mirroring_axes'))
        _,probabilities=predictor.predict_single_npy_array(image, properties, save_or_return_probabilities=True)
        # Average softmax probabilities AFTER each fold is restored to the image grid,
        # matching the upstream predictor's ensemble_method='avg' (not mean logits).
        yield np.asarray(probabilities[1], dtype=np.float32)
        del checkpoint, parameters, probabilities


def run(args):
    sys.dont_write_bytecode=True
    # CUBLAS_WORKSPACE_CONFIG is read when CUDA initialises. Set it before torch.
    apply_deterministic_environment()
    require_execution_device()
    import importlib.metadata
    import numpy as np
    import SimpleITK as sitk
    nnunet_version=importlib.metadata.version('nnunetv2')
    dna_version=importlib.metadata.version('dynamic-network-architectures')
    if nnunet_version!=REQUIRED_NNUNET or dna_version!=REQUIRED_DNA:
        raise ValueError(
            f'This adapter requires nnunetv2=={REQUIRED_NNUNET} and '
            f'dynamic-network-architectures=={REQUIRED_DNA}; '
            f'found {nnunet_version} and {dna_version}')
    verify_bundle(args.weights,args.manifest)
    require_installed_source(args.source)
    version=subprocess.check_output([str(args.total_python),'-c',
        'from importlib.metadata import version; print(version("TotalSegmentator"))'],text=True).strip()
    if version!='1.5.7':raise ValueError('Atlas registration requires TotalSegmentator==1.5.7 label IDs')
    args.work.mkdir(parents=True,exist_ok=False)
    os.environ['nnUNet_raw']=str(args.work/'raw')
    os.environ['nnUNet_preprocessed']=str(args.work/'preprocessed')
    os.environ['nnUNet_results']=str(args.weights/'nnUNet_results')
    os.environ['nnUNet_compile']=DETERMINISTIC_CONTRACT['nnunet_compile']
    original=sitk.ReadImage(str(args.input))
    canonical=sitk.DICOMOrient(original,'LPS')
    if not np.allclose(np.array(canonical.GetDirection()).reshape(3,3),np.eye(3),atol=1e-5):
        raise ValueError('Upstream atlas registration requires axis-aligned LPS CT; oblique input needs a separately reviewed resampling policy')
    canonical_path=args.work/'canonical.nii.gz';sitk.WriteImage(canonical,str(canonical_path))
    total=args.work/'total-v1.nii.gz'
    weights_root=Path(args.total_weights)
    require_task256(weights_root)
    require_task256_metadata(weights_root)
    env=os.environ.copy();env['TOTALSEG_HOME_DIR']=str(args.work/'totalseg-home')
    env['PYTHONDONTWRITEBYTECODE']='1'
    # A private cache would make TotalSegmentator 1.5.7 download task 256.
    env['TOTALSEG_WEIGHTS_PATH']=str(weights_root)
    # Thread caps stay in the child. The parent still runs the LNQ network.
    env['CUBLAS_WORKSPACE_CONFIG']=TOTAL_DETERMINISTIC_CONTRACT['cublas_workspace_config']
    total_threads=str(TOTAL_DETERMINISTIC_CONTRACT['omp_num_threads'])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','nnUNet_def_n_proc'):
        env[key]=total_threads
    subprocess.run([str(args.total_python),str(Path(__file__).with_name('run_total_v1.py')),
                    '--state-dir',str(args.work/'total-v1-state'),'--input',str(canonical_path),
                    '--output',str(total)],env=env,check=True)
    total_contract_path=args.work/'total_deterministic_contract.json'
    if not total_contract_path.is_file():
        raise RuntimeError('TotalSegmentator deterministic contract was not recorded')
    total_contract=json.loads(total_contract_path.read_text())
    if total_contract!=TOTAL_DETERMINISTIC_CONTRACT:
        raise RuntimeError('TotalSegmentator deterministic contract mismatch')
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
    # Diagnostic only. The mask below is still thresholded on the registration grid
    # and restored with nearest neighbour. This map is the pre-threshold foreground
    # probability on the original CT grid.
    if probabilities.shape!=sitk.GetArrayFromImage(reference).shape:
        raise ValueError('Foreground probability shape differs from the registration image')
    probability_image=sitk.GetImageFromArray(np.asarray(probabilities,dtype=np.float32))
    probability_image.CopyInformation(reference)
    restored_probability=sitk.Resample(probability_image,original,sitk.Transform(),
                                       sitk.sitkLinear,0,sitk.sitkFloat32)
    sitk.WriteImage(restored_probability,str(args.work/'foreground-probability.nii.gz'))
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
    (args.work/'deterministic_contract.json').write_text(
        json.dumps(DETERMINISTIC_CONTRACT, indent=2, sort_keys=True)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','output','work','weights','manifest','source','total-python','total-weights'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(argv)
    run(args)


if __name__=='__main__':main()
