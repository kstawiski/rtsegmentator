"""Call pinned upstream numerical code; adapt only paths, ports and observability.

No model implementation is copied here. Each import uses the installed source.
U-Mamba's literal input/output paths are relocated with an audited AST transform.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

# Source and installed environments are immutable, including direct harness calls.
sys.dont_write_bytecode = True


def bound_operator(cls, **attributes):
    """Use an operator's methods without constructing the Holoscan scheduler."""
    obj = SimpleNamespace(**attributes)
    for name, member in vars(cls).items():
        if isinstance(member, staticmethod):
            setattr(obj, name, member.__func__)
        elif callable(member) and not name.startswith('__'):
            setattr(obj, name, MethodType(member, obj))
    return obj


class Ports:
    def __init__(self, **values):
        self.values = values

    def receive(self, name):
        return self.values[name]

    def emit(self, value, name):
        self.values[name] = value


def nci_image(path):
    import nibabel as nib
    import numpy as np
    from monai.deploy.core import Image
    path = Path(path)
    sdk_file = path.parent / 'sdk-images.json'
    role = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
    if sdk_file.is_file():
        meta = json.loads(sdk_file.read_text())[role]
        for key in ('nifti_affine_transform', 'dicom_affine_transform'):
            if key in meta:
                meta[key] = np.asarray(meta[key], dtype=float)
        return Image(np.load(path.parent / f'{role}.npy', allow_pickle=False), meta)
    image = nib.load(path)
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError('NCI input must be a finite scalar 3-D image')
    spacing = image.header.get_zooms()
    meta = {'nifti_affine_transform': image.affine.copy(),
            'row_pixel_spacing': float(spacing[0]),
            'col_pixel_spacing': float(spacing[1]),
            'depth_pixel_spacing': float(spacing[2])}
    fixture = path.parent / 'fixture.json'
    role = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
    if fixture.is_file():
        sdk = json.loads(fixture.read_text()).get('sdk_metadata', {}).get(role)
        if sdk:
            meta = {'nifti_affine_transform': np.asarray(sdk['nifti_affine_transform'], dtype=float),
                    'row_pixel_spacing': float(sdk['row_pixel_spacing']),
                    'col_pixel_spacing': float(sdk['col_pixel_spacing']),
                    'depth_pixel_spacing': float(sdk['depth_pixel_spacing'])}
    return Image(array.T, meta)


def nci_dicom_images(directories):
    """Use the unmodified SDK loader and volume operator on each selected role."""
    from monai.deploy.operators import DICOMDataLoaderOperator, DICOMSeriesToVolumeOperator
    from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries, SelectedSeries
    loader = bound_operator(DICOMDataLoaderOperator, _must_load=True,
                            _logger=logging.getLogger('nci-dicom'))
    converter = bound_operator(DICOMSeriesToVolumeOperator)
    result = {}
    for role, directory in directories.items():
        studies = loader.load_data_to_studies(Path(directory))
        if len(studies) != 1 or len(studies[0].get_all_series()) != 1:
            raise ValueError(f'NCI {role} must contain exactly one assessed DICOM series')
        study = studies[0]
        selection = StudySelectedSeries(study)
        selection.add_selected_series(SelectedSeries(role, study.get_all_series()[0]))
        result[role] = converter.convert_to_image([selection])
    return result


def save_nci_native(images, native):
    """Lossless SDK transport: original int16 arrays and double-precision metadata.

    NIfTI is a geometry/export companion. Inference never reads its rounded affine.
    """
    import nibabel as nib
    import numpy as np
    native = Path(native)
    native.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for role, image in images.items():
        np.save(native / f'{role}.npy', image.asnumpy(), allow_pickle=False)
        meta = image.metadata()
        nib.save(nib.Nifti1Image(image.asnumpy().T, meta['nifti_affine_transform']), native / f'{role}.nii.gz')
        metadata[role] = meta
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)
    (native / 'sdk-images.json').write_text(json.dumps(metadata, default=encode, allow_nan=False) + '\n')


def nci_crop_guard(lesion, images, output, dataset):
    """Use the affine initialisation performed by upstream compute() before saving."""
    import nibabel as nib
    import numpy as np
    selected = [images[k] for k in ('image_t2', 'image_adc', 'image_highb', 'image_organ_seg')]
    for image in selected:
        image.metadata()['affine'] = image.metadata()['nifti_affine_transform']
    lesion.convert_and_save(*selected, output)
    t2 = nib.as_closest_canonical(nib.load(output / 't2/t2.nii.gz'))
    gland = nib.as_closest_canonical(nib.load(output / 'organ/organ.nii.gz'))
    if t2.shape != gland.shape:
        raise ValueError('NCI gland/T2 shape mismatch; upstream would wait for interactive input')
    data = dataset(output, 'testing')[0]
    if not np.isfinite(data['image']).all() or any(size < 32 for size in data['image'].shape[1:]):
        raise ValueError('NCI normalization/window is undefined for this gland crop')
    return data['image'].shape


def nci_organ(image, weights, output):
    """The application's organ operator settings, without the graph scheduler."""
    import torch
    from monai.deploy.operators.monai_seg_inference_operator import (
        MonaiSegInferenceOperator, InMemImageReader, InfererType)
    organ_module = importlib.import_module('organ_seg_operator')
    organ = bound_operator(organ_module.ProstateSegOperator,
                          _input_dataset_key='image', _pred_dataset_key='pred')
    pre = organ.pre_process(InMemImageReader(image), str(output))
    post = organ.post_process(pre, str(output))
    model = torch.jit.load(str(weights / 'organ/model.ts'), map_location='cuda').eval()
    inference = bound_operator(MonaiSegInferenceOperator, _pre_transform=pre, _post_transforms=post,
        _input_dataset_key='image', _pred_dataset_key='pred', _inferer=InfererType.SLIDING_WINDOW,
        _roi_size=(128, 128, 16), sw_batch_size=4, overlap=0.5, model=model,
        _logger=logging.getLogger('nci-organ'))
    result = inference.compute_impl(image, None)
    del model, inference
    torch.cuda.empty_cache()
    return result


def prepare_nci_folds(output):
    """Pre-create the shared directory before upstream starts its five threads.

    A job may not reuse fold files. Their absence here and presence after the
    call establishes fresh outputs without relying on filesystem clock precision.
    """
    folder = Path(output) / 'lesion'
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f'fold{i}_lesion_prob.nii.gz' for i in range(5)]
    stale = [str(i) for i, path in enumerate(paths) if path.exists() or path.is_symlink()]
    if stale:
        raise ValueError('NCI refuses pre-existing fold probability files: folds ' + ', '.join(stale))
    return paths


def verify_nci_folds(paths):
    missing = [str(i) for i, path in enumerate(paths)
               if path.is_symlink() or not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError('NCI inference did not write fresh probability files for folds ' + ', '.join(missing))


def nci_pipeline(root, native, output):
    """Upstream SDK compute_impl and unchanged lesion/classifier compute methods.

    The classifier forward hook observes logits; it never alters them. A wrapper
    observes its preprocessed data to export the same 0.5-mm connected components.
    Empty lesion maps skip the upstream classifier, whose empty-crop path is invalid.
    """
    import nibabel as nib
    import numpy as np
    import torch
    from monai.deploy.operators.monai_seg_inference_operator import (
        MonaiSegInferenceOperator, InMemImageReader, InfererType)
    root, native, output = Path(root), Path(native), Path(output)
    source = root / 'sources/nci_monai_lesion/prostate-mri-lesion-seg/prostate_mri_lesion_seg_app'
    sys.path.insert(0, str(source))
    organ_module = importlib.import_module('organ_seg_operator')
    lesion_module = importlib.import_module('custom_lesion_seg_operator')
    classifier_module = importlib.import_module('custom_lesion_classifier_operator')
    output.mkdir(parents=True, exist_ok=True)
    weights = root / 'weights/nci_monai_lesion'
    images = {key: nci_image(native / f'{role}.nii.gz') for key, role in (
        ('image_t2', 't2w'), ('image_adc', 'adc'), ('image_highb', 'hbv'))}
    images['image_organ_seg'] = nci_organ(images['image_t2'], weights, output)
    if not np.any(images['image_organ_seg'].asnumpy()):
        raise ValueError('NCI organ model returned an empty gland; upstream lesion bounding box is undefined')
    attributes = dict(model_path=weights, output_folder=output,
        input_name_image_t2='image_t2', input_name_image_adc='image_adc',
        input_name_image_highb='image_highb', input_name_image_organ_seg='image_organ_seg',
        output_name_seg='seg_image', output_name_saved_images_folder='saved')
    lesion = bound_operator(lesion_module.ProstateLesionSegOperator, **attributes)
    # Guard the upstream interactive shape-error path, nonfinite normalization,
    # and <32-voxel windows before any lesion model forward. No numerical change.
    nci_crop_guard(lesion, images, output, lesion_module.SegmentationDataset)
    ports = Ports(**images)
    fold_paths = prepare_nci_folds(output)
    try:
        lesion.compute(ports, ports, None)
    except Exception:
        # Upstream hides thread exceptions; a subsequent merge may fail first.
        verify_nci_folds(fold_paths)
        raise
    verify_nci_folds(fold_paths)
    images['image_lesion_seg'] = ports.values['seg_image']
    probability = nib.load(output / 'lesion/merged_lesion_prob.nii.gz')
    lesion_mask = nib.load(output / 'lesion/lesion_mask.nii.gz')
    if not np.isfinite(probability.get_fdata()).all():
        raise ValueError('NCI produced nonfinite probabilities')
    observed = {'probability_path': str(output / 'lesion/merged_lesion_prob.nii.gz'),
                'gland_path': str(output / 'organ/organ.nii.gz'),
                'lesion_path': str(output / 'lesion/lesion_mask.nii.gz')}
    if not np.any(lesion_mask.get_fdata()):
        observed['empty_lesions'] = True
        return observed
    attributes['input_name_image_lesion_seg'] = 'image_lesion_seg'
    classifier = bound_operator(classifier_module.ProstateLesionClassifierOperator, **attributes)
    original_preprocess = classifier.preprocess
    def capture_preprocess(*args):
        data = original_preprocess(*args)
        if not np.isfinite(np.asarray(data['image'])).all():
            raise ValueError('NCI classifier normalization produced nonfinite values')
        if any(s < 64 for s in data['image'].shape[-3:]):
            raise ValueError('NCI classifier cannot construct its published 64-cubed crops')
        observed['classifier_data'] = data
        return data
    classifier.preprocess = capture_preprocess
    original_resnet = classifier_module.ResNet
    def observing_resnet(*args, **kwargs):
        net = original_resnet(*args, **kwargs)
        def observe(module, inputs, outputs):
            observed['classifier_logits'] = outputs.detach().cpu().numpy().copy()
        net.register_forward_hook(observe)
        return net
    classifier_module.ResNet = observing_resnet
    try:
        classifier.compute(Ports(**images), Ports(), None)
    finally:
        classifier_module.ResNet = original_resnet
    return observed


def relocated_code(source, replacements):
    """Replace exact string constants only; all numerical AST nodes stay intact."""
    tree = ast.parse(Path(source).read_text(), filename=str(source))
    changes = []
    class Relocate(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                changes.append({'line': node.lineno, 'literal': node.value, 'replacement': replacements[node.value]})
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node
    tree = Relocate().visit(tree)
    return compile(ast.fix_missing_locations(tree), str(source), 'exec'), changes


def load_relocated(name, path, replacements):
    import types
    module = types.ModuleType(name)
    module.__file__ = str(path)
    code, changes = relocated_code(path, replacements)
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module, changes


def umamba_pipeline(root, native, output):
    """Run the authors' complete crop and detector wrappers, relocating I/O only."""
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    import torch
    root, native, output = Path(root), Path(native), Path(output)
    base = root / 'sources/umambamtl/gc_algorithms/base_container'
    source = base / 'models/umamba_mtl'
    sys.path[:0] = [str(source), str(base / 'packages/umamba')]
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'input'
    weight_links = output / 'weights'
    weight_links.mkdir()
    for role in ('t2w', 'adc', 'hbv'):
        directory = inputs / role
        directory.mkdir(parents=True)
        # ITKReader/ITKWriter uses MHA, exactly as the Grand Challenge wrapper.
        sitk.WriteImage(sitk.ReadImage(str(native / f'{role}.nii.gz')), str(directory / f'{role}.mha'))
    for fold in range(5):
        (weight_links / f'f{fold}.ckpt').symlink_to(root / f'weights/umambamtl/detector/f{fold}.ckpt')
    (weight_links / 'prostate_weights.ckpt').symlink_to(root / 'weights/umambamtl/prostate_crop/prostate_weights.ckpt')
    prostate = output / 'images/prostate'
    detection = output / 'images/cspca-detection-map'
    prostate.mkdir(parents=True)
    detection.mkdir(parents=True)
    replacements = {
        '/input/images/transverse-t2-prostate-mri/': str(inputs / 't2w') + '/',
        '/input/images/transverse-adc-prostate-mri/': str(inputs / 'adc') + '/',
        '/input/images/transverse-hbv-prostate-mri/': str(inputs / 'hbv') + '/',
        '/input//images/transverse-hbv-prostate-mri/': str(inputs / 'hbv') + '/',
        '/output/images/prostate/': str(prostate) + '/',
        '/output/images/cspca-detection-map/': str(detection) + '/',
        '/output/cspca-case-level-likelihood.json': str(output / 'cspca-case-level-likelihood.json'),
        'weights/': str(weight_links) + '/',
    }
    crop, crop_changes = load_relocated('process_prostate', source / 'process_prostate.py', replacements)
    module, changes = load_relocated('rtseg_umamba_upstream', source / 'process.py', replacements)
    original_extract = module.extract_lesion_candidates
    captured = {'io_relocations': {'crop': crop_changes, 'detector': changes}}
    def observe(probability, **kwargs):
        captured['probability_xyz'] = probability.copy()
        result = original_extract(probability, **kwargs)
        captured['detection_xyz'] = result[0].copy()
        captured['candidates'] = result[1]
        captured['indexed_xyz'] = result[2].copy()
        return result
    module.extract_lesion_candidates = observe
    algorithm = module.csPCaAlgorithm()
    algorithm.predict()
    original_map = sitk.ReadImage(str(detection / 'cspca_detection_map.mha'))
    captured['detection_image'] = original_map
    captured['case_likelihood'] = json.loads((output / 'cspca-case-level-likelihood.json').read_text())
    return captured
