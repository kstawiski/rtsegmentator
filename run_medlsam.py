"""Run MedLSAM with the released StructSeg head-and-neck support bank."""

from __future__ import annotations

import argparse
import configparser
import importlib
import json
import os
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--medlam", required=True)
    parser.add_argument("--medsam", required=True)
    parser.add_argument("--support-root", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--support-cases", default="1,2,3")
    parser.add_argument("--classes", default="1-22")
    args = parser.parse_args()

    required = (args.input, args.source, args.medlam, args.medsam, args.support_root)
    for path in required:
        if not Path(path).exists():
            raise SystemExit(f"required MedLSAM path does not exist: {path}")
    cases = [int(x) for x in args.support_cases.split(",") if x.strip()]
    if not 3 <= len(cases) <= 10:
        raise SystemExit("MedLSAM requires 3-10 fixed support cases")
    classes: list[int] = []
    for part in args.classes.split(","):
        bounds = [int(x) for x in part.split("-")]
        classes.extend(range(bounds[0], bounds[-1] + 1))
    if not classes or any(x < 1 or x > 22 for x in classes):
        raise SystemExit("MedLSAM classes must be within 1-22")

    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    import cv2  # noqa: PLC0415
    import SimpleITK as sitk  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from skimage import transform  # noqa: PLC0415
    import MedLAM.detection_functions as detection  # noqa: PLC0415
    from MedLAM.Anatomy_detection import AnatomyDetection  # noqa: PLC0415
    # Import the model builder without executing the upstream package
    # initializer. It imports the optional predictor, which imports a broken
    # torchvision binary in the isolated runtime even though inference does
    # not use torchvision.
    package = types.ModuleType("segment_anything")
    package.__path__ = [str(source / "MedSAM" / "segment_anything")]
    sys.modules["segment_anything"] = package
    sam_model_registry = importlib.import_module(
        "segment_anything.build_sam"
    ).sam_model_registry

    # The release returns length-one arrays where scalar coordinates are
    # required. NumPy 2 rejects the implicit conversion; preserve the selected
    # voxel exactly while returning Python scalars.
    def scalar_center(image=None, *, img=None):
        image = img if image is None else image
        indices = np.nonzero(image)
        middle = len(indices[0]) // 2
        return int(indices[0][middle]), int(indices[1][middle])

    detection.get_center_cor = scalar_center

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    support_root = Path(args.support_root).resolve()
    image_paths = [support_root / str(case) / "data.nii.gz" for case in cases]
    label_paths = [support_root / str(case) / "label.nii.gz" for case in cases]
    if not all(path.is_file() for path in image_paths + label_paths):
        raise SystemExit("MedLSAM support bank is incomplete")
    image_list = work / "support-images.txt"
    label_list = work / "support-labels.txt"
    image_list.write_text("\n".join(map(str, image_paths)) + "\n")
    label_list.write_text("\n".join(map(str, label_paths)) + "\n")
    config = configparser.ConfigParser()
    config["data"] = {
        "support_image_ls": str(image_list),
        "support_label_ls": str(label_list),
        "query_image_ls": str(work / "unused-query-images.txt"),
        "query_label_ls": str(work / "unused-query-labels.txt"),
        "gt_slice_threshold": "10",
        "fg_class": json.dumps(classes),
        "bbox_mode": "WPL",
        "slice_interval": "False",
        "feature_scale_ls": "[feature0, feature1, feature2]",
    }
    config["weight"] = {"medlam_load_path": str(Path(args.medlam).resolve())}
    config_path = work / "config.ini"
    with config_path.open("w") as handle:
        config.write(handle)

    locator = AnatomyDetection(str(config_path))
    _, corners, original_shape = locator.get_extreme_corner(str(Path(args.input).resolve()))
    missing = sorted(set(classes) - {int(x) for x in corners})
    if missing:
        raise SystemExit(f"MedLSAM failed to localize classes: {missing}")

    sam = sam_model_registry["vit_b"](checkpoint=str(Path(args.medsam).resolve())).cuda().eval()
    image_sitk = sitk.ReadImage(args.input)
    image_zyx = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    clipped = np.clip(image_zyx, -500, 1000)
    denominator = float(clipped.max() - clipped.min())
    normalized = ((clipped - clipped.min()) / max(denominator, 1e-8) * 255).astype(np.uint8)
    normalized[image_zyx == 0] = 0
    best = np.zeros(image_zyx.shape, dtype=np.float32)
    output_zyx = np.zeros(image_zyx.shape, dtype=np.uint8)
    original_zyx = np.asarray(original_shape)[::-1]

    prepared: dict[int, tuple[np.ndarray, torch.Tensor]] = {}

    def embedding(slice_index: int):
        if slice_index not in prepared:
            resized = transform.resize(normalized[slice_index], (1024, 1024), order=2,
                                       preserve_range=True, mode="constant", anti_aliasing=True)
            rgb = np.repeat(np.uint8(resized)[..., None], 3, axis=-1)
            # The slice and its box are already expressed in MedSAM's native
            # 1024 x 1024 coordinate system, so ResizeLongestSide is identity.
            tensor = torch.as_tensor(rgb.transpose(2, 0, 1), device="cuda")
            model_input = sam.preprocess(tensor[None])
            with torch.inference_mode():
                encoded = sam.image_encoder(model_input)
            prepared[slice_index] = (rgb, encoded)
        return prepared[slice_index]

    localized = {}
    for class_id in classes:
        box3d = np.asarray(corners[class_id][0], dtype=np.float64)
        z0, z1 = sorted((int(round(box3d[0, 0])), int(round(box3d[1, 0]))))
        x0, x1 = sorted((box3d[0, 2] * 1024 / original_zyx[2],
                         box3d[1, 2] * 1024 / original_zyx[2]))
        y0, y1 = sorted((box3d[0, 1] * 1024 / original_zyx[1],
                         box3d[1, 1] * 1024 / original_zyx[1]))
        box2d = np.asarray([max(0, x0 - 10), max(0, y0 - 10),
                            min(1023, x1 + 10), min(1023, y1 + 10)])
        localized[str(class_id)] = {"z": [z0, z1], "box_1024": box2d.tolist()}
        for z in range(max(0, z0), min(image_zyx.shape[0] - 1, z1) + 1):
            rgb, encoded = embedding(z)
            box_tensor = torch.as_tensor(box2d, dtype=torch.float32,
                                         device="cuda").reshape(1, 1, 4)
            with torch.inference_mode():
                sparse, dense = sam.prompt_encoder(points=None, boxes=box_tensor, masks=None)
                probability, _ = sam.mask_decoder(
                    image_embeddings=encoded,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                probability = torch.sigmoid(probability)
                probability = torch.nn.functional.interpolate(
                    probability, size=(1024, 1024), mode="bilinear", align_corners=False
                )[0, 0].cpu().numpy()
            probability = cv2.resize(probability, (image_zyx.shape[2], image_zyx.shape[1]),
                                     interpolation=cv2.INTER_LINEAR)
            update = (probability > 0.5) & (probability > best[z])
            best[z][update] = probability[update]
            output_zyx[z][update] = class_id

    if not np.any(output_zyx):
        raise SystemExit("MedLSAM returned an empty segmentation")
    source_nifti = nib.load(args.input)
    output_xyz = output_zyx.transpose(2, 1, 0)
    if output_xyz.shape != source_nifti.shape:
        raise SystemExit(f"MedLSAM output shape {output_xyz.shape} != input {source_nifti.shape}")
    header = source_nifti.header.copy(); header.set_data_dtype(np.uint8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(output_xyz, source_nifti.affine, header), output)
    counts = {str(x): int(np.sum(output_xyz == x)) for x in classes}
    (work / "localization.json").write_text(json.dumps(localized, indent=2) + "\n")
    print(json.dumps({"support_cases": cases, "classes": classes,
                      "nonempty_classes": sum(value > 0 for value in counts.values()),
                      "voxel_counts": counts, "encoded_slices": len(prepared)}))


if __name__ == "__main__":
    main()
