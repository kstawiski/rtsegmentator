#!/usr/bin/env python3
"""Derive public ROI contracts from exporter inputs without loading models or weights.

Use the same Python/TotalSegmentator installation as the worker, or supply
--totalsegmentator-root. The bundle is read only. Detailed provenance is written
only when --resolution-log is supplied, outside the public catalogue.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import difflib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from structure_labels import (INLINE_LABELS, PROMPTED_TASKS, REPO_LABEL_FILES, dataset_json,
                              dataset_labels, final_roi_names, kept_label, lut_labels,
                              repo_label_path, sct_labels, sct_output_name, spec_labels,
                              unique_roi_names)
from nv_segment_logic import MODES, generate
from prostate_models import TASKS as PROSTATE_TASKS, model_folder, validate_umu_dataset

INTERNAL_TOTAL = {"test", "total_highres_test", "total_v1", "vertebrae_pp", "vertebrae_pp_refined"}


def registries(root=ROOT):
    """Evaluate only app.py's metadata registry block, without importing the web app."""
    tree = ast.parse((root / "app.py").read_text())
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FALLBACK_TASKS" for t in n.targets))
    stop = next(i for i in range(start, len(tree.body)) if isinstance(tree.body[i], ast.FunctionDef))
    ns = {"__file__": str(root / "app.py"), "json": json, "Path": Path}
    exec(compile(ast.Module(tree.body[start:stop], type_ignores=[]), "app registries", "exec"), ns)
    return ns


def total_maps(package_root=None):
    if package_root is None:
        spec = importlib.util.find_spec("totalsegmentator")
        if spec is None or spec.origin is None:
            raise ValueError("TotalSegmentator metadata unavailable; supply --totalsegmentator-root")
        package_root = Path(spec.origin).parent
    package_root = Path(package_root)
    maps = runpy.run_path(str(package_root / "map_to_binary.py"))["class_map"]
    config = package_root / "map_tasks_config.py"
    configs = runpy.run_path(str(config))["TASK_CONFIGS"] if config.exists() else None
    tasks = {k for k in maps if (configs is None or k in configs)
             and k not in INTERNAL_TOTAL and not k.endswith("_auxiliary")}
    return maps, tasks, package_root / "map_to_binary.py"


def module_assignment(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise ValueError(f"Metadata assignment {name} unavailable in {path.name}")


def package_file(bundle, env, relative):
    paths = sorted((bundle / "envs" / env / "lib").glob(f"python*/site-packages/{relative}"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {env} package metadata file: {relative}")
    return paths[0]


def moose_dataset(bundle, task):
    path = package_file(bundle, "moose", "moosez/models.py")
    model = "clin_ct_dental" if task == "dentalsegmentator" else task.removeprefix("moose_")
    metadata = module_assignment(path, "MODEL_METADATA")
    entry = next(value for key, value in zip(metadata.keys, metadata.values) if ast.literal_eval(key) == model)
    folder = next(ast.literal_eval(value) for key, value in zip(entry.keys, entry.values)
                  if isinstance(key, ast.Name) and key.id == "KEY_FOLDER_NAME")
    directory = bundle / "weights/moose" / folder
    configurations = [p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name.count("__") == 2]
    if len(configurations) != 1:
        raise ValueError(f"Ambiguous or missing MOOSE configuration for {task}")
    return configurations[0] / "dataset.json", path


def fastsurfer_labels(source):
    """Include the hemisphere classes created by FastSurfer's postprocessor.

    Read the small inference LUT and the postprocessor's class-list/offset AST;
    importing data_utils would also import the model runtime. Refuse an unknown
    mapping expression instead of silently publishing an incomplete vocabulary.
    """
    base = source / "FastSurferCNN/config"
    vocabulary = lut_labels(base / "FastSurfer_ColorLUT.tsv", freesurfer=True)
    path = source / "FastSurferCNN/data_loader/data_utils.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "split_cortex_labels")
    assignment = next(n.value for n in fn.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "labels_list" for t in n.targets))
    classes = ast.literal_eval(assignment.args[0])
    loop = next(n for n in fn.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "label_current")
    writes = [n.value for n in ast.walk(loop) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                      and t.value.id == "aparc" for t in n.targets)]
    if (len(writes) != 1 or not isinstance(writes[0], ast.BinOp)
            or not isinstance(writes[0].op, ast.Add) or not isinstance(writes[0].left, ast.Name)
            or writes[0].left.id != "label_current"):
        raise ValueError("Unsupported FastSurfer cortical label split; review its output vocabulary")
    offset = ast.literal_eval(writes[0].right)
    indices = {int(k) for k in vocabulary} | {k + offset for k in classes}
    fix_loop = next(n for n in fn.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "prob_class_lh")
    fix_classes = ast.literal_eval(fix_loop.iter)
    fix_offset = next(n.value for n in fix_loop.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "prob_class_rh" for t in n.targets))
    if (not isinstance(fix_offset, ast.BinOp) or not isinstance(fix_offset.op, ast.Add)
            or not isinstance(fix_offset.left, ast.Name) or fix_offset.left.id != "prob_class_lh"
            or not set(fix_classes + [k + ast.literal_eval(fix_offset.right) for k in fix_classes]) <= indices):
        raise ValueError("Unsupported FastSurfer cortical correction; review its output vocabulary")
    all_writes = [ast.unparse(n.value) for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                          and t.value.id == "aparc" for t in n.targets)]
    if sorted(all_writes) != sorted([ast.unparse(writes[0]), "prob_class_lh", "prob_class_rh"]):
        raise ValueError("Unsupported FastSurfer output assignment; review its output vocabulary")
    output_names = lut_labels(base / "FreeSurferColorLUT.txt", freesurfer=True)
    return {str(k): output_names[str(k)] for k in sorted(indices)}, [
        str(base / "FastSurfer_ColorLUT.tsv"), str(path) + ": split_cortex_labels",
        str(base / "FreeSurferColorLUT.txt"), "labels_from_segmentation_lut.py: present-label subset"]


# SCT tasks whose final model has a single target: sct_deepseg then writes exactly
# the requested prediction.nii.gz (checked against the installed model config).
SCT_SINGLE_TARGET_MODELS = {"sct_tumor_t2": "t2_tumor"}


def fixed(labels, source, note=""):
    names = final_roi_names(labels)
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("ROI names collide ignoring case")
    return {"kind": "fixed", "names": names, "count": len(names), "source": source,
            "note": note or "Only structures found in the scan are written to the RTSTRUCT."}


def resolve(task, details, ns, bundle, maps, map_path, root=ROOT):
    """Return a public contract and private provenance; unavailable metadata raises."""
    if task in PROSTATE_TASKS:
        if task.startswith("picai_"):
            return {"kind": "variable", "names": ["csPCa_L01_p…"], "count": None,
                    "source": "PI-CAI baseline / RGA 0.2.4 indexed lesion candidates",
                    "note": "Zero to five candidates (maximum 5); names csPCa_L01_p0.73 through csPCa_L05_p…. "
                            "ROI scores are rounded to two decimals; JSON retains full precision. Empty results are valid."}, [
                                str(root / "prostate_postprocess.py") + ": extract_picai / roi_name"]
        model = PROSTATE_TASKS[task][0]
        path = root / "labels" / f"{model}.json"
        provenance = [str(path), "structure_labels.final_roi_names"]
        if task == "umu_prostate_zones_urethra_t2":
            metadata = bundle / model_folder(task) / "dataset.json"
            validate_umu_dataset(json.loads(metadata.read_text()))
            provenance.append(str(metadata))
        return fixed(json.loads(path.read_text()), "Prostate exporter label map"), provenance
    if details.get("output") == "report":
        return {"kind": "report", "names": [], "count": None, "source": "Report exporter",
                "note": "JSON, Markdown and HTML; no RTSTRUCT."}, ["run_task.py: nv_reason_ct_report"]
    if task in PROMPTED_TASKS:
        path = repo_label_path(task, root)
        names = final_roi_names(json.loads(path.read_text()))
        return {"kind": "prompted", "names": names, "count": None, "source": "Prompt target exporter label map",
                "note": "One ROI per prompt-selected target, named " + ", ".join(names) + ". Multiple seed points select the same target."}, [str(path), "structure_labels.repo_label_path", "structure_labels.final_roi_names"]
    if task in MODES:
        source = bundle / MODES[task]["source"]
        labels, _ = generate(source, task)
        inputs = [source / "configs/inference.json", source / "configs/metadata.json"]
        if (source / "configs/label_mappings.json").is_file():
            inputs.append(source / "configs/label_mappings.json")
        return fixed(labels, "NV-Segment selected channel definitions"), [*map(str, inputs), "nv_segment_logic.generate"]
    if task in REPO_LABEL_FILES:
        path = repo_label_path(task, root)
        note = ("All detected nodes share this aggregate ROI; individual nodes are not numbered."
                if task.startswith("lnq_") or task in {"compai_lnq", "dbdmp_lnq"} else "")
        return fixed(json.loads(path.read_text()), "Exporter label map", note), [str(path), "structure_labels.repo_label_path", "structure_labels.final_roi_names"]
    if task in INLINE_LABELS:
        return fixed(INLINE_LABELS[task], "Exporter label contract"), ["structure_labels.INLINE_LABELS", "structure_labels.final_roi_names"]
    for registry in ("CERR_TASKS", "ANTSPYNET_TASKS", "PYCAD_TASKS", "MRANNOTATOR_TASKS", "CADS_OPEN_TASKS", "CADS_RESEARCH_TASKS", "NNUNET_V1_TASKS", "MONAI_BUNDLE_TASKS"):
        if task in ns[registry]:
            spec = ns[registry][task]
            labels = spec_labels(spec, root, bundle)
            source = (bundle / "weights/monai-bundles" / spec["bundle"] / "configs/metadata.json" if spec.get("labels_from_metadata") else
                      root / spec["labels_file"] if spec.get("labels_file") else root / ("catalog/" + {"CERR_TASKS":"cerr", "ANTSPYNET_TASKS":"antspynet", "PYCAD_TASKS":"pycad-model-zoo", "MRANNOTATOR_TASKS":"mrannotator", "CADS_OPEN_TASKS":"cads-open", "CADS_RESEARCH_TASKS":"cads-research", "NNUNET_V1_TASKS":"nnunet-v1", "MONAI_BUNDLE_TASKS":"monai-bundles"}[registry] + ".json"))
            return fixed(labels, "MONAI output channel definitions" if spec.get("labels_from_metadata") else "Exporter registry label map"), [str(source), "structure_labels.spec_labels", "structure_labels.final_roi_names"]
    if task in ns["NNUNET_V2_TASKS"]:
        spec = ns["NNUNET_V2_TASKS"][task]
        if "keep_label" in spec:
            labels = kept_label(spec["keep_label"], spec.get("keep_name"))
            provenance = [str(root / "catalog/nnunet-v2.json"), "structure_labels.kept_label"]
        else:
            path = dataset_json(bundle / spec["model_folder"])
            labels = dataset_labels(json.loads(path.read_text()).get("labels", {}))
            provenance = [str(path), "structure_labels.dataset_json", "structure_labels.dataset_labels"]
        return fixed(labels, "nnU-Net export label map"), provenance
    if task in ns["MULTITALENT_V1_TASKS"]:
        path = root / "labels/multitalent-v1-masks.json"
        names = unique_roi_names(json.loads(path.read_text()).values())
        return fixed(dict(enumerate(names, 1)), "MultiTalent binary-mask exporter names"), [str(path), "run_multitalent_v1.py: mask names in insertion order", "structure_labels.unique_roi_names"]
    if task in ns["SCT_TASKS"]:
        spec = ns["SCT_TASKS"][task]
        if task in SCT_SINGLE_TARGET_MODELS:
            # SCT keeps the requested file name when its model has one target, so the
            # combiner sees a single prediction.nii.gz, not one file per alias.
            model = SCT_SINGLE_TARGET_MODELS[task]
            config = bundle / "runtime-homes/sct/sct_7.3/data/deepseg_models" / model / f"{model}.json"
            targets = json.loads(config.read_text())["loader_parameters"]["target_suffix"]
            if len(targets) != 1:
                raise ValueError(f"SCT model {model} now has {len(targets)} targets; update the SCT export contract.")
            name = sct_output_name("prediction.nii.gz", spec["output_names"], single=True)
            if name.startswith("SCT_Label_"):
                raise ValueError("SCT writes one mask that the exporter cannot name (it would be written as "
                                 + name + "); the export naming contract must be fixed before this model is enabled.")
            return fixed({"1": name}, "SCT exporter label map"), [str(config), "combine_sct_outputs.py: single output"]
        labels = sct_labels(spec, root)
        provenance = [str(root / spec.get("labels_file", "catalog/sct.json")), "structure_labels.sct_labels",
                      "structure_labels.sct_matched_name",
                      "run_task.py: preserve_overlaps / convert_masks_validate_rtstruct.py"
                      if spec.get("preserve_overlaps") else "combine_sct_outputs.py"]
        return fixed(labels, "SCT exporter label map"), provenance
    if task in ns["MOOSE_TASKS"] or task == "dentalsegmentator":
        path, metadata = moose_dataset(bundle, task)
        return fixed(dataset_labels(json.loads(path.read_text())["labels"]), "MOOSE checkpoint dataset labels"), [str(metadata), str(path), "structure_labels.dataset_labels"]
    if task in ns["GOUHFI_TASKS"]:
        path = bundle / "sources/GOUHFI" / ns["GOUHFI_TASKS"][task]["lut"]
        return fixed(lut_labels(path), "GOUHFI output lookup table"), [str(path), "structure_labels.lut_labels"]
    if task in ns["FASTSURFER_TASKS"]:
        labels, provenance = fastsurfer_labels(bundle / "sources/FastSurfer")
        return fixed(labels, "FastSurfer output classes after cortical hemisphere splitting"), provenance
    if task == "vibesegmentator":
        path = package_file(bundle, "segmentation-modern", "TPTBox/segmentation/VibeSeg/vibeseg.py")
        labels = ast.literal_eval(module_assignment(path, "VibeSeg_map"))
        return fixed(labels, "VIBESegmentator exporter class map"), [str(path), "VibeSeg_map"]
    if task.startswith("lnsegfm_"):
        path = bundle / "results/ln-seg-fm/labels.json"
        return fixed(json.loads(path.read_text()), "LN-Seg-FM exporter label map"), [str(path), "structure_labels.final_roi_names"]
    if task == "choi_tmli":
        from choi_ontology import confirmed_labels
        path = root / "labels/choi-tmli.json"
        return fixed(confirmed_labels(path), "Confirmed Choi exporter label map"), [str(path), "choi_ontology.confirmed_labels"]
    if task in ns["LNCTVSEG_TASKS"]:
        raise ValueError("No export label map: the additional checkpoint class and architecture-specific inference contract are undocumented; the nnU-Net omission policy is not assigned to this architecture.")
    if task in maps:
        return fixed(maps[task], "TotalSegmentator task class map"), [str(map_path), "class_map[" + task + "]", "keep_labels.py", "structure_labels.final_roi_names"]
    raise ValueError("No RTSTRUCT exporter label contract is configured; model status: " + details.get("status", "not integrated").replace("_", " ") + ".")


def build(bundle, total_root=None, root=ROOT):
    ns = registries(root)
    maps, total_tasks, map_path = total_maps(total_root)
    tasks = {name: {} for name in total_tasks | set(ns["FALLBACK_TASKS"])}
    tasks.update(ns["EXTERNAL_TASKS"])
    tasks.update(ns["INFORMATIONAL_MODELS"])
    catalog, log = {}, {}
    failures = []
    for task, details in sorted(tasks.items()):
        try:
            entry, provenance = resolve(task, details, ns, bundle, maps, map_path, root)
        except (OSError, ValueError, KeyError) as exc:
            # Only explicitly disabled entries may remain unresolved. Never turn
            # missing metadata into a plausible but invented structure list.
            if details.get("enabled", True) and not details.get("blocked_status"):
                failures.append(f"{task}: {exc}")
                continue
            reason = str(exc)
            # Missing file diagnostics belong in private provenance, not the JSON.
            if isinstance(exc, OSError):
                reason = "Required exporter metadata file is absent: " + Path(exc.filename or "metadata").name + "."
            if isinstance(exc, KeyError):
                reason = "Exporter label metadata is absent from the model registry (missing " + str(exc) + ")."
            entry = {"kind": "unresolved", "names": [], "count": None, "source": "Disabled model metadata",
                     "note": reason}
            provenance = [str(exc)]
        catalog[task] = entry
        log[task] = {"kind": entry["kind"], "count": entry["count"], "sources": provenance}
    if failures:
        raise ValueError("Enabled task resolution failed:\n" + "\n".join(failures))
    return catalog, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--totalsegmentator-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "catalog/structures.json")
    parser.add_argument("--resolution-log", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        catalog, log = build(args.bundle_root, args.totalsegmentator_root)
    except (ValueError, OSError, SystemExit) as exc:
        parser.exit(1, str(exc) + "\n")
    rendered = json.dumps(catalog, indent=2, ensure_ascii=True) + "\n"
    if args.resolution_log:
        args.resolution_log.write_text(json.dumps(log, indent=2) + "\n")
    if args.check:
        existing = args.output.read_text() if args.output.exists() else ""
        if existing != rendered:
            sys.stdout.writelines(difflib.unified_diff(existing.splitlines(True), rendered.splitlines(True),
                                                     fromfile="committed structures", tofile="resolved structures"))
            return 1
    else:
        args.output.write_text(rendered)
    print(f"{'Checked' if args.check else 'Generated'} {len(catalog)} tasks: " +
          ", ".join(f"{k}={v}" for k, v in sorted(Counter(e["kind"] for e in catalog.values()).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
