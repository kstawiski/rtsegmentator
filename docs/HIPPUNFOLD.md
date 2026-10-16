# HippUnfold local inference

The ADNI T1 task uses v1.5.3 with the bigbrain atlas. The Maguire T2 task uses
the v2 development image with multihist7. The collector accepts their final
`space-cropT1w` and `space-cropT2w` subfield maps, respectively. v1 omits the
`label-hipp` entity and v2 includes it. Each hemisphere is resampled with nearest
neighbour interpolation onto the model-input grid before combination. The
existing left/right label map is preserved. Missing or ambiguous hemispheres
stop conversion.

Before running Maguire T2, stage this file in the model bundle:

```text
runtime-homes/hippunfold-cache/model/Dataset001_HippUnfoldMag.nnUNetTrainer__nnUNetPlans__3d_fullres.tar.gz
```

[Zenodo record 18432751](https://zenodo.org/records/18432751) publishes
1,138,683,578 bytes and MD5 `fc2981e7b17b643b3326d3caab4912f7`. Verify both before
inference. The repository manifest additionally records SHA-256
`ee4e28d246cb1acb86df6b99be50c66b9d762356385643092877a9723a552de7`.
The runner checks that the archive exists and is nonempty; it does not compute
the checksum on every run. For a different host location, pass
`--model-archive /path/to/the/archive.tar.gz` to `run_hippunfold.py`.

Keep the pre-staged `template/CITI168`, `template/upenn`, and
`atlases/tpl-multihist7` directories under that cache, and the prebuilt
`runtime-homes/hippunfold-v2-overlay-dir`
beside it. The cache, archive, and v2 runtime overlay are mounted read-only.
All derived outputs and Snakemake work files go under the job work directory.

The v2 image's `download_nnunet_model` rule uses `storage(model_dict["url"])`.
Setting the cache directory alone does not remove its HTTP dependency. Only
the HTTP storage plugin is bundled, so a filesystem URL is insufficient.
`hippunfold_offline.smk` uses documented
[Snakemake module rule exclusions](https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html#modules)
to import the bundled processing rules without model, template, or atlas
download/copy rules. The staged resources become ordinary external inputs;
the model archive is no longer a temporary output that Snakemake can delete.
An incomplete resource cache fails with a local missing-input error. No
container or bundled workflow files are modified.

This adapter targets the bundled v2 workflow and its rule names. Check it
again when updating the image. CPU regression tests cover collection,
resampling, missing outputs, and command construction. They do not establish
GPU inference success or clinical readiness.
