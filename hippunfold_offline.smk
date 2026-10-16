# The v2 image wraps its model URL in storage(), which checks HTTP existence
# while building the DAG. Import only inference/processing rules. Resources
# under HIPPUNFOLD_CACHE_DIR must already exist and remain external inputs.
# Mount beside the bundled Snakefile: some rules also use workflow.basedir
# for resources. Module imports preserve relative script/conda paths.
module hippunfold:
    snakefile: "/src/hippunfold/workflow/Snakefile"
    config: config

use rule * from hippunfold exclude download_nnunet_model, download_extract_template, download_surf_template_atlas, cp_atlas_surf_gii, cp_atlas_metric_gii
