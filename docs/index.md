# RTsegmentator

RTsegmentator is a web portal and REST API that runs open segmentation models on
DICOM CT, MR and PET series and returns DICOM RTSTRUCT contours. It is research
software: review every contour before any clinical use.

A user uploads a study, picks a series and one or more models, and downloads a ZIP
with one RTSTRUCT per model. The same steps run unattended through the
[automation API](API.md) or the [Python client](client.md).

## What the portal does

- **Runs many models behind one interface.** The committed catalog describes 279
  tasks: TotalSegmentator tasks, nnU-Net and MONAI bundles, nodal, prostate, brain
  and PET models, prompted tools such as MedSAM, and an AI draft report for CT.
  The [model catalog](models.md) lists each task's modality, exported structures,
  licence and reference. A deployment enables only the tasks whose runtime, assets
  and orientation evidence are installed, and shows why each other task is disabled.
- **Exports contours that match the source images.** Masks are mapped directly onto
  the acquired DICOM grid. Before an RTSTRUCT is released, export checks the
  referenced SOP instances, the physical bounds and a contour-to-mask round trip
  (Dice ≥ 0.95, plane tolerance 0.05 voxel). See [technical checks](validation.md).
- **Describes every result.** Each result ZIP carries a
  [manifest](result-manifest.md) with the source series, each model's outcome and
  exact ROI names, its licence and limitation text, and a SHA-256 checksum for every
  file.
- **Feeds analysis pipelines.** [RTpipeline](rt-suite.md) sends a course's planning
  CT to the portal, verifies the returned package and uses the structures for
  dose-volume histograms and radiomics.

## Quick start

A single host with an NVIDIA GPU:

```bash
cp .env.example .env
# Point RTSEG_MODEL_BUNDLE and RTSEG_RUNTIME_BUNDLE at your licensed model assets.
docker compose build
docker compose up -d
curl -fsS http://localhost:8080/api/v1/health
```

Open `http://localhost:8080`. Model weights, runtimes and patient images are not part
of the repository or the base image; [deployment](DEPLOYMENT.md) and
[containers](CONTAINERS.md) describe the mounts and an SSH GPU worker. The portal has
no user authentication: put it behind authenticated TLS before it receives medical
data.

Submit a series from the command line once the portal is running:

```bash
python rtseg_client.py --url http://localhost:8080 submit ./CT_SERIES \
    --model total --out ./results
```

## Scope and limits

RTsegmentator checks that each RTSTRUCT is geometrically consistent with its source
images and that each model ran under its documented input contract. These checks do
not establish that a model's contours are anatomically or clinically correct for a
given patient, scanner or protocol. Model-specific restrictions are recorded in the
[model ledger](external-models.md) and are returned with every result.

The Apache-2.0 licence covers the original RTsegmentator source code. Each model's
code, weights and training data remain under their authors' terms.
