# Operations and safety

## Upload workflow

1. Upload loose DICOM files or an archive.
2. The portal reads DICOM metadata and groups images by Study, Series and modality.
3. Invalid, non-image and unrelated files are reported and ignored.
4. Select exactly one assessed image series.
5. Select compatible models. CT defaults to `total`; MR defaults to `total_mr`; PET has no TotalSegmentator default.
6. For prompted tools, choose a slice and add the required points or box.
7. Submit the job and download the resulting RTSTRUCT archive after completion.

Multiple series or modalities in one upload are never silently merged. Registration-dependent PET/CT workflows require a compatible CT in the same Study and Frame of Reference.

## Prompted models

- **PAM** — one positive point or axial box; propagates a target through the 3D volume.
- **SAT3D** — at least one positive 3D point; accepts negative points within its prompt-centred ROI.
- **SAM-Med2D** — point or box prompts are slice-wise. Prompt every slice that should be contoured.
- **BiomedParse** — represented in the catalog but unavailable until its approval-gated checkpoint is supplied.

Prompt coordinates are stored as `[column, row, slice]` and transferred with the selected series.

## RTSTRUCT acceptance gate

An automatic output is downloadable only when it satisfies all of these checks:

1. Prediction geometry matches the geometry-preserving model input.
2. Every nonzero label has an explicit ROI name.
3. The SOP Class is DICOM RT Structure Set Storage.
4. At least one ROI and one contour set are nonempty.
5. Every contour image reference belongs to the selected source DICOM series.

Empty candidate-detector results are reported as no candidates; the system does not fabricate contours.

## Privacy

- Treat the portal as handling protected medical information.
- Restrict port 8080 to trusted networks or place it behind authenticated TLS termination.
- Do not commit `data/`, DICOM, NIfTI, RTSTRUCT files, prompts from patient jobs or model licenses.
- Runtime jobs are retained only for the configured retention interval.
- Review filesystem permissions and backup policies on both hosts.

This repository's `.gitignore` excludes common medical-image formats, generated jobs, weights, checkpoints, credentials and virtual environments.

## GPU and job lifecycle

Jobs execute under a serialized worker lock. Each model runs in its own process group. Cancellation or failure terminates the child group, and successful jobs remove model-specific temporary volumes. Model processes exit after inference, releasing CUDA allocations.

Check active GPU processes:

```bash
ssh konrad@cpd-konrad-worker \
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## Health checks

```bash
systemctl is-active dicom-rt-portal.service
curl -fsS http://127.0.0.1:8080/api/tasks
curl -fsS http://127.0.0.1:8080/api/jobs
journalctl -u dicom-rt-portal.service -n 100 --no-pager
```

The model catalog should contain a description, structures and reference URL for every entry, including disabled models.

## Model validation evidence

See `EXTERNAL_MODELS.md` for fixture modality, checkpoint state, known domain limitations, voxel counts and RTSTRUCT validation results. A model being technically runnable does not imply clinical suitability for a different modality, disease, contrast phase or anatomical protocol.
