# Public release checklist

- Choose and add an explicit license for this repository; do not infer that upstream model licenses apply to portal code.
- Review every bundled label/config file for its upstream license and attribution requirements.
- Keep `data/`, DICOM, RTSTRUCT, prompts, logs, credentials, license keys, model weights, and checkpoints out of source control and container layers.
- Scan Git history as well as the working tree for secrets and medical data before creating a remote repository.
- Generate an SBOM and vulnerability scan for each published image.
- Publish image and offline-bundle checksums and sign release artifacts.
- Document supported GPUs, minimum VRAM, tested driver versions, and model-specific resource requirements.
- Put production deployments behind authenticated TLS; the application is not a multi-tenant identity system.
- Preserve the research-use warning and model-specific clinical/domain limitations.
