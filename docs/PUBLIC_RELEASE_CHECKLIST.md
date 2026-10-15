# Public release checklist

- Keep the repository LICENSE and review upstream model/config licensing separately; a portal license does not grant model rights.
- Review every bundled label/config file for its upstream license and attribution requirements.
- Keep `data/`, DICOM, RTSTRUCT, prompts, logs, credentials, license keys, model weights, and checkpoints out of source control and container layers.
- Scan Git history as well as the working tree for secrets and medical data before creating a remote repository.
- Generate an SBOM and vulnerability scan for each published image.
- Publish image and offline-bundle checksums and sign release artifacts.
- Document supported GPUs, minimum VRAM, tested driver versions, and model-specific resource requirements.
- Put production deployments behind authenticated TLS; the application is not a multi-tenant identity system.
- Preserve the research-use warning and model-specific clinical/domain limitations.

- Run the forbidden-content test and scan the complete Git history; a clean current tree does not clean earlier commits.
- Keep private fixture maps and artifact identifiers outside the repository. Use public or synthetic fixtures in committed tests.
- Verify model availability and geometry on the target installation. Local source checks do not establish clinical or manuscript readiness.
