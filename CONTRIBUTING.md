# Contributing

Run `python -m pytest -q` before proposing a change, and `mkdocs build --strict`
when documentation changes (`pip install -r requirements-docs.txt`). The browser
UI checks are described in the README.

New model integrations must document the upstream reference and licence, the
exact input modality and protocol, the label ontology, preprocessing, the expected
structures and DICOM RTSTRUCT validation evidence. Regenerate
`catalog/structures.json` and `docs/models.md` with the scripts named in the
[API documentation](docs/API.md) and `scripts/build_model_docs.py`. Keep
unavailable models visible with an explicit blocker instead of silently hiding
them.

Changes to `rtseg_client.py` or to the result manifest are interface changes:
RTpipeline vendors the client byte for byte and imports packages through the
manifest. Keep manifest changes backward compatible within schema version 1, or
publish a new major schema version.

Never commit patient data, generated clinical output, weights, access tokens,
license keys, virtual environments or offline payloads.

Participation is governed by the [code of conduct](CODE_OF_CONDUCT.md).
