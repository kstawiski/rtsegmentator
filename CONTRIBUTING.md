# Contributing

Run `python -m pytest` before proposing a change. New model integrations must
document the upstream reference and license, exact input modality and protocol,
label ontology, preprocessing, expected structures and DICOM RTSTRUCT validation
evidence. Keep unavailable models visible with an explicit blocker instead of
silently hiding them.

Never commit patient data, generated clinical output, weights, access tokens,
license keys, virtual environments or offline payloads.
