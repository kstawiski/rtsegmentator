# Acceptance markers

These records gate portal availability for newly integrated model families.
They are copied into the private model bundle only after the named task has
completed inference and strict DICOM RTSTRUCT validation. A downloaded weight
directory by itself is therefore insufficient to make a task selectable.

These are historical records, not newly executed validation. Private fixture
identifiers and private artifact hashes are withheld. Numeric measurements
remain unchanged. Public audit fixtures are configured in
`catalog/orientation-audit-fixtures.json`; private reproduction uses an external
`RTSEG_ORIENTATION_FIXTURES` JSON file. Do not substitute a public fixture name
into a historical private-case result or enable a model solely by copying a marker.
