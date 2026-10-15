"""Validate clinician-confirmed Choi label names before any named export."""
import json
from pathlib import Path

BILATERAL = ('Cervical', 'Axillary', 'Common_Iliac', 'External_Iliac',
             'Internal_Iliac', 'Obturator', 'Inguinal')
REGIONS = tuple(f'{region}_{side}' for region in BILATERAL for side in ('Left', 'Right')) + (
    'Superior_Mediastinal', 'Inferior_Mediastinal', 'Para_Aortic', 'Presacral')


def confirmed_labels(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 1 or data.get('confirmed') is not True:
        raise ValueError('Choi label mapping requires clinician confirmation')
    labels = data.get('labels')
    if not isinstance(labels, dict) or set(labels) != {str(i) for i in range(1, 19)}:
        raise ValueError('Choi mapping must cover every label 1–18')
    if any(not isinstance(v, str) for v in labels.values()) or set(labels.values()) != set(REGIONS):
        raise ValueError('Choi mapping must assign the 18 distinct published regions exactly once')
    if any(not isinstance(data.get(key), str) or not data[key].strip()
           for key in ('reviewer', 'reviewed_utc')):
        raise ValueError('Choi mapping requires a reviewer and review date')
    evidence = data.get('evidence')
    if not isinstance(evidence, list) or not evidence or any(not isinstance(x, str) or not x.strip() for x in evidence):
        raise ValueError('Choi mapping requires review evidence identifiers')
    return labels
