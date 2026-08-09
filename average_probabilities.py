"""Average geometry-matched scalar probability NIfTI files."""
from pathlib import Path
import sys

import SimpleITK as sitk

output, *inputs = map(Path, sys.argv[1:])
if len(inputs) < 2:
    raise SystemExit("provide an output and at least two probability maps")
images = [sitk.ReadImage(str(path), sitk.sitkFloat32) for path in inputs]
reference = images[0]
for image in images[1:]:
    if image.GetSize() != reference.GetSize() or image.GetSpacing() != reference.GetSpacing():
        raise SystemExit("probability-map geometry mismatch")
average = sum(images[1:], images[0]) / len(images)
output.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(average, str(output), True)
print({"maps": len(images), "size": average.GetSize()})
