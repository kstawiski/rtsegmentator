"""Crop an image to the nonzero mask bounding box with a voxel margin."""
from pathlib import Path
import sys

import SimpleITK as sitk

image_path, mask_path, output_path = map(Path, sys.argv[1:4])
margin = [int(x) for x in (sys.argv[4:7] if len(sys.argv) >= 7 else (32, 32, 16))]
image, mask = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
if image.GetSize() != mask.GetSize():
    raise SystemExit("image and mask sizes differ")
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(mask > 0)
if not stats.HasLabel(1):
    raise SystemExit("crop mask is empty")
box = stats.GetBoundingBox(1)
start = [max(0, box[i] - margin[i]) for i in range(3)]
end = [min(image.GetSize()[i], box[i] + box[i + 3] + margin[i]) for i in range(3)]
cropped = sitk.RegionOfInterest(image, [end[i] - start[i] for i in range(3)], start)
output_path.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(cropped, str(output_path), True)
print({"start": start, "size": cropped.GetSize(), "spacing": cropped.GetSpacing()})
