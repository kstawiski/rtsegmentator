"""Read an NRRD payload independent of its filename and write NIfTI."""

import sys
import SimpleITK as sitk

reader = sitk.ImageFileReader()
reader.SetFileName(sys.argv[1])
reader.SetImageIO("NrrdImageIO")
sitk.WriteImage(reader.Execute(), sys.argv[2])
