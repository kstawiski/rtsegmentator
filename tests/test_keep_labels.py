import json

import nibabel as nib
import numpy as np

from keep_labels import main


def test_auxiliary_labels_removed_without_geometry_change(tmp_path):
    path, labels = tmp_path / "prediction.nii.gz", tmp_path / "labels.json"
    data = np.array([0, 1, 2, 99, 2, 1, 0, 99], dtype=np.int16).reshape(2, 2, 2)
    affine = np.diag([-1., -2., 3., 1.])
    nib.save(nib.Nifti1Image(data, affine), path)
    labels.write_text(json.dumps({"1": "one", "2": "two"}))
    main(path, labels)
    image = nib.load(path)
    np.testing.assert_array_equal(np.asarray(image.dataobj), np.where(data == 99, 0, data))
    np.testing.assert_array_equal(image.affine, affine)
    before = path.read_bytes()
    main(path, labels)
    assert path.read_bytes() == before
