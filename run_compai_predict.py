"""Run the trusted CompAI nnU-Net release on PyTorch 2.6 and newer."""

import torch


release_torch_load = torch.load


def load_release_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return release_torch_load(*args, **kwargs)


torch.load = load_release_checkpoint

from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder


if __name__ == "__main__":
    predict_entry_point_modelfolder()
