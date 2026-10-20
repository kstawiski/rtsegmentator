#!/usr/bin/env python3
"""nnU-Net v1 prediction with deterministic cuDNN after trainer restoration."""
from functools import wraps


CUDNN_POLICY = {'deterministic': True, 'benchmark': False,
                'applied_after': 'load_model_and_checkpoint_files (trainer restore and initialize)'}


def main():
    import torch
    from nnunet.inference import predict
    from nnunet.inference.predict_simple import main as predict_main

    restore = predict.load_model_and_checkpoint_files

    @wraps(restore)
    def deterministic_restore(*args, **kwargs):
        restored = restore(*args, **kwargs)
        # The checkpoint's training flag can reset these during restoration.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return restored

    predict.load_model_and_checkpoint_files = deterministic_restore
    try:
        # Leave sys.argv intact: the upstream parser receives the same options.
        return predict_main()
    finally:
        predict.load_model_and_checkpoint_files = restore


if __name__ == '__main__':
    main()
