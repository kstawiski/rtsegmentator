"""Execute the copied official MONAI config and capture inference memory metrics."""
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import monai
import torch
from monai.bundle import ConfigWorkflow

p=argparse.ArgumentParser(); p.add_argument('--load-only',action='store_true');a=p.parse_args()
start=time.monotonic()
w=ConfigWorkflow(config_file='configs/inference.json',workflow_type='infer',logging_file=False)
try:
    w.initialize()
    network=w.parser.get_parsed_content('network')
    print(json.dumps({'torch':torch.__version__,'monai':monai.__version__,
        'checkpoint_load':'strict','state_tensors':len(network.state_dict()),'load_seconds':time.monotonic()-start}),flush=True)
    if not a.load_only: w.run()
finally:
    w.finalize()
metrics={'elapsed_seconds':time.monotonic()-start,'load_only':a.load_only,
         'cuda_available':torch.cuda.is_available(),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
         'peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
Path('bundle_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics),flush=True)
