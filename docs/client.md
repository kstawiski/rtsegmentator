# Python client

`rtseg_client.py` is a single-file client for the [automation API](API.md). It uses
only the Python standard library (Python 3.10 or newer), so it can be copied into
other projects; RTpipeline ships the same file.

## Command line

```bash
export RTSEG_URL=https://rtseg.example.org

python rtseg_client.py health
python rtseg_client.py models --enabled
python rtseg_client.py submit ./patient01 --model total --model lung_nodules \
    --modality CT --out ./results
python rtseg_client.py status JOB_ID
python rtseg_client.py download JOB_ID --out ./results --extract
python rtseg_client.py verify ./results/JOB_ID.zip
```

`submit` uploads every file below the given paths through a resumable upload
session, waits for the portal's series assessment and creates one job per selected
series. Select series with `--modality` (CT, MR or PT) or `--series-uid`; series the
portal marks unavailable are never selected. With `--out`, the client waits for each
job, downloads its ZIP and verifies it against the packaged manifest. The output is a
JSON summary with each job's models, outcomes and ROI counts.

Prompted models take `--prompt-json`, multi-sequence MRI models take `--roles`, and
the prostate SBRT crop takes `--postprocess prostate_sbrt_crop`; the formats are the
ones described in the [API reference](API.md).

## Authentication and TLS

RTsegmentator has no built-in user authentication. When the portal sits behind an
authenticating reverse proxy, pass the required headers with `--header "Name: value"`
or in the `RTSEG_CLIENT_HEADERS` environment variable (one `Name: value` per line).
Use `--ca-bundle` for a private certificate authority. Certificate verification
cannot be switched off.

## Verification

`verify` and `download` fail when the ZIP has no `rtsegmentator_manifest.json`, when
a listed file is missing or its size or SHA-256 differs, when the ZIP contains a file
the manifest does not list, or when a member name could escape the extraction
directory. `submit --out` and `download` also compare the packaged manifest with the
one the portal serves for the job. The manifest is not signed: these checks detect
corruption, incomplete transfers and a package that does not belong to the job, not
deliberate tampering by someone who can rewrite both the files and the manifest.

## Python use

```python
from pathlib import Path
import rtseg_client as rc

client = rc.RTsegClient("https://rtseg.example.org", headers={"Authorization": "Bearer ..."})
session = client.upload_files(rc.collect_files([Path("patient01")]))
upload = client.assess(session)
series = rc.select_series(upload["series"], modalities=["CT"])
batch = client.submit(upload["upload_id"], [s["key"] for s in series], ["total"])
for job in batch["jobs"]:
    client.wait(job["id"], poll=10)
    package = client.download(job["id"], Path("results") / f"{job['id']}.zip")
    manifest = rc.verify_result_zip(package)
    for model in manifest["models"]:
        print(model["name"], model["outcome"], (model.get("rtstruct") or {}).get("roi_names"))
```

Transient failures (HTTP 429, 502, 503, 504 and connection errors) are retried with
exponential back-off. A failed job raises `JobFailed` with the portal's error text.
