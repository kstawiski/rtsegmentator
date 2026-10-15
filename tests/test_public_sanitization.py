"""Scan tracked files and pending public additions; never scan Git's object store here."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).parents[1]
# Bracketed separators let the scanner inspect its own source without exemptions.
SITE = re.compile(
    r"/umed[-]projekty|/local[-]data|/home/(?:kon[r]ad)|cpd[-]kon[r]ad[-]worker|"
    r"\b(?:ar[g]os|kons[t]api|kon[r]ad|ksta[w]iski)\b|(?<![\w])s[1](?![\w])|"
    r"(?<![\d.])(?:100[.]64|10[.]\d{1,3}|172[.]\d{1,3})[.]\d{1,3}[.]\d{1,3}(?![\d.])|"
    r"https?://[^\s\"<>]*(?:next[c]loud|colab[.]research|tail[n]et|tail[s]cale|[.]ts[.]net)|"
    r"miednice[-]\d+|(?<!\d)(?:287[1]73|489[8]71)(?!\d)", re.I
)
# Existing generic container account, retained from the public baseline.
WORKER_HOME_SUFFIXES = {
    ".env.example": {"/jobs"}, "Dockerfile": {"", "/jobs", "/.ssh"},
    "app.py": {"/jobs"}, "compose.remote-worker.yaml": {"/jobs"},
    "docker/entrypoint.sh": {"/.ssh", "/.ssh/authorized_keys"},
    "docs/DEPLOYMENT.md": {"/jobs"},
}
WORKER_HOME = re.compile(r"/home/" + r"worker([^\s\"'`:<>()]*)")

SECRET = re.compile(r"\b(?:hf[_]|ghp[_]|github_pat[_]|sk[-])[A-Za-z0-9_-]{16,}|-----BEGI[N] (?:[A-Z ]*PRIVATE KEY|OPENSSH)")
MEDICAL_OR_ARCHIVE = re.compile(r"\.(?:dcm|dicom|nii(?:\.gz)?|nrrd|mha|mhd|raw|npz|npy|zip|tar(?:\.gz)?|tgz|7z|rar)$", re.I)


# One existing anatomical sacral-level label is not a deployment host.
ANATOMIC_LABEL = '"structures": "C1–C7, T1–T12, L1–L5 and S' + '1 vertebrae",'


# The published cuRAND package uses four version components, which resemble an IPv4 address.
NV_CURAND_PIN = "nvidia-curand==" + ".".join(("10", "4", "0", "35"))


def site_matches(name, line):
    if name == "requirements-nv-reason-ct.txt" and line.strip() == NV_CURAND_PIN:
        return []
    if name == "catalog/pycad-model-zoo.json" and line.lstrip("+- ").rstrip() == ANATOMIC_LABEL:
        return []
    return SITE.findall(line)


def public_files():
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, capture_output=True)
    return sorted({p for p in result.stdout.decode().split("\0") if p and (ROOT / p).is_file()})


def test_public_files_contain_no_site_data_or_secrets():
    failures = []
    for name in public_files():
        if MEDICAL_OR_ARCHIVE.search(name) or re.search(r"[.]bak|[.]pre[-]", name) or name == "HANDOFF.md":
            failures.append(f"forbidden artifact: {name}")
        data = (ROOT / name).read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected binary source file: {name}")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if site_matches(name, line) or SECRET.search(line):
                failures.append(f"forbidden content: {name}:{number}")
            for suffix in WORKER_HOME.findall(line):
                if suffix not in WORKER_HOME_SUFFIXES.get(name, set()):
                    failures.append(f"unapproved worker home: {name}:{number}")
    assert not failures, "\n".join(failures)


def test_cuda_version_exception_is_limited_to_one_requirement():
    assert not site_matches("requirements-nv-reason-ct.txt", NV_CURAND_PIN)
    assert site_matches("docs/DEPLOYMENT.md", NV_CURAND_PIN)
    assert site_matches("requirements-nv-reason-ct.txt", "# host " + NV_CURAND_PIN.split("==")[1])
