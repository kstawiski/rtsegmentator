"""Scan tracked files and pending public additions; never scan Git's object store here."""
from pathlib import Path
import hashlib
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
CATALOG_SACRAL_NAME = '"vertebrae S' + '1"'
NV_SACRAL_LABEL = '"127": "vertebrae S' + '1"'


# The published cuRAND package uses four version components, which resemble an IPv4 address.
NV_CURAND_PIN = "nvidia-curand==" + ".".join(("10", "4", "0", "35"))


# The bundled interface font (SIL Open Font License, static/fonts/OFL.txt) is the only binary source, pinned by content.
APPROVED_BINARIES = {
    "static/fonts/open-sans-latin-wght-normal.woff2": "d8e4fe0452aa2076429a9bb5d8757d00a994dd95986cf950e9a1a371b9a072a0",
    "static/fonts/open-sans-latin-ext-wght-normal.woff2": "d5bab8e28732fe3d10dcef4f77b9c248605bbb2a87d289a2539251ceafab536a",
}


# The suite's public repository and documentation addresses name the GitHub account that
# publishes them, not a deployment host or path. Only these exact URL prefixes are exempt.
PUBLIC_SUITE_URL = re.compile(
    r"https://(?:github[.]com/ksta[w]iski|ksta[w]iski[.]github[.]io)/(?:rtsegmentator|rtpipeline)"
    r"(?![\w-])[^\s\"'<>)]*"
)


def site_matches(name, line):
    line = PUBLIC_SUITE_URL.sub("", line)
    if name == "catalog/structures.json" and line.strip().rstrip(",") == CATALOG_SACRAL_NAME:
        return []
    if name in {"labels/nv_segment_ct_auto.json", "labels/nv_segment_ctmr_ct_body.json"} and line.strip() == NV_SACRAL_LABEL:
        return []
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
            if hashlib.sha256(data).hexdigest() != APPROVED_BINARIES.get(name):
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


def test_nv_sacral_label_exception_is_limited_to_exact_label_files():
    assert not site_matches("labels/nv_segment_ct_auto.json", NV_SACRAL_LABEL)
    assert not site_matches("labels/nv_segment_ctmr_ct_body.json", NV_SACRAL_LABEL)
    assert site_matches("docs/OPERATIONS.md", NV_SACRAL_LABEL)
    assert site_matches("labels/nv_segment_ct_auto.json", NV_SACRAL_LABEL + ' host')


def test_binary_exception_is_limited_to_the_pinned_font_files():
    assert set(APPROVED_BINARIES) <= set(public_files())
    for name, digest in APPROVED_BINARIES.items():
        assert name.startswith("static/fonts/") and name.endswith(".woff2")
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_structure_catalog_sacral_exception_is_limited_to_an_exact_anatomical_name():
    assert not site_matches("catalog/structures.json", "      " + CATALOG_SACRAL_NAME + ",")
    assert site_matches("docs/API.md", CATALOG_SACRAL_NAME)
    assert site_matches("catalog/structures.json", CATALOG_SACRAL_NAME + " host")
    assert site_matches("catalog/structures.json", '"note": ' + CATALOG_SACRAL_NAME)


def test_public_suite_url_exception_is_limited_to_the_two_repositories():
    account = "ksta" + "wiski"
    for url in (f"https://github.com/{account}/rtpipeline", f"https://{account}.github.io/rtsegmentator/api/",
                f"https://github.com/{account}/rtsegmentator/releases"):
        assert not site_matches("README.md", url)
    for text in (f"https://github.com/{account}/other", f"https://github.com/{account}/rtpipeline-private",
                 f"/home/{account}/rtpipeline", f"{account}@host", f"https://hub.docker.com/r/{account}/x"):
        assert site_matches("README.md", text)
