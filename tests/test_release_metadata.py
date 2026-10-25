"""Release metadata agrees with the code version and stays free of site data."""
import json
import re
from pathlib import Path

import yaml

import rtseg_client
import rtseg_version

ROOT = Path(__file__).parents[1]


def test_version_agrees_across_release_metadata():
    version = rtseg_version.__version__
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    codemeta = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))
    assert str(citation["version"]) == version
    assert codemeta["version"] == codemeta["softwareVersion"] == version
    assert rtseg_client.CLIENT_VERSION == version
    assert re.search(rf"^## {re.escape(version)} — \d{{4}}-\d{{2}}-\d{{2}}$",
                     (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)


def test_zenodo_metadata_is_valid_and_licence_matches():
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert zenodo["upload_type"] == "software"
    assert zenodo["license"] == citation["license"] == "Apache-2.0"
    assert zenodo["creators"] and all(creator.get("name") for creator in zenodo["creators"])


def test_portal_reports_the_release_version(portal):
    from fastapi.testclient import TestClient

    assert TestClient(portal.app).get("/api/v1/config").json()["version"] == rtseg_version.__version__
