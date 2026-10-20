"""CPU checks for the offline UI and the presentation-only model catalog."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_api import load_app

ROOT = Path(__file__).parents[1]
STATIC = ROOT / "static"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and "src" in attrs:
            self.assets.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.assets.append(attrs["href"])


def test_home_and_static_assets_are_served_from_any_working_directory(tmp_path, monkeypatch):
    app = load_app(tmp_path)
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert page.text == (STATIC / "index.html").read_text()
        assert "Review &amp; start" in page.text
        assert "Select all CT and MR" in page.text
        assert 'id="catalogPicker" aria-label="All compatible models"' in page.text
        parser = PageParser()
        parser.feed(page.text)
        assert set(parser.assets) == {"/static/app.js", "/static/viewer.js", "/static/app.css"}
        for name, content_type in [("app.js", "javascript"), ("viewer.js", "javascript"), ("app.css", "text/css"),
                                   ("ui-catalog.json", "application/json")]:
            response = client.get("/static/" + name)
            assert response.status_code == 200
            assert content_type in response.headers["content-type"]
            assert response.content == (STATIC / name).read_bytes()
        assert client.get("/static/missing.css").status_code == 404
        assert client.get("/static/%2e%2e/app.py").status_code == 404


def test_static_files_have_no_external_resources():
    for path in STATIC.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".js", ".css", ".json"}:
            text = path.read_text()
            assert not re.search(r"https?://|(?:src|href)\s*=\s*['\"]//|@import", text), path.name


def test_interface_font_is_bundled_with_its_licence(tmp_path):
    css = (STATIC / "app.css").read_text()
    fonts = re.findall(r"url\(([^)]+)\)", css)
    assert fonts == ["fonts/open-sans-latin-ext-wght-normal.woff2", "fonts/open-sans-latin-wght-normal.woff2"]
    assert "SIL Open Font License" in (STATIC / "fonts" / "OFL.txt").read_text()
    with TestClient(load_app(tmp_path)) as client:
        for name in fonts:
            response = client.get("/static/" + name)
            assert response.status_code == 200
            assert response.content == (STATIC / name).read_bytes()
            assert response.content[:4] == b"wOF2"


def test_every_script_dom_id_exists_and_ids_are_unique():
    parser = PageParser()
    parser.feed((STATIC / "index.html").read_text())
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate HTML IDs"
    script = (STATIC / "app.js").read_text()
    # All static ID lookups use the documented $ helper; dynamic rows use data attributes.
    referenced = set(re.findall(r"\$\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", script))
    referenced |= set(re.findall(r"getElementById\(\s*['\"]([^'\"]+)['\"]\)", script))
    assert len(referenced) > 80
    assert referenced <= set(parser.ids), referenced - set(parser.ids)
    assert not re.search(r"querySelector(?:All)?\(['\"]#", script)
    assert "HTML = r'''" not in (ROOT / "app.py").read_text()


def test_presentation_covers_committed_tasks_and_unknown_runtime_tasks(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable; presentation execution check needs Node")
    load_app(tmp_path)
    module = sys.modules["app"]
    tasks = [{"name": name, "modality": module.task_modality(name)} for name in module.FALLBACK_TASKS]
    tasks += [{"name": name, **details} for source in (module.EXTERNAL_TASKS, module.INFORMATIONAL_MODELS)
              for name, details in source.items()]
    tasks += [{"name": "future_worker_model", "modality": "CT"}]
    code = """
const fs = require('node:fs');
const ui = require('./static/app.js');
const metadata = require('./static/ui-catalog.json');
const tasks = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const task of tasks) {
  const info = ui.presentation(task, metadata);
  if (!ui.GROUPS.some(([id]) => id === info.group) || !info.title || !info.purpose || info.title.includes('_'))
    throw Error('Missing presentation: ' + task.name);
}
if (ui.presentation(tasks.at(-1), metadata).group !== 'other') throw Error('Missing explicit fallback');
console.log(tasks.length + ' task presentations checked, including unknown runtime fallback');
"""
    result = subprocess.run([node, "-e", code], input=json.dumps(tasks), text=True,
                            cwd=ROOT, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_javascript_contract_checks():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    for source in ("static/app.js", "static/viewer.js", "tests/ui_fixture.js", "tests/ui_browser_checks.js"):
        check = subprocess.run([node, "--check", source], cwd=ROOT, capture_output=True, text=True)
        assert check.returncode == 0, check.stdout + check.stderr
    result = subprocess.run([node, "--test", "tests/test_ui_logic.cjs"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_presentation_metadata_has_no_execution_overrides():
    data = json.loads((STATIC / "ui-catalog.json").read_text())
    assert data["fallback"]
    for name, item in data["overrides"].items():
        assert set(item) <= {"group", "title", "purpose"}, name


def test_primary_actions_viewers_results_and_secondary_tools_contract():
    html = (STATIC / "index.html").read_text()
    parser = PageParser()
    parser.feed(html)
    required = {"segmentAction", "reportAction", "catalogPicker", "restoreDefaults", "selectAllSeries", "clearSeries", "seriesCount", "seriesViewer", "batchSiblings",
                "jobDetailPage", "jobViewer", "reportPanel", "structuresPanel", "downloadsPanel",
                "reportTab", "structuresTab", "downloadsTab", "deleteDialog", "confirmDelete", "moreTools"}
    assert required <= set(parser.ids)
    assert "Choose models" in html
    assert "Choose other models" not in html
    assert "Segment structures" in html and "NV-Reason-CT" in html
    assert 'id="reportAction" aria-describedby="reportDescription reportReason"' in html
    assert re.search(r'<details id="moreTools"[^>]*>.*?href="#tools"', html)
    for name in ("report", "structures", "downloads"):
        assert f'aria-controls="{name}Panel"' in html
        assert f'aria-labelledby="{name}Tab"' in html
    script = (STATIC / "viewer.js").read_text()
    assert "ctx.fill('evenodd')" in script
    assert 'role="img"' in script and 'aria-label="DICOM slice"' in script


def test_catalog_licence_badges_have_separate_key_and_noncommercial_flags(tmp_path):
    load_app(tmp_path)
    module = sys.modules["app"]
    entries = {**module.EXTERNAL_TASKS, **module.INFORMATIONAL_MODELS}
    # The licensed flag is reserved for runtime TotalSegmentator commercial tasks.
    assert not [name for name, item in entries.items() if item.get("licensed")]
    noncommercial = {
        "nv_segment_ctmr_ct_body", "nv_segment_ctmr_mri_body", "nv_segment_ctmr_mri_brain",
        "medsam2_ct_lesion_prompted", "medical_sam2_prompted_ct", "nninteractive_v1_prompted",
        "multitalent_v1_unet_ct", "multitalent_v1_resenc_ct", "lnq_atlas",
    }
    for name in noncommercial:
        assert entries[name]["noncommercial"] is True, name
    ct = entries["nv_segment_ct_auto"]
    assert ct["licensed"] is False
    assert not ct.get("noncommercial")
    assert ct["license_note"] == "NVIDIA Open Model License."
    # Research-draft presentation wording is not itself a weight licence restriction.
    assert not entries["nv_reason_ct_report"].get("noncommercial")
    for name, item in entries.items():
        notes = " ".join(str(item.get(key, "")) for key in ("license", "license_note", "description"))
        if re.search(r"non[ -]?commercial|CC BY-NC|research(?:/education)? use only", notes, re.I):
            assert item.get("noncommercial") is True, name
