"""The generated model catalog page stays in step with the committed metadata."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_model_docs_page_is_current(portal):
    spec = importlib.util.spec_from_file_location("build_model_docs", ROOT / "scripts/build_model_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / "docs/models.md").read_text(encoding="utf-8") == module.build(portal), (
        "docs/models.md is stale; run python scripts/build_model_docs.py")
