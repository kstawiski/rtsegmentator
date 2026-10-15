import threading


def test_cache_returns_old_value_during_refresh(portal, monkeypatch):
    values = iter([{"tasks": [{"name": "first"}]}, {"tasks": [{"name": "second"}]}])
    entered, finish = threading.Event(), threading.Event()
    calls = []
    def build():
        calls.append(1)
        if len(calls) > 1:
            entered.set()
            assert finish.wait(5)
        return next(values)
    monkeypatch.setattr(portal, "build_catalog", build)
    assert portal.tasks()["tasks"][0]["name"] == "first"
    assert len(calls) == 1
    assert portal.tasks()["tasks"][0]["name"] == "first"
    portal.catalog_state["built"] = 0
    try:
        assert portal.tasks()["tasks"][0]["name"] == "first"
        assert entered.wait(5)
        assert not portal.refresh_catalog_in_background()
    finally:
        finish.set()
    # Join only this refresh: a completion event avoids leaving a writer behind.
    import time
    deadline = time.monotonic() + 5
    while portal.catalog_state["refreshing"] and time.monotonic() < deadline:
        time.sleep(.01)
    assert portal.tasks()["tasks"][0]["name"] == "second"
    assert len(calls) == 2


def test_enabled_file_and_explicit_names_are_combined(portal, monkeypatch, tmp_path):
    path = tmp_path / "enabled-models.txt"
    path.write_text("# accepted tasks\nmodel_b\n\n")
    monkeypatch.setattr(portal, "ENABLED_MODELS_FILE", path)
    monkeypatch.setattr(portal, "ENABLED_EXTERNAL_MODELS", {"model_a"})
    assert portal.enabled_external_models() == {"model_a", "model_b"}
    assert portal.enabled_models_mtime() == path.stat().st_mtime_ns
