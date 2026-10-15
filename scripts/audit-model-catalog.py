#!/usr/bin/env python3
"""Fail unless every represented external model is accepted or precisely blocked."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
import model_preflight


def main() -> int:
    preflight = model_preflight.audit()
    models = {**app.EXTERNAL_TASKS, **app.INFORMATIONAL_MODELS}
    errors = []
    accepted = []
    blocked = []

    for name, details in sorted(models.items()):
        declared_disabled = details.get("enabled") is False
        blocker = details.get("blocked_status") or details.get("status")
        check = preflight.get(name)

        if declared_disabled:
            if not blocker or not details.get("description") or not details.get("reference"):
                errors.append(f"{name}: disabled entry lacks blocker, description, or reference")
            else:
                blocked.append({"name": name, "blocker": blocker})
            continue

        if check is None:
            errors.append(f"{name}: runnable entry has no preflight definition")
            continue
        if check["ready"] and check["orientation_ready"]:
            accepted.append(name)
            continue
        if not details.get("blocked_status"):
            errors.append(f"{name}: not accepted and has no precise blocked_status: {check}")
            continue
        blocked.append({"name": name, "blocker": details["blocked_status"],
                        "missing": check["missing"],
                        "orientation_ready": check["orientation_ready"]})

    report = {
        "represented_external_models_or_tasks": len(models),
        "accepted_runnable": len(accepted),
        "disabled_with_precise_blocker": len(blocked),
        "preflight_definitions": len(preflight),
        "errors": errors,
        "accepted": accepted,
        "blocked": blocked,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
