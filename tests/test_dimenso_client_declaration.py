"""DIMENSO ADDITION (dimenso-context#564, robotics-api ADR 0029): the client declares its capabilities.

The backend reads teleop/dimenso_client.json to decide what the headset supports, instead of inferring it from
an implementation file existing. This test keeps the declaration honest: every declared capability names a
module that exists in this checkout, and every Dimenso module is declared.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = json.loads((ROOT / "teleop" / "dimenso_client.json").read_text())


def test_every_declared_capability_is_implemented_here():
    assert DECLARATION["schemaVersion"] == 1
    assert sorted(DECLARATION["capabilities"]) == sorted(DECLARATION["implementedBy"])
    for capability, path in DECLARATION["implementedBy"].items():
        assert (ROOT / path).is_file(), f"{capability} declared but {path} is missing"


def test_every_dimenso_module_is_declared():
    # Every Dimenso module this repo owns under teleop/, at any depth (submodules excluded: not ours).
    owned = [p for p in (ROOT / "teleop").rglob("dimenso_*.py")
             if not {"televuer", "teleimager", "dex-retargeting"} & set(p.relative_to(ROOT).parts)]
    modules = {str(p.relative_to(ROOT)) for p in owned}
    assert modules == set(DECLARATION["implementedBy"].values())
