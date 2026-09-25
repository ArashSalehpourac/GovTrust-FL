import pytest

from src.t2.provenance import REQUIRED_MANIFEST_FIELDS, validate_completed_manifest


def _manifest():
    m = {key: "x" for key in REQUIRED_MANIFEST_FIELDS}
    m.update({
        "status": "completed",
        "git_sha": "a" * 40,
        "git_dirty": False,
        "config_hash": "b" * 64,
        "preprocessor_fingerprint": "c" * 64,
        "privacy": {},
    })
    return m


def test_completed_clean_manifest_passes():
    validate_completed_manifest(_manifest())


def test_dirty_or_incomplete_manifest_rejected():
    dirty = _manifest()
    dirty["git_dirty"] = True
    with pytest.raises(ValueError):
        validate_completed_manifest(dirty)
    incomplete = _manifest()
    incomplete["status"] = "running"
    with pytest.raises(ValueError):
        validate_completed_manifest(incomplete)
