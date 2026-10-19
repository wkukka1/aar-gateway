"""Small persisted json files: a corrupt one is kept and reported, never silently erased."""

from __future__ import annotations

import logging
import os

import pytest

from aar_gateway import interceptor, share
from aar_gateway.config import load_settings, preferences_path, read_json


def test_read_json_missing_is_the_default_and_corrupt_is_logged_by_path(tmp_path, caplog):
    path = tmp_path / "x.json"
    assert read_json(path, {"d": 1}) == {"d": 1}
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert read_json(path, {}) == {}
    assert str(path) in caplog.text and "not json" not in caplog.text
    with pytest.raises(ValueError):
        read_json(path, {}, strict=True)
    path.write_text("[1, 2]", encoding="utf-8")  # valid json, wrong shape
    assert read_json(path, {}) == {}


def test_saving_a_preference_over_a_corrupt_file_keeps_the_corrupt_one_aside():
    interceptor.configure(load_settings(None), None)
    path = preferences_path()
    path.write_text('{"quality_margin": 0.3, "mode": "pinn', encoding="utf-8")  # cut short by a crash
    interceptor.set_allow_upgrade(True)
    kept = path.with_name(path.name + ".corrupt")
    assert kept.read_text(encoding="utf-8").startswith('{"quality_margin": 0.3')  # recoverable by hand
    assert interceptor.read_preferences() == {"allow_upgrade": True}


def test_a_good_preferences_file_is_merged_not_replaced():
    interceptor.configure(load_settings(None), None)
    interceptor.set_quality_margin(0.2)
    interceptor.set_allow_upgrade(True)
    assert interceptor.read_preferences() == {"quality_margin": 0.2, "allow_upgrade": True}
    assert not preferences_path().with_name("preferences.json.corrupt").exists()


def test_consent_toggled_within_the_same_mtime_is_still_seen():
    share.set_enabled(True)
    assert share.enabled() is True
    stamp = os.stat(share.consent_path()).st_mtime_ns
    share.set_enabled(False)
    os.utime(share.consent_path(), ns=(stamp, stamp))  # a coarse clock: same mtime as before
    assert share.enabled() is False
