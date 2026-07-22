import json

import numpy as np
import pytest

from reccon_study.cli import ALL_SESSIONS, bootstrap_ci, parse_sessions, small_n_marker, write_report


def test_parse_sessions_comma_list():
    assert parse_sessions("4,5") == ["Session4", "Session5"]


def test_parse_sessions_single():
    assert parse_sessions("5") == ["Session5"]


def test_parse_sessions_default_is_all_sessions():
    assert parse_sessions(None) == ALL_SESSIONS
    assert parse_sessions("") == ALL_SESSIONS


def test_parse_sessions_strips_whitespace():
    assert parse_sessions(" 4 , 5 ") == ["Session4", "Session5"]


def test_parse_sessions_returns_a_copy_not_the_shared_list():
    result = parse_sessions(None)
    result.append("Session99")
    assert "Session99" not in ALL_SESSIONS


def test_small_n_marker_below_threshold():
    assert small_n_marker(10, threshold=30) == " [SMALL N]"


def test_small_n_marker_at_or_above_threshold():
    assert small_n_marker(30, threshold=30) == ""
    assert small_n_marker(100, threshold=30) == ""


def test_write_report_writes_valid_json(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3]}
    path = write_report("myreport", payload, out_dir=tmp_path)
    assert path == tmp_path / "myreport.json"
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == payload


def test_write_report_creates_out_dir(tmp_path):
    out_dir = tmp_path / "nested" / "dir"
    write_report("x", {"k": "v"}, out_dir=out_dir)
    assert (out_dir / "x.json").exists()


def test_bootstrap_ci_known_distribution_brackets_true_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=1.0, size=200)

    def stat(idx):
        return float(values[idx].mean())

    point, lo, hi = bootstrap_ci(len(values), stat, n_resamples=2000, seed=1)
    assert point == pytest.approx(values.mean())
    assert lo < point < hi
    assert lo < 5.0 < hi  # true mean should be well within a 95% CI at n=200


def test_bootstrap_ci_deterministic_by_seed():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def stat(idx):
        return float(values[idx].mean())

    r1 = bootstrap_ci(len(values), stat, n_resamples=500, seed=42)
    r2 = bootstrap_ci(len(values), stat, n_resamples=500, seed=42)
    assert r1 == r2


def test_bootstrap_ci_empty_returns_nan():
    point, lo, hi = bootstrap_ci(0, lambda idx: 0.0, n_resamples=10, seed=1)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)
