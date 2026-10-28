"""Unit and performance tests for Phase 3: Pattern Library."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import PatternsConfig
from patterns.builtin import (
    generate_double_bottom,
    generate_double_top,
    generate_resistance_bounce,
    generate_support_bounce,
    get_builtin_templates,
)
from patterns.library import PatternLibrary
from patterns.similarity import (
    compute_combined_similarity,
    compute_vectorized_correlation,
    dtw_distance_to_similarity,
    dtw_sakoe_chiba,
    resample_to_points,
    z_normalize,
)


@pytest.fixture
def library():
    """Return PatternLibrary initialized with built-in templates."""
    return PatternLibrary(templates_dir="patterns/templates", template_points=64)


# ── Built-in Shapes Tests ─────────────────────────────────────────────


def test_builtin_shapes_self_match(library):
    """Every built-in shape matches itself with near 100% similarity."""
    for tpl in library.templates:
        matches = library.match_window(tpl.data)
        top_match = matches[0]
        assert top_match["pattern"] == tpl.name
        assert top_match["similarity"] >= 99.5


def test_builtin_shapes_vs_noisy_versions(library):
    """A shape with added Gaussian noise maintains high similarity (> 85%)."""
    np.random.seed(42)
    for tpl in library.templates:
        noisy = tpl.data + np.random.normal(0.0, 0.2, len(tpl.data))
        matches = library.match_window(noisy)
        matched_dict = {m["pattern"]: m["similarity"] for m in matches}
        assert matched_dict[tpl.name] >= 85.0


def test_builtin_shapes_vs_random_noise(library):
    """Random noise scores low similarity (< 50%) across all templates."""
    np.random.seed(999)
    noise = np.random.randn(64)
    matches = library.match_window(noise)
    for m in matches:
        assert m["similarity"] < 50.0


def test_builtin_shapes_vs_opposite_shape(library):
    """Opposite shapes (double_top vs double_bottom, support vs resistance bounce) score 0%."""
    db = generate_double_bottom(64)
    matches_dt = library.match_window(-db.data)
    matched_dict = {m["pattern"]: m["similarity"] for m in matches_dt}
    # Opposite shape has negative correlation, which clips to 0.0
    assert matched_dict["double_bottom"] == 0.0

    sb = generate_support_bounce(64)
    matches_sb = library.match_window(-sb.data)
    matched_dict_sb = {m["pattern"]: m["similarity"] for m in matches_sb}
    assert matched_dict_sb["support_bounce"] == 0.0


# ── Similarity Function Tests ─────────────────────────────────────────


def test_time_stretched_copy_scores_high(library):
    """A time-stretched / compressed copy of a pattern scores high (> 80%)."""
    db = generate_double_bottom(64)
    # Stretch first half and compress second half
    first_half = resample_to_points(db.data[:32], num_points=36)
    second_half = resample_to_points(db.data[32:], num_points=28)
    stretched = np.concatenate([first_half, second_half])

    matches = library.match_window(stretched)
    matched_dict = {m["pattern"]: m["similarity"] for m in matches}
    assert matched_dict["double_bottom"] >= 80.0


def test_mirrored_copy_scores_low(library):
    """A mirrored (time-reversed) copy of a directional pattern scores low / 0%."""
    sb = generate_support_bounce(64)
    mirrored_sb = sb.data[::-1]
    matches_sb = library.match_window(mirrored_sb)
    matched_dict_sb = {m["pattern"]: m["similarity"] for m in matches_sb}
    assert matched_dict_sb["support_bounce"] < 20.0

    rb = generate_resistance_bounce(64)
    mirrored_rb = rb.data[::-1]
    matches_rb = library.match_window(mirrored_rb)
    matched_dict_rb = {m["pattern"]: m["similarity"] for m in matches_rb}
    assert matched_dict_rb["resistance_bounce"] < 20.0


def test_flat_window_returns_zero_without_error(library):
    """Flat window (all prices equal) guards against divide-by-zero and scores 0.0."""
    flat_data = np.full(64, 1.0850)
    matches = library.match_window(flat_data)
    assert len(matches) == len(library.templates)
    for m in matches:
        assert m["similarity"] == 0.0


def test_negative_correlation_clips_to_zero():
    """Negative correlation must be clipped to 0.0."""
    t = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([4.0, 3.0, 2.0, 1.0])  # strictly negative correlation
    t_norm, _ = z_normalize(t)
    q_norm, _ = z_normalize(q)
    matrix = t_norm.reshape(1, -1)

    sim = compute_vectorized_correlation(matrix, q_norm)
    assert sim[0] == 0.0


# ── Template Loading & Validation Tests ────────────────────────────────


def test_empty_templates_folder_loads_builtins_only(tmp_path):
    """Empty templates folder does not crash and loads only built-in templates."""
    empty_dir = tmp_path / "empty_templates"
    empty_dir.mkdir()
    lib = PatternLibrary(templates_dir=empty_dir, template_points=64)
    assert len(lib.templates) == 4
    names = {t.name for t in lib.templates}
    assert names == {"double_bottom", "double_top", "support_bounce", "resistance_bounce"}


def test_template_validation_missing_json(tmp_path):
    """Template .npy missing metadata JSON raises ValueError naming the file."""
    t_dir = tmp_path / "templates"
    t_dir.mkdir()
    np.save(t_dir / "custom_pattern.npy", np.zeros(64))

    with pytest.raises(ValueError, match="custom_pattern.npy: missing accompanying metadata JSON"):
        PatternLibrary(templates_dir=t_dir, template_points=64)


def test_template_validation_wrong_length(tmp_path):
    """Template array with wrong length raises ValueError naming the file."""
    t_dir = tmp_path / "templates"
    t_dir.mkdir()
    np.save(t_dir / "short_pattern.npy", np.zeros(32))
    with open(t_dir / "short_pattern.json", "w", encoding="utf-8") as fh:
        json.dump({"name": "short", "direction": "up", "type": "reversal"}, fh)

    with pytest.raises(ValueError, match="short_pattern.npy: expected length 64, got 32"):
        PatternLibrary(templates_dir=t_dir, template_points=64)


def test_template_validation_invalid_direction(tmp_path):
    """Template with invalid direction raises ValueError naming the file."""
    t_dir = tmp_path / "templates"
    t_dir.mkdir()
    np.save(t_dir / "bad_dir.npy", np.linspace(0, 1, 64))
    with open(t_dir / "bad_dir.json", "w", encoding="utf-8") as fh:
        json.dump({"name": "bad", "direction": "sideways", "type": "reversal"}, fh)

    with pytest.raises(ValueError, match="bad_dir.json: direction must be 'up' or 'down'"):
        PatternLibrary(templates_dir=t_dir, template_points=64)


def test_template_validation_invalid_type(tmp_path):
    """Template with invalid pattern type raises ValueError naming the file."""
    t_dir = tmp_path / "templates"
    t_dir.mkdir()
    np.save(t_dir / "bad_type.npy", np.linspace(0, 1, 64))
    with open(t_dir / "bad_type.json", "w", encoding="utf-8") as fh:
        json.dump({"name": "bad", "direction": "up", "type": "breakout"}, fh)

    with pytest.raises(ValueError, match="bad_type.json: type must be 'continuation' or 'reversal'"):
        PatternLibrary(templates_dir=t_dir, template_points=64)


# ── Asset Search & Gap Handling Tests ─────────────────────────────────


def test_find_patterns_for_asset_skips_gapped_windows(library):
    """Windows with timestamp gaps are skipped; only continuous windows are evaluated."""
    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    # Create 50 5m candles but introduce a gap at index 35
    records = []
    curr = base
    for i in range(50):
        if i == 35:
            curr += timedelta(minutes=25)  # gap!
        else:
            curr += timedelta(minutes=5)
        records.append({"ts": curr, "close": 1.10 + i * 0.001, "open": 1.10, "high": 1.11, "low": 1.09, "volume": 10})

    df_5m = pd.DataFrame(records)
    cfg = PatternsConfig(
        timeframes=["5m"],
        min_similarity=60.0,
        template_points=64,
        window_lengths=[30, 45],  # 45 will include the gap at 35; 30 will only use 20..49 which still includes the gap!
    )
    # The last 30 bars (20..49) include index 35 (the gap), so it must be skipped
    matches = library.find_patterns_for_asset({"5m": df_5m}, cfg)
    assert matches == []


# ── No Look-Ahead Test ────────────────────────────────────────────────


def test_pattern_matching_no_look_ahead(library):
    """Pattern similarity at time t MUST be completely identical when future candles are appended."""
    np.random.seed(777)
    n_past = 60
    n_future = 30
    prices_past = 100.0 + np.cumsum(np.random.randn(n_past) * 0.4)
    prices_future = prices_past[-1] + np.cumsum(np.random.randn(n_future) * 0.4)
    prices_extended = np.concatenate([prices_past, prices_future])

    # Evaluate at index n_past (past window)
    matches_past = library.match_window(prices_past[-45:])

    # Evaluate using the exact same window slice from extended data
    matches_extended_slice = library.match_window(prices_extended[:n_past][-45:])

    for m_past, m_ext in zip(matches_past, matches_extended_slice):
        assert m_past["pattern"] == m_ext["pattern"]
        assert m_past["similarity"] == m_ext["similarity"]


# ── Performance Benchmark Test ────────────────────────────────────────


def test_performance_benchmark(tmp_path):
    """Performance requirement: 4 assets x 4 window lengths x 20 templates in under 50 ms."""
    t_dir = tmp_path / "bench_templates"
    t_dir.mkdir()

    # Create 20 synthetic templates of length 64
    for i in range(20):
        arr = np.sin(np.linspace(0, (i + 1) * np.pi, 64))
        np.save(t_dir / f"tpl_{i}.npy", arr)
        with open(t_dir / f"tpl_{i}.json", "w", encoding="utf-8") as fh:
            json.dump({"name": f"tpl_{i}", "direction": "up", "type": "reversal"}, fh)

    bench_lib = PatternLibrary(templates_dir=t_dir, template_points=64, include_builtins=False)
    assert len(bench_lib.templates) == 20

    # 4 assets x 4 window lengths
    np.random.seed(42)
    assets = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    window_lengths = [30, 45, 60, 90]
    asset_data = {
        asset: np.random.randn(100)
        for asset in assets
    }

    # Warm-up run (ensures JIT/CPU caches are warm)
    bench_lib.match_window(asset_data["EURUSD"][-30:], dtw_shortlist=3)

    # Benchmark run
    start_time = time.perf_counter()

    eval_count = 0
    for asset in assets:
        for w_len in window_lengths:
            window = asset_data[asset][-w_len:]
            matches = bench_lib.match_window(window, dtw_shortlist=3)
            assert len(matches) == 20
            eval_count += 1

    elapsed_time = time.perf_counter() - start_time
    elapsed_ms = elapsed_time * 1000.0

    print(f"\n[PERFORMANCE] Evaluated {len(assets)} assets x {len(window_lengths)} windows x 20 templates ({eval_count} total evaluations) in {elapsed_ms:.2f} ms")
    assert elapsed_time < 0.050, f"Benchmark exceeded 50 ms threshold: took {elapsed_ms:.2f} ms"

