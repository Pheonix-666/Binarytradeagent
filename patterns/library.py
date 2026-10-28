"""Pattern library — template loading, validation, fast vectorized matching, and DTW refinement.

Sections 6.6 & 12:
  - Loads all templates once at startup into one 2D numpy matrix.
  - Validates template files (length, metadata JSON, direction, type).
  - Empty templates folder loads built-ins without crashing.
  - Multi-timeframe and multi-window candidate search without look-ahead.
  - Windows are taken from closed candles with no gaps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.config import PatternsConfig
from core.logging_setup import get_logger
from data.candle_builder import detect_gaps
from patterns.builtin import PatternTemplate, get_builtin_templates
from patterns.similarity import (
    DEFAULT_DTW_WEIGHT,
    DEFAULT_SAKOE_CHIBA_RADIUS,
    compute_combined_similarity,
    compute_vectorized_correlation,
    dtw_distance_to_similarity,
    dtw_sakoe_chiba,
    resample_to_points,
    z_normalize,
)

logger = get_logger()

VALID_DIRECTIONS = {"up", "down"}
VALID_TYPES = {"continuation", "reversal"}


class PatternLibrary:
    """Manages loaded pattern templates and evaluates chart windows for matches."""

    def __init__(
        self,
        templates_dir: Union[str, Path] = "patterns/templates",
        template_points: int = 64,
        include_builtins: bool = True,
    ):
        self.templates_dir = Path(templates_dir)
        self.template_points = template_points
        self.include_builtins = include_builtins
        self.templates: list[PatternTemplate] = []
        self.templates_matrix: np.ndarray = np.empty((0, template_points), dtype=np.float64)
        self.load_templates()

    def load_templates(self) -> None:
        """Load templates from templates_dir and built-ins once into one matrix.

        Fails with clear error naming the invalid file if any template array or
        metadata fails validation. Empty templates folder loads built-ins cleanly.
        """
        loaded: list[PatternTemplate] = []
        custom_names: set[str] = set()

        if self.templates_dir.exists():
            npy_files = list(self.templates_dir.glob("*.npy"))
            for npy_file in npy_files:
                json_file = npy_file.with_suffix(".json")
                if not json_file.exists():
                    raise ValueError(
                        f"Invalid template file {npy_file.name}: "
                        f"missing accompanying metadata JSON {json_file.name}"
                    )

                # Validate array
                try:
                    arr = np.load(npy_file)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid template file {npy_file.name}: could not load array: {exc}"
                    )

                if arr.ndim != 1:
                    raise ValueError(
                        f"Invalid template file {npy_file.name}: "
                        f"expected 1-D array, got {arr.ndim}-D"
                    )
                if len(arr) != self.template_points:
                    raise ValueError(
                        f"Invalid template file {npy_file.name}: "
                        f"expected length {self.template_points}, got {len(arr)}"
                    )

                # Validate metadata JSON
                try:
                    with open(json_file, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid template metadata {json_file.name}: JSON parse error: {exc}"
                    )

                name = meta.get("name")
                if not name or not isinstance(name, str):
                    raise ValueError(
                        f"Invalid template metadata {json_file.name}: missing or invalid 'name'"
                    )

                direction = meta.get("direction", "").lower()
                if direction not in VALID_DIRECTIONS:
                    raise ValueError(
                        f"Invalid template metadata {json_file.name}: "
                        f"direction must be 'up' or 'down', got '{direction}'"
                    )

                ptype = meta.get("type", "").lower()
                if ptype not in VALID_TYPES:
                    raise ValueError(
                        f"Invalid template metadata {json_file.name}: "
                        f"type must be 'continuation' or 'reversal', got '{ptype}'"
                    )

                normed, is_flat = z_normalize(arr)
                if is_flat:
                    raise ValueError(
                        f"Invalid template file {npy_file.name}: array is flat (zero variance)"
                    )

                tpl = PatternTemplate(
                    name=name,
                    direction=direction,
                    pattern_type=ptype,
                    data=normed,
                    timeframe_hint=meta.get("timeframe_hint"),
                )
                loaded.append(tpl)
                custom_names.add(name)

        # Include built-in templates (unless overridden by custom template or disabled)
        builtins: list[PatternTemplate] = []
        if self.include_builtins:
            builtins = get_builtin_templates(num_points=self.template_points)
            for b in builtins:
                if b.name not in custom_names:
                    loaded.append(b)

        self.templates = loaded
        if self.templates:
            self.templates_matrix = np.vstack([t.data for t in self.templates])
        else:
            self.templates_matrix = np.empty((0, self.template_points), dtype=np.float64)

        logger.info(
            f"PatternLibrary loaded {len(self.templates)} templates "
            f"({len(custom_names)} custom, {len(builtins)} built-ins)"
        )

    def match_window(
        self,
        window_close: Union[np.ndarray, Sequence[float]],
        dtw_shortlist: int = 3,
        dtw_weight: float = DEFAULT_DTW_WEIGHT,
        dtw_radius: int = DEFAULT_SAKOE_CHIBA_RADIUS,
    ) -> list[dict]:
        """Match a 1-D price window against all loaded templates.

        Steps:
          1. Resample window to template_points.
          2. Z-normalize (guard against flat window: score 0.0).
          3. Fast vectorized Pearson correlation across all templates.
          4. DTW with Sakoe-Chiba band for top dtw_shortlist matches.
          5. Combine: (1 - dtw_weight) * corr_sim + dtw_weight * dtw_sim.
          6. Return sorted best-first.
        """
        arr = np.asarray(window_close, dtype=np.float64)
        if len(arr) == 0 or len(self.templates) == 0:
            return []

        # 1. Resample to template_points
        resampled = resample_to_points(arr, num_points=self.template_points)

        # 2. Z-normalize
        normed, is_flat = z_normalize(resampled)

        if is_flat:
            # Flat window guard: similarity is 0.0 for all templates
            return [
                {
                    "pattern": t.name,
                    "direction": t.direction,
                    "type": t.pattern_type,
                    "similarity": 0.0,
                    "refined": False,
                }
                for t in self.templates
            ]

        # 3. Vectorized Pearson correlation
        corr_sims = compute_vectorized_correlation(
            self.templates_matrix, normed, is_flat=is_flat
        )

        # 4. Rank candidates by correlation
        ranked_indices = np.argsort(corr_sims)[::-1]
        final_scores = np.copy(corr_sims)
        is_refined = np.zeros(len(self.templates), dtype=bool)

        # 5. DTW refinement on top dtw_shortlist
        num_shortlist = min(dtw_shortlist, len(ranked_indices))
        for idx in ranked_indices[:num_shortlist]:
            is_refined[idx] = True
            if corr_sims[idx] <= 0.0:
                final_scores[idx] = 0.0
                continue
            template_arr = self.templates_matrix[idx]
            dtw_dist = dtw_sakoe_chiba(normed, template_arr, radius=dtw_radius)
            dtw_sim = dtw_distance_to_similarity(dtw_dist)
            final_scores[idx] = compute_combined_similarity(
                corr_similarity=corr_sims[idx],
                dtw_similarity=dtw_sim,
                dtw_weight=dtw_weight,
                is_flat=False,
            )

        # 6. Sort results: compare like with like
        # Refined matches are sorted best-first among themselves.
        # Unrefined matches are sorted best-first among themselves and placed after refined.
        results = [
            {
                "pattern": self.templates[i].name,
                "direction": self.templates[i].direction,
                "type": self.templates[i].pattern_type,
                "similarity": round(float(final_scores[i]), 2),
                "refined": bool(is_refined[i]),
            }
            for i in range(len(self.templates))
        ]
        results.sort(key=lambda m: (m["refined"], m["similarity"]), reverse=True)

        return results

    def find_patterns_for_asset(
        self,
        candles_by_tf: dict[str, pd.DataFrame],
        patterns_cfg: PatternsConfig,
        refined_only: bool = True,
    ) -> list[dict]:
        """Find best matching patterns across configured timeframes and window lengths.

        Rules:
          - Uses closed candles from each timeframe in patterns_cfg.timeframes.
          - Windows with gaps are skipped (no gaps permitted).
          - If refined_only=True (default), filters out unrefined matches.
          - Output per asset: list of {pattern, direction, type, similarity, timeframe, window_length}
            sorted best-first.
        """
        all_candidates: list[dict] = []

        for tf in patterns_cfg.timeframes:
            df = candles_by_tf.get(tf, pd.DataFrame())
            if df.empty:
                continue

            for w_len in patterns_cfg.window_lengths:
                if len(df) < w_len:
                    continue

                window_df = df.tail(w_len)

                # Gap check: windows must have no gaps
                gaps = detect_gaps(window_df, tf)
                if gaps:
                    logger.debug(
                        f"Skipping pattern window {w_len} on {tf}: {len(gaps)} gap(s) found"
                    )
                    continue

                closes = window_df["close"].values
                matches = self.match_window(
                    closes,
                    dtw_shortlist=patterns_cfg.dtw_shortlist,
                    dtw_weight=patterns_cfg.dtw_weight,
                    dtw_radius=patterns_cfg.dtw_sakoe_chiba_radius,
                )

                for m in matches:
                    candidate = dict(m)
                    candidate["timeframe"] = tf
                    candidate["window_length"] = w_len
                    all_candidates.append(candidate)

        if refined_only:
            all_candidates = [c for c in all_candidates if c.get("refined", False)]

        # Sort candidates comparing like with like: refined matches first, then by similarity
        all_candidates.sort(
            key=lambda x: (x.get("refined", False), x["similarity"]), reverse=True
        )
        return all_candidates


_DEFAULT_LIBRARY: Optional[PatternLibrary] = None


def get_pattern_library(templates_dir: Union[str, Path] = "patterns/templates") -> PatternLibrary:
    """Return a shared singleton PatternLibrary instance."""
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        _DEFAULT_LIBRARY = PatternLibrary(templates_dir=templates_dir)
    return _DEFAULT_LIBRARY
