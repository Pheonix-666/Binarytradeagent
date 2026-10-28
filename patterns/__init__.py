"""Patterns package — chart pattern recognition and matching."""

from patterns.builtin import PatternTemplate, get_builtin_templates
from patterns.library import PatternLibrary
from patterns.similarity import (
    compute_combined_similarity,
    compute_vectorized_correlation,
    dtw_sakoe_chiba,
    resample_to_points,
    z_normalize,
)

__all__ = [
    "PatternLibrary",
    "PatternTemplate",
    "get_builtin_templates",
    "resample_to_points",
    "z_normalize",
    "compute_vectorized_correlation",
    "dtw_sakoe_chiba",
    "compute_combined_similarity",
]
