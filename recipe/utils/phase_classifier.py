# Copyright (c) 2025 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory phase classification and statistics utilities.

Maps RL role names to execution phases (``Inference`` / ``Training``) and
computes per-phase memory statistics from a memory event DataFrame.

This module implements **G2** (classify and statistics memory by phase) of the
memory analysis feature.  The phase of each memory event is derived from the
``role`` of its owning profiling directory, because RL training loops interleave
two broad phases:

- **Training**: gradient computation + optimizer step + weight update
  (roles such as ``actor_update``, ``critic_update``).
- **Inference**: forward-only passes for rollout / log-prob recomputation /
  reference model (roles such as ``rollout_generate``,
  ``actor_compute_log_prob``, ``ref_compute_log_prob``).

The classifier is intentionally keyword-based (no external config) so it works
out of the box with the directory-name conventions used across the codebase
(see :class:`MemoryClusterParser`), while remaining overrideable via the
constructor for project-specific role naming.
"""

from typing import Any, Optional

import pandas as pd
from loguru import logger

#: All possible phase values produced by :class:`MemoryPhaseClassifier`.
PHASE_TRAINING = "Training"
PHASE_INFERENCE = "Inference"
PHASE_UNKNOWN = "Unknown"
ALL_PHASES: tuple[str, ...] = (PHASE_TRAINING, PHASE_INFERENCE, PHASE_UNKNOWN)

# Role-name keywords that indicate a training (gradient/update) phase.
# A role is classified as ``Training`` if it contains any of these substrings
# (case-insensitive).  ``Training`` keywords take precedence over ``Inference``
# keywords so that a hypothetical role such as ``actor_update_generate`` is
# treated as Training rather than Inference.
_TRAINING_KEYWORDS: tuple[str, ...] = (
    "update",
    "train",
    "backward",
    "optimizer",
    "optim_step",
    "step",
    "grad",
)

# Role-name keywords that indicate an inference (forward-only) phase.
# Only consulted when no ``Training`` keyword matched.
_INFERENCE_KEYWORDS: tuple[str, ...] = (
    "rollout",
    "generate",
    "compute_log_prob",
    "log_prob",
    "infer",
    "ref_",
    "reference",
    "forward",
    "sample",
    "decode",
)


class MemoryPhaseClassifier:
    """Classify memory events into RL execution phases.

    Example:
        >>> clf = MemoryPhaseClassifier()
        >>> clf.classify("actor_update")
        'Training'
        >>> clf.classify("rollout_generate")
        'Inference'
        >>> clf.classify("ref_compute_log_prob")
        'Inference'

    The classifier is stateless aside from the keyword lists; a single shared
    instance is safe to reuse across ranks / roles.
    """

    def __init__(
        self,
        training_keywords: Optional[tuple[str, ...]] = None,
        inference_keywords: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.training_keywords = training_keywords or _TRAINING_KEYWORDS
        self.inference_keywords = inference_keywords or _INFERENCE_KEYWORDS

    def classify(self, role: str) -> str:
        """Classify a single role name into a phase.

        ``Training`` keywords are checked **before** ``Inference`` keywords so
        that update/step roles are never misread as inference even if their
        name happens to contain an inference keyword.

        Args:
            role: RL role name (e.g. ``actor_update``, ``rollout_generate``).

        Returns:
            One of :data:`PHASE_TRAINING`, :data:`PHASE_INFERENCE`,
            :data:`PHASE_UNKNOWN`.
        """
        if not role:
            return PHASE_UNKNOWN
        role_lower = role.lower()
        for kw in self.training_keywords:
            if kw in role_lower:
                return PHASE_TRAINING
        for kw in self.inference_keywords:
            if kw in role_lower:
                return PHASE_INFERENCE
        return PHASE_UNKNOWN

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add / overwrite a ``phase`` column on a memory event DataFrame.

        Computes the phase from the ``role`` column for every row.  Rows with a
        missing ``role`` column or empty role receive :data:`PHASE_UNKNOWN`.
        The input DataFrame is not modified; a copy is returned.

        Args:
            df: Memory event DataFrame with a ``role`` column (as produced by
                :class:`recipe.parser.memory_parser.MemoryClusterParser`).

        Returns:
            A copy of *df* with a ``phase`` column of type ``str``.
        """
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        out = df.copy()
        if "role" in out.columns:
            out["phase"] = out["role"].map(self.classify).astype(str)
        else:
            out["phase"] = PHASE_UNKNOWN
        return out

    def compute_statistics(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Compute per-phase memory statistics from a memory event DataFrame.

        Groups events by ``phase`` (classifying first if the column is absent)
        and reports allocation counts / sizes / peak memory for each phase.

        Statistics per phase:

        - ``phase``: the phase name (``Training`` / ``Inference`` / ``Unknown``)
        - ``roles``: sorted list of roles observed in this phase
        - ``event_count``: total events (allocations + deallocations)
        - ``alloc_count``: number of allocations (``size_kb > 0``)
        - ``dealloc_count``: number of deallocations (``size_kb < 0``)
        - ``alloc_kb``: sum of allocation sizes (positive ``size_kb``), in KB
        - ``dealloc_kb``: absolute sum of deallocation sizes, in KB
        - ``net_kb``: ``alloc_kb - dealloc_kb`` (net retained), in KB
        - ``alloc_mb``: ``alloc_kb`` converted to MB
        - ``peak_allocated_mb``: max ``total_allocated_mb`` observed
        - ``peak_active_mb``: max ``total_active_mb`` observed
        - ``t_start_ms`` / ``t_end_ms``: time span of events in this phase

        Args:
            df: Memory event DataFrame.  If the ``phase`` column is missing it
                is derived from ``role`` via :meth:`classify_dataframe`.

        Returns:
            A dict keyed by phase name → stats dict.  Phases with no events are
            omitted from the result (so an inference-only run yields only the
            ``Inference`` key).  An empty input DataFrame yields an empty dict.
        """
        if df is None or df.empty:
            logger.debug("compute_statistics: empty input DataFrame")
            return {}

        if "phase" not in df.columns:
            df = self.classify_dataframe(df)

        kb_to_mb = 1.0 / 1024.0
        stats: dict[str, dict[str, Any]] = {}

        for phase, group in df.groupby("phase"):
            size_series = group["size_kb"].astype(float)
            alloc_mask = size_series > 0
            dealloc_mask = size_series < 0

            alloc_kb = float(size_series[alloc_mask].sum())
            dealloc_kb = float((-size_series[dealloc_mask]).sum())

            roles = (
                sorted(group["role"].astype(str).unique().tolist())
                if "role" in group.columns
                else []
            )

            phase_stats: dict[str, Any] = {
                "phase": str(phase),
                "roles": roles,
                "event_count": int(len(group)),
                "alloc_count": int(alloc_mask.sum()),
                "dealloc_count": int(dealloc_mask.sum()),
                "alloc_kb": round(alloc_kb, 4),
                "dealloc_kb": round(dealloc_kb, 4),
                "net_kb": round(alloc_kb - dealloc_kb, 4),
                "alloc_mb": round(alloc_kb * kb_to_mb, 4),
                "peak_allocated_mb": 0.0,
                "peak_active_mb": 0.0,
                "t_start_ms": 0.0,
                "t_end_ms": 0.0,
            }

            if "total_allocated_mb" in group.columns:
                phase_stats["peak_allocated_mb"] = round(
                    float(group["total_allocated_mb"].astype(float).max()), 4
                )
            if "total_active_mb" in group.columns:
                phase_stats["peak_active_mb"] = round(
                    float(group["total_active_mb"].astype(float).max()), 4
                )
            if "start_time_ms" in group.columns:
                t_start = float(group["start_time_ms"].astype(float).min())
                phase_stats["t_start_ms"] = round(t_start, 4)
            if "duration_ms" in group.columns and "start_time_ms" in group.columns:
                t_end = float(
                    (
                        group["start_time_ms"].astype(float)
                        + group["duration_ms"].astype(float)
                    ).max()
                )
                phase_stats["t_end_ms"] = round(t_end, 4)

            stats[str(phase)] = phase_stats

        logger.info(
            f"Phase statistics: {len(stats)} phases — "
            + ", ".join(
                f"{p}={s['event_count']}ev/{s['alloc_mb']:.2f}MB"
                for p, s in stats.items()
            )
        )
        return stats