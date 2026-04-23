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

import json
from loguru import logger
import os

from .ascend_parser import AscendClusterParser
from .parser import register_cluster_parser
from rl_insight.utils.schema import Constant, EventRow
from rl_insight.data import DataEnum


@register_cluster_parser("mstx")
class MstxClusterParser(AscendClusterParser):
    input_type: DataEnum = DataEnum.MULTI_JSON_MSTX

    def __init__(self, params) -> None:
        super().__init__(params)

    def parse_analysis_data(
        self, profiler_data_path: str, rank_id: int, role: str
    ) -> list[EventRow]:
        data: list[dict] = []
        events: list[EventRow] = []

        with open(profiler_data_path, encoding="utf-8") as f:
            data = json.load(f)

        if data is None or not data:
            logger.warning(f"Rank {rank_id}: No rollout events found in json")
            return events

        process_id = None
        start_ids = None
        end_ids = None
        for row in data:
            if (
                row.get("ph") == "M"
                and row.get("args", {}).get("name") == "Overlap Analysis"
            ):
                process_id = row.get("pid")
                break

        if process_id is None:
            logger.warning(
                f"Rank {rank_id}: Overlap Analysis process not found in json"
            )
            return events

        for row in data:
            if row.get("pid") != process_id or row.get("ph") != "X":
                continue

            args = row.get("args")
            if not isinstance(args, dict):
                continue

            if "ts" not in row or "dur" not in row:
                logger.warning("Row missing required fields: ts or dur. Skipping row.")
                continue

            try:
                start_time_ns = float(row["ts"])
                duration_ns = float(row["dur"])
                end_time_ns = start_time_ns + duration_ns

                if start_ids is None or start_time_ns < start_ids:
                    start_ids = start_time_ns
                if end_ids is None or end_time_ns > end_ids:
                    end_ids = end_time_ns

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to convert time values: {e}. Row data: {row}. Skipping row."
                )
                continue

        if start_ids is None or end_ids is None:
            logger.warning(f"Rank {rank_id}: No valid timing rows for Overlap Analysis")
            return events

        us_to_ms = Constant.US_TO_MS
        start_time_ms = start_ids / us_to_ms
        duration_ms = (end_ids - start_ids) / us_to_ms
        end_time_ms = start_time_ms + duration_ms

        event_data: EventRow = {
            "name": role,
            "role": role,
            "domain": "default",
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "duration_ms": duration_ms,
            "rank_id": rank_id,
            "tid": process_id,
        }
        events.append(event_data)

        return events

    def _get_profiler_data_path(self, rank_id, data_path):
        return os.path.join(
            data_path, Constant.ASCEND_PROFILER_OUTPUT, "trace_view.json"
        )