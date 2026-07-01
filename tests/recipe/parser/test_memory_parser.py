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

import csv
import json
import pytest

from recipe.parser import MemoryClusterParser, get_cluster_parser_cls
from recipe.parser.parser import CLUSTER_PARSER_REGISTRY
from recipe.utils.phase_classifier import (
    MemoryPhaseClassifier,
    PHASE_INFERENCE,
    PHASE_TRAINING,
    PHASE_UNKNOWN,
)
from recipe.utils.schema import Constant


def _write_trace_view_json(path, events):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f)


def _write_operator_memory_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Name",
                "Size(KB)",
                "Allocation Time(us)",
                "Release Time(us)",
                "Active Release Time(us)",
                "Duration(us)",
                "Allocation Total Allocated(MB)",
                "Allocation Total Reserved(MB)",
                "Allocation Total Active(MB)",
                "Release Total Allocated(MB)",
                "Release Total Reserved(MB)",
                "Release Total Active(MB)",
                "Stream Ptr",
                "Device Type",
            ]
        )
        for row in rows:
            writer.writerow(row)


def _create_ascend_profile_dir(
    tmp_path, role="actor_update", rank_id=0, trace_events=None, memory_rows=None
):
    role_dir = tmp_path / role
    role_dir.mkdir(exist_ok=True)
    ascend_pt_dir = role_dir / "20250101_120000_ascend_pt"
    ascend_pt_dir.mkdir()
    profiler_info = ascend_pt_dir / f"profiler_info_{rank_id}.json"
    profiler_info.write_text("{}")
    metadata = ascend_pt_dir / "profiler_metadata.json"
    metadata.write_text(json.dumps({"role": role}))

    output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
    output_dir.mkdir()

    if trace_events is not None:
        _write_trace_view_json(str(output_dir / "trace_view.json"), trace_events)
    if memory_rows is not None:
        _write_operator_memory_csv(str(output_dir / "operator_memory.csv"), memory_rows)

    return str(tmp_path)


SAMPLE_TRACE_EVENTS = [
    {
        "ph": "X",
        "name": "aten::empty",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "1000000.0",
        "dur": 500.0,
        "args": {"Call stack": "fsdp2.py(112): train_batch;\r\nmodel.py(50): forward"},
    },
    {
        "ph": "X",
        "name": "aten::matmul",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "2000000.0",
        "dur": 1000.0,
        "args": {"Call stack": "model.py(60): forward;\r\nlayer.py(30): __call__"},
    },
    {
        "ph": "X",
        "name": "aten::empty",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "3000000.0",
        "dur": 300.0,
        "args": {"Call stack": "fsdp2.py(120): train_batch;\r\nmodel.py(55): forward"},
    },
    {
        "ph": "X",
        "name": "aten::relu",
        "cat": "kernel",
        "pid": 1,
        "tid": 1,
        "ts": "4000000.0",
        "dur": 200.0,
    },
    {
        "ph": "X",
        "name": "aten::cumsum",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "5000000.0",
        "dur": 800.0,
    },
]

SAMPLE_MEMORY_ROWS = [
    [
        "aten::empty",
        1024.0,
        "1000100.0",
        "",
        "",
        "",
        100.0,
        200.0,
        50.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::empty",
        2048.0,
        "3000050.0",
        "",
        "",
        "",
        200.0,
        300.0,
        100.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::matmul",
        4096.0,
        "2000500.0",
        "",
        "",
        "",
        300.0,
        400.0,
        150.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::unknown",
        512.0,
        "6000000.0",
        "",
        "",
        "",
        50.0,
        100.0,
        25.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::empty",
        -1024.0,
        "7000000.0",
        "",
        "",
        "",
        100.0,
        200.0,
        50.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
]


# =============================================================================
# Parser Registration Tests
# =============================================================================


class TestMemoryParserRegistry:
    def test_memory_parser_registered(self):
        assert "memory" in CLUSTER_PARSER_REGISTRY
        assert CLUSTER_PARSER_REGISTRY["memory"] == MemoryClusterParser

    def test_get_memory_parser_cls(self):
        parser_cls = get_cluster_parser_cls("memory")
        assert parser_cls == MemoryClusterParser


# =============================================================================
# _build_call_stack_index Tests
# =============================================================================


class TestBuildCallStackIndex:
    def test_filters_cpu_op_only(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "op2",
                "cat": "kernel",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {"Call stack": "stack2"},
            },
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert "op1" in index
        assert "op2" not in index

    def test_filters_events_without_call_stack(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "op2",
                "cat": "cpu_op",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {},
            },
            {"ph": "X", "name": "op3", "cat": "cpu_op", "ts": "3000.0", "dur": 300.0},
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert "op1" in index
        assert "op2" not in index
        assert "op3" not in index

    def test_groups_by_name(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "aten::empty",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "aten::empty",
                "cat": "cpu_op",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {"Call stack": "stack2"},
            },
            {
                "ph": "X",
                "name": "aten::matmul",
                "cat": "cpu_op",
                "ts": "3000.0",
                "dur": 300.0,
                "args": {"Call stack": "stack3"},
            },
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert len(index["aten::empty"]["entries"]) == 2
        assert len(index["aten::matmul"]["entries"]) == 1

    def test_sorted_by_ts(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "3000.0",
                "dur": 300.0,
                "args": {"Call stack": "stack3"},
            },
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {"Call stack": "stack2"},
            },
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        ts_values = index["op1"]["ts_list"]
        assert ts_values == [1000.0, 2000.0, 3000.0]

    def test_empty_json(self, tmp_path):
        _write_trace_view_json(str(tmp_path / "trace_view.json"), [])

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert len(index) == 0


# =============================================================================
# _match_call_stack Tests
# =============================================================================


class TestMatchCallStack:
    def setup_method(self):
        self.parser = MemoryClusterParser(
            {Constant.INPUT_PATH: "/tmp", Constant.RANK_LIST: "all"}
        )
        self.index = {
            "aten::empty": {
                "entries": [
                    {
                        "ts": 1000.0,
                        "dur": 500.0,
                        "call_stack": "fsdp2.py(10): func1;\r\nmodel.py(20): func2",
                    },
                    {
                        "ts": 3000.0,
                        "dur": 300.0,
                        "call_stack": "fsdp2.py(30): func3;\r\nmodel.py(40): func4",
                    },
                ],
                "ts_list": [1000.0, 3000.0],
            },
            "aten::matmul": {
                "entries": [
                    {
                        "ts": 5000.0,
                        "dur": 1000.0,
                        "call_stack": "model.py(50): forward",
                    },
                ],
                "ts_list": [5000.0],
            },
        }

    def test_match_found(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 1100.0, self.index
        )
        assert call_stack == "fsdp2.py(10): func1;\r\nmodel.py(20): func2"
        assert call_stack_top == "fsdp2.py(10): func1"

    def test_match_closest_ts(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 3500.0, self.index
        )
        assert call_stack == "fsdp2.py(30): func3;\r\nmodel.py(40): func4"
        assert call_stack_top == "fsdp2.py(30): func3"

    def test_name_not_found(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::unknown", 1000.0, self.index
        )
        assert call_stack == ""
        assert call_stack_top == ""

    def test_all_ts_greater_than_allocation_time(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 500.0, self.index
        )
        assert call_stack == ""
        assert call_stack_top == ""

    def test_exact_ts_match(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 3000.0, self.index
        )
        assert call_stack == "fsdp2.py(30): func3;\r\nmodel.py(40): func4"


# =============================================================================
# _parse_operator_memory Tests
# =============================================================================


class TestParseOperatorMemory:
    def test_parse_basic(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    1024.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [
                    {
                        "ts": 1000000.0,
                        "dur": 500.0,
                        "call_stack": "fsdp2.py(10): func;\r\nmodel.py(20): func2",
                    },
                ],
                "ts_list": [1000000.0],
            },
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert len(results) == 1
        row = results[0]
        assert row["name"] == "aten::empty"
        assert row["size_kb"] == 1024.0
        assert row["start_time_ms"] == pytest.approx(1000.1)
        assert row["call_stack"] == "fsdp2.py(10): func;\r\nmodel.py(20): func2"
        assert row["call_stack_top"] == "fsdp2.py(10): func"
        assert row["role"] == "actor_update"
        assert row["rank_id"] == 0
        assert row["device_type"] == "NPU:0"

    def test_parse_negative_size(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    -1024.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [
                    {"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"},
                ],
                "ts_list": [1000000.0],
            },
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert len(results) == 1
        assert results[0]["size_kb"] == -1024.0

    def test_parse_duration_conversion(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    1024.0,
                    "1000100.0",
                    "",
                    "",
                    "5000.0",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [{"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"}],
                "ts_list": [1000000.0],
            }
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert results[0]["duration_ms"] == pytest.approx(5.0)

    def test_parse_empty_duration(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    1024.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [{"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"}],
                "ts_list": [1000000.0],
            }
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert results[0]["duration_ms"] == 0.0

    def test_parse_unmatched_call_stack(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::unknown",
                    512.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    50.0,
                    100.0,
                    25.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {}

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert results[0]["call_stack"] == ""
        assert results[0]["call_stack_top"] == ""

    def test_parse_empty_allocation_total_fields(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    -1024.0,
                    "1000100.0",
                    "2000100.0",
                    "2000100.0",
                    "1000.0",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [{"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"}],
                "ts_list": [1000000.0],
            }
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert len(results) == 1
        assert results[0]["total_allocated_mb"] == 0.0
        assert results[0]["total_reserved_mb"] == 0.0
        assert results[0]["total_active_mb"] == 0.0


# =============================================================================
# _extract_timestamp_key Tests
# =============================================================================


class TestExtractTimestampKey:
    def test_timestamp_format(self):
        assert (
            MemoryClusterParser._extract_timestamp_key(
                "/data/actor/20250101_120000_ascend_pt"
            )
            == "20250101_120000"
        )

    def test_sort_order(self):
        paths = [
            "/data/role/20250102_010000_ascend_pt",
            "/data/role/20250101_230000_ascend_pt",
            "/data/role/20250101_120000_ascend_pt",
        ]
        sorted_paths = sorted(paths, key=MemoryClusterParser._extract_timestamp_key)
        assert sorted_paths[0].endswith("20250101_120000_ascend_pt")
        assert sorted_paths[1].endswith("20250101_230000_ascend_pt")
        assert sorted_paths[2].endswith("20250102_010000_ascend_pt")


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestMemoryParserEndToEnd:
    def test_full_pipeline(self, tmp_path):
        input_path = _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: input_path, Constant.RANK_LIST: "all"}
        )

        data_maps = parser.allocate_prof_data(input_path)
        assert len(data_maps) == 1
        assert data_maps[0]["role"] == "actor_update"
        assert data_maps[0]["rank_id"] == 0

        events = parser.parse_analysis_data(
            data_maps[0]["profiler_data_path"],
            data_maps[0]["rank_id"],
            data_maps[0]["role"],
        )

        assert len(events) == 5

        aten_empty_events = [e for e in events if e["name"] == "aten::empty"]
        assert len(aten_empty_events) == 3

        first_empty = [e for e in aten_empty_events if e["size_kb"] == 1024.0][0]
        assert first_empty["call_stack"] != ""
        assert "fsdp2.py(112)" in first_empty["call_stack_top"]

        matmul_events = [e for e in events if e["name"] == "aten::matmul"]
        assert len(matmul_events) == 1
        assert matmul_events[0]["call_stack"] != ""

        unknown_events = [e for e in events if e["name"] == "aten::unknown"]
        assert len(unknown_events) == 1
        assert unknown_events[0]["call_stack"] == ""
        assert unknown_events[0]["call_stack_top"] == ""

        release_events = [e for e in events if e["size_kb"] < 0]
        assert len(release_events) == 1

    def test_multiple_roles(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )
        _create_ascend_profile_dir(
            tmp_path,
            role="rollout_generate",
            rank_id=1,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )

        data_maps = parser.allocate_prof_data(str(tmp_path))
        assert len(data_maps) == 2

        roles = {dm["role"] for dm in data_maps}
        assert roles == {"actor_update", "rollout_generate"}

    def test_missing_trace_view(self, tmp_path):
        role_dir = tmp_path / "actor"
        role_dir.mkdir()
        ascend_pt_dir = role_dir / "20250101_120000_ascend_pt"
        ascend_pt_dir.mkdir()
        (ascend_pt_dir / "profiler_info_0.json").write_text("{}")
        (ascend_pt_dir / "profiler_metadata.json").write_text(
            json.dumps({"role": "actor"})
        )
        output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
        output_dir.mkdir()
        _write_operator_memory_csv(
            str(output_dir / "operator_memory.csv"), SAMPLE_MEMORY_ROWS
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )

        events = parser.parse_analysis_data(str(output_dir), 0, "actor")
        assert len(events) == 0

    def test_missing_operator_memory(self, tmp_path):
        role_dir = tmp_path / "actor"
        role_dir.mkdir()
        ascend_pt_dir = role_dir / "20250101_120000_ascend_pt"
        ascend_pt_dir.mkdir()
        (ascend_pt_dir / "profiler_info_0.json").write_text("{}")
        (ascend_pt_dir / "profiler_metadata.json").write_text(
            json.dumps({"role": "actor"})
        )
        output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
        output_dir.mkdir()
        _write_trace_view_json(str(output_dir / "trace_view.json"), SAMPLE_TRACE_EVENTS)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )

        events = parser.parse_analysis_data(str(output_dir), 0, "actor")
        assert len(events) == 0

    def test_empty_profiler_data_path(self):
        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: "/tmp", Constant.RANK_LIST: "all"}
        )

        events = parser.parse_analysis_data("", 0, "actor")
        assert len(events) == 0

    def test_non_all_rank_list(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "0,1"}
        )

        data_maps = parser.allocate_prof_data(str(tmp_path))
        assert len(data_maps) == 0


# =============================================================================
# Phase Classification Tests (G2)
# =============================================================================


class TestPhaseClassifier:
    """Tests for :class:`recipe.utils.phase_classifier.MemoryPhaseClassifier`."""

    def setup_method(self):
        self.clf = MemoryPhaseClassifier()

    @pytest.mark.parametrize(
        "role,expected",
        [
            ("actor_update", PHASE_TRAINING),
            ("critic_update", PHASE_TRAINING),
            ("actor_train", PHASE_TRAINING),
            ("backward_pass", PHASE_TRAINING),
            ("optimizer_step", PHASE_TRAINING),
            ("optim_step", PHASE_TRAINING),
            ("grad_compute", PHASE_TRAINING),
            ("rollout_generate", PHASE_INFERENCE),
            ("actor_compute_log_prob", PHASE_INFERENCE),
            ("ref_compute_log_prob", PHASE_INFERENCE),
            ("reference_model", PHASE_INFERENCE),
            ("forward_pass", PHASE_INFERENCE),
            ("sample_decode", PHASE_INFERENCE),
            ("infer_only", PHASE_INFERENCE),
        ],
    )
    def test_classify_known_roles(self, role, expected):
        assert self.clf.classify(role) == expected

    def test_classify_training_takes_precedence(self):
        # A role that matches both an inference keyword (generate) and a
        # training keyword (update) should be classified as Training.
        assert self.clf.classify("actor_update_generate") == PHASE_TRAINING

    def test_classify_empty_role(self):
        assert self.clf.classify("") == PHASE_UNKNOWN
        assert self.clf.classify(None) == PHASE_UNKNOWN

    def test_classify_unknown_role(self):
        assert self.clf.classify("some_random_role") == PHASE_UNKNOWN

    def test_classify_case_insensitive(self):
        assert self.clf.classify("ACTOR_UPDATE") == PHASE_TRAINING
        assert self.clf.classify("Rollout_Generate") == PHASE_INFERENCE

    def test_classify_custom_keywords(self):
        clf = MemoryPhaseClassifier(
            training_keywords=("foo",),
            inference_keywords=("bar",),
        )
        assert clf.classify("my_foo_role") == PHASE_TRAINING
        assert clf.classify("my_bar_role") == PHASE_INFERENCE
        # Default keywords no longer apply
        assert clf.classify("actor_update") == PHASE_UNKNOWN


class TestComputeStatistics:
    """Tests for :meth:`MemoryPhaseClassifier.compute_statistics`."""

    def setup_method(self):
        self.clf = MemoryPhaseClassifier()

    def _make_df(self, rows):
        import pandas as pd

        return pd.DataFrame(rows)

    def test_empty_dataframe_returns_empty_dict(self):
        import pandas as pd

        assert self.clf.compute_statistics(pd.DataFrame()) == {}

    def test_none_returns_empty_dict(self):
        assert self.clf.compute_statistics(None) == {}

    def test_stats_for_single_phase(self):
        df = self._make_df(
            [
                {
                    "role": "actor_update",
                    "size_kb": 1024.0,
                    "start_time_ms": 100.0,
                    "duration_ms": 50.0,
                    "total_allocated_mb": 10.0,
                    "total_active_mb": 5.0,
                },
                {
                    "role": "actor_update",
                    "size_kb": -512.0,
                    "start_time_ms": 200.0,
                    "duration_ms": 10.0,
                    "total_allocated_mb": 8.0,
                    "total_active_mb": 4.0,
                },
            ]
        )
        clf = MemoryPhaseClassifier()
        stats = clf.compute_statistics(df)

        assert list(stats.keys()) == ["Training"]
        s = stats["Training"]
        assert s["phase"] == "Training"
        assert s["roles"] == ["actor_update"]
        assert s["event_count"] == 2
        assert s["alloc_count"] == 1
        assert s["dealloc_count"] == 1
        assert s["alloc_kb"] == pytest.approx(1024.0)
        assert s["dealloc_kb"] == pytest.approx(512.0)
        assert s["net_kb"] == pytest.approx(512.0)
        assert s["alloc_mb"] == pytest.approx(1.0)
        assert s["peak_allocated_mb"] == pytest.approx(10.0)
        assert s["peak_active_mb"] == pytest.approx(5.0)
        assert s["t_start_ms"] == pytest.approx(100.0)
        assert s["t_end_ms"] == pytest.approx(210.0)

    def test_stats_for_multiple_phases(self):
        df = self._make_df(
            [
                {
                    "role": "actor_update",
                    "size_kb": 2048.0,
                    "start_time_ms": 100.0,
                    "duration_ms": 50.0,
                    "total_allocated_mb": 20.0,
                    "total_active_mb": 10.0,
                },
                {
                    "role": "rollout_generate",
                    "size_kb": 1024.0,
                    "start_time_ms": 300.0,
                    "duration_ms": 30.0,
                    "total_allocated_mb": 15.0,
                    "total_active_mb": 7.0,
                },
                {
                    "role": "rollout_generate",
                    "size_kb": -1024.0,
                    "start_time_ms": 400.0,
                    "duration_ms": 5.0,
                    "total_allocated_mb": 12.0,
                    "total_active_mb": 6.0,
                },
            ]
        )
        clf = MemoryPhaseClassifier()
        stats = clf.compute_statistics(df)

        assert set(stats.keys()) == {"Training", "Inference"}
        assert stats["Training"]["event_count"] == 1
        assert stats["Training"]["alloc_mb"] == pytest.approx(2.0)
        assert stats["Inference"]["event_count"] == 2
        assert stats["Inference"]["alloc_count"] == 1
        assert stats["Inference"]["dealloc_count"] == 1
        assert stats["Inference"]["roles"] == ["rollout_generate"]

    def test_stats_uses_existing_phase_column(self):
        df = self._make_df(
            [
                {
                    "role": "actor_update",
                    "phase": "Inference",  # explicit override
                    "size_kb": 1024.0,
                    "start_time_ms": 100.0,
                    "duration_ms": 50.0,
                    "total_allocated_mb": 10.0,
                    "total_active_mb": 5.0,
                },
            ]
        )
        clf = MemoryPhaseClassifier()
        stats = clf.compute_statistics(df)
        # Should respect the pre-existing phase column, not re-classify.
        assert list(stats.keys()) == ["Inference"]
        assert stats["Inference"]["roles"] == ["actor_update"]


class TestParserPhaseIntegration:
    """Integration: the parser populates ``phase`` and exposes stats (G2)."""

    def test_phase_field_populated_in_results(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        # ``run`` drives allocate → map → reduce, populating ``events_summary``.
        # Single rank ⇒ serial processing (no ProcessPoolExecutor).
        df = parser.run(str(tmp_path))

        assert df is not None and len(df) > 0
        assert "phase" in df.columns
        assert (df["phase"] == "Training").all()

    def test_phase_inference_for_rollout(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="rollout_generate",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        df = parser.run(str(tmp_path))

        assert df is not None and len(df) > 0
        assert "phase" in df.columns
        assert (df["phase"] == "Inference").all()

    def test_compute_phase_statistics_returns_training(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        parser.run(str(tmp_path))

        stats = parser.compute_phase_statistics()
        assert "Training" in stats
        assert stats["Training"]["roles"] == ["actor_update"]
        assert stats["Training"]["event_count"] == len(parser.events_summary)

    def test_compute_phase_statistics_no_events_returns_empty(self):
        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: "/tmp", Constant.RANK_LIST: "all"}
        )
        assert parser.compute_phase_statistics() == {}

    def test_compute_phase_statistics_multiple_phases(self, tmp_path):
        # Parse two roles directly (avoids ProcessPoolExecutor in tests) and
        # assemble ``events_summary`` to verify multi-phase statistics.
        import pandas as pd

        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )
        _create_ascend_profile_dir(
            tmp_path,
            role="rollout_generate",
            rank_id=1,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        out_actor = (
            tmp_path
            / "actor_update"
            / "20250101_120000_ascend_pt"
            / "ASCEND_PROFILER_OUTPUT"
        )
        out_rollout = (
            tmp_path
            / "rollout_generate"
            / "20250101_120000_ascend_pt"
            / "ASCEND_PROFILER_OUTPUT"
        )

        events = []
        events += parser.parse_analysis_data(str(out_actor), 0, "actor_update")
        events += parser.parse_analysis_data(str(out_rollout), 1, "rollout_generate")
        parser.events_summary = pd.DataFrame(events)

        stats = parser.compute_phase_statistics()
        assert set(stats.keys()) == {"Training", "Inference"}
        assert stats["Training"]["roles"] == ["actor_update"]
        assert stats["Inference"]["roles"] == ["rollout_generate"]