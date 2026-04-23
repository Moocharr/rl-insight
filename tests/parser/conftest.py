import csv
import json
import os

import pandas as pd
import pytest


OPERATOR_MEMORY_COLUMNS = [
    "Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)",
    "Active Release Time(us)", "Duration(us)",
    "Allocation Total Allocated(MB)", "Allocation Total Reserved(MB)",
    "Allocation Total Active(MB)", "Release Total Allocated(MB)",
    "Release Total Reserved(MB)", "Release Total Active(MB)",
    "Stream Ptr", "Device Type",
]

TRACE_VIEW_CSV_COLUMNS = [
    "ph", "name", "cat", "pid", "tid", "ts_us", "dur_us",
    "ts_end_us", "has_call_stack", "call_stack_top", "call_stack",
    "seq_num", "fwd_thread_id",
]

SAMPLE_TRACE_EVENTS = [
    {
        "ph": "X", "name": "aten::empty", "cat": "cpu_op",
        "pid": 1, "tid": 1, "ts": "1000000.0", "dur": 500.0,
        "args": {"Call stack": "fsdp2.py(112): train_batch;\r\nmodel.py(50): forward"},
    },
    {
        "ph": "X", "name": "aten::matmul", "cat": "cpu_op",
        "pid": 1, "tid": 1, "ts": "2000000.0", "dur": 1000.0,
        "args": {"Call stack": "model.py(60): forward;\r\nlayer.py(30): __call__"},
    },
    {
        "ph": "X", "name": "aten::empty", "cat": "cpu_op",
        "pid": 1, "tid": 1, "ts": "3000000.0", "dur": 300.0,
        "args": {"Call stack": "fsdp2.py(120): train_batch;\r\nmodel.py(55): forward"},
    },
    {
        "ph": "X", "name": "aten::relu", "cat": "kernel",
        "pid": 1, "tid": 1, "ts": "4000000.0", "dur": 200.0,
    },
    {
        "ph": "X", "name": "aten::cumsum", "cat": "cpu_op",
        "pid": 1, "tid": 1, "ts": "5000000.0", "dur": 800.0,
    },
]

SAMPLE_MEMORY_ROWS = [
    ["aten::empty", 1024.0, "1000100.0", "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
    ["aten::empty", 2048.0, "3000050.0", "", "", "", 200.0, 300.0, 100.0, "", "", "", "123", "NPU:0"],
    ["aten::matmul", 4096.0, "2000500.0", "", "", "", 300.0, 400.0, 150.0, "", "", "", "123", "NPU:0"],
    ["aten::unknown", 512.0, "6000000.0", "", "", "", 50.0, 100.0, 25.0, "", "", "", "123", "NPU:0"],
    ["aten::empty", -1024.0, "7000000.0", "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
]


def write_operator_memory_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OPERATOR_MEMORY_COLUMNS)
        for row in rows:
            writer.writerow(row)


def write_operator_memory_csv_pandas(path, rows):
    df = pd.DataFrame(rows, columns=OPERATOR_MEMORY_COLUMNS)
    df.to_csv(path, index=False)


def write_trace_view_csv(path, rows):
    df = pd.DataFrame(rows, columns=TRACE_VIEW_CSV_COLUMNS)
    df.to_csv(path, index=False)


def write_trace_view_json(path, events):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f)


def create_ascend_profile_dir(tmp_path, role="actor_update", rank_id=0,
                                trace_events=None, memory_rows=None):
    ascend_pt_dir = tmp_path / f"{role}_ascend_pt"
    ascend_pt_dir.mkdir()
    profiler_info = ascend_pt_dir / f"profiler_info_{rank_id}.json"
    profiler_info.write_text("{}")
    metadata = ascend_pt_dir / "profiler_metadata.json"
    metadata.write_text(json.dumps({"role": role}))

    output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
    output_dir.mkdir()

    if trace_events is not None:
        write_trace_view_json(str(output_dir / "trace_view.json"), trace_events)
    if memory_rows is not None:
        write_operator_memory_csv(str(output_dir / "operator_memory.csv"), memory_rows)

    return str(tmp_path)


@pytest.fixture
def data_dir(tmp_path):
    operator_memory_path = str(tmp_path / "operator_memory.csv")
    trace_view_path = str(tmp_path / "trace_view.csv")
    output_path = str(tmp_path / "match_result.csv")
    return operator_memory_path, trace_view_path, output_path