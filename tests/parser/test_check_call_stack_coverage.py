import bisect
import os
from collections import defaultdict

import pandas as pd
import pytest

from .conftest import write_operator_memory_csv_pandas, write_trace_view_csv


def match_memory_with_trace(operator_memory_path, trace_view_path):
    df_memory = pd.read_csv(operator_memory_path)
    df_trace = pd.read_csv(trace_view_path)

    if len(df_memory) == 0:
        return pd.DataFrame(), []

    df_memory['Allocation Time(us)'] = pd.to_numeric(
        df_memory['Allocation Time(us)'], errors='coerce'
    )
    df_trace['ts_us'] = pd.to_numeric(df_trace['ts_us'], errors='coerce')
    df_trace['ts_end_us'] = pd.to_numeric(df_trace['ts_end_us'], errors='coerce')

    trace_by_name = defaultdict(list)
    for idx, row in df_trace.iterrows():
        trace_by_name[row['name']].append({
            'trace_idx': idx,
            'ts_us': row['ts_us'],
            'ts_end_us': row['ts_end_us'],
            'dur_us': row['dur_us'],
            'cat': row['cat'],
            'call_stack': row.get('call_stack', '')
        })

    for name in trace_by_name:
        trace_by_name[name].sort(key=lambda x: x['ts_us'])

    trace_ts_index = {}
    for name, records in trace_by_name.items():
        trace_ts_index[name] = [r['ts_us'] for r in records]

    results = []
    unmatched_records = []

    for mem_idx, mem_row in df_memory.iterrows():
        mem_name = mem_row['Name']
        mem_time = mem_row['Allocation Time(us)']
        mem_size = mem_row['Size(KB)']

        best_match = None
        min_time_diff = float('inf')

        if mem_name in trace_by_name:
            ts_list = trace_ts_index[mem_name]
            idx = bisect.bisect_right(ts_list, mem_time) - 1
            if idx >= 0:
                best_match = trace_by_name[mem_name][idx]
                min_time_diff = mem_time - best_match['ts_us']

        if best_match:
            results.append({
                'mem_idx': mem_idx,
                'mem_name': mem_name,
                'mem_size_kb': mem_size,
                'mem_allocation_time': mem_time,
                'matched': True,
                'trace_idx': best_match['trace_idx'],
                'trace_ts_us': best_match['ts_us'],
                'trace_ts_end_us': best_match['ts_end_us'],
                'trace_dur_us': best_match['dur_us'],
                'trace_cat': best_match['cat'],
                'time_diff_us': min_time_diff,
                'call_stack': best_match['call_stack']
            })
        else:
            unmatched_records.append({
                'mem_idx': mem_idx,
                'mem_name': mem_name,
                'mem_size_kb': mem_size,
                'mem_allocation_time': mem_time
            })
            results.append({
                'mem_idx': mem_idx,
                'mem_name': mem_name,
                'mem_size_kb': mem_size,
                'mem_allocation_time': mem_time,
                'matched': False,
                'trace_idx': None,
                'trace_ts_us': None,
                'trace_ts_end_us': None,
                'trace_dur_us': None,
                'trace_cat': None,
                'time_diff_us': None,
                'call_stack': None
            })

    return pd.DataFrame(results), unmatched_records


def save_match_result(results_df, output_path):
    results_df.to_csv(output_path, index=False)


class TestMatchMemoryWithTraceBasic:
    def test_exact_match(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1000.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 900.0, 200.0, 1100.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 1
        assert results.iloc[0]["matched"] == True
        assert results.iloc[0]["call_stack"] == "stack"
        assert len(unmatched) == 0

    def test_no_name_match(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1000.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::matmul", "cpu_op", 1, 1, 900.0, 200.0, 1100.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 1
        assert results.iloc[0]["matched"] == False
        assert len(unmatched) == 1
        assert unmatched[0]["mem_name"] == "aten::empty"

    def test_no_ts_match(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 500.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 900.0, 200.0, 1100.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 1
        assert results.iloc[0]["matched"] == False
        assert len(unmatched) == 1

    def test_closest_ts_match(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 2000.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 500.0, 100.0, 600.0, "yes", "top1", "stack1", -1, 0],
            ["X", "aten::empty", "cpu_op", 1, 1, 1500.0, 100.0, 1600.0, "yes", "top2", "stack2", -1, 0],
            ["X", "aten::empty", "cpu_op", 1, 1, 1800.0, 100.0, 1900.0, "yes", "top3", "stack3", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 1
        assert results.iloc[0]["matched"] == True
        assert results.iloc[0]["call_stack"] == "stack3"
        assert results.iloc[0]["time_diff_us"] == pytest.approx(200.0)


class TestMatchMemoryWithTraceMultiple:
    def test_multiple_memory_one_trace(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1500.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
            ["aten::empty", 2048.0, 1600.0, "", "", "", 200.0, 300.0, 100.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 1000.0, 800.0, 1800.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 2
        assert all(results["matched"])
        assert len(unmatched) == 0

    def test_mixed_matched_unmatched(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1500.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
            ["aten::unknown", 512.0, 1500.0, "", "", "", 50.0, 100.0, 25.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 1000.0, 800.0, 1800.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 2
        matched_rows = results[results["matched"] == True]
        unmatched_rows = results[results["matched"] == False]
        assert len(matched_rows) == 1
        assert len(unmatched_rows) == 1
        assert len(unmatched) == 1

    def test_output_csv_written(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1500.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 1000.0, 800.0, 1800.0, "yes", "top", "stack", -1, 0],
        ])

        results, _ = match_memory_with_trace(operator_memory_path, trace_view_path)
        save_match_result(results, output_path)

        assert os.path.exists(output_path)
        output_df = pd.read_csv(output_path)
        assert len(output_df) == 1
        assert output_df.iloc[0]["matched"] == True


class TestMatchMemoryWithTraceEdgeCases:
    def test_empty_operator_memory(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 900.0, 200.0, 1100.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 0
        assert len(unmatched) == 0

    def test_empty_trace_view(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1000.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 1
        assert results.iloc[0]["matched"] == False
        assert len(unmatched) == 1

    def test_negative_size_release(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", -1024.0, 2000.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [
            ["X", "aten::empty", "cpu_op", 1, 1, 1000.0, 800.0, 1800.0, "yes", "top", "stack", -1, 0],
        ])

        results, unmatched = match_memory_with_trace(
            operator_memory_path, trace_view_path
        )

        assert len(results) == 1
        assert results.iloc[0]["matched"] == True