import os

import pandas as pd
import pytest

from check_call_stack_coverage import match_memory_with_trace, save_match_result
from .conftest import write_operator_memory_csv_pandas, write_trace_view_csv


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
            operator_memory_path, trace_view_path, output_path
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

        results, unmatched = _match(operator_memory_path, trace_view_path)

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

        results, unmatched = _match(operator_memory_path, trace_view_path)

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

        results, unmatched = _match(operator_memory_path, trace_view_path)

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

        results, unmatched = _match(operator_memory_path, trace_view_path)

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

        results, _, _, _, _ = match_memory_with_trace(operator_memory_path, trace_view_path)
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

        results, unmatched = _match(operator_memory_path, trace_view_path)

        assert len(results) == 0
        assert len(unmatched) == 0

    def test_empty_trace_view(self, data_dir):
        operator_memory_path, trace_view_path, output_path = data_dir

        write_operator_memory_csv_pandas(operator_memory_path, [
            ["aten::empty", 1024.0, 1000.0, "", "", "", 100.0, 200.0, 50.0, "", "", "", "123", "NPU:0"],
        ])
        write_trace_view_csv(trace_view_path, [])

        results, unmatched = _match(operator_memory_path, trace_view_path)

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

        results, unmatched = _match(operator_memory_path, trace_view_path)

        assert len(results) == 1
        assert results.iloc[0]["matched"] == True