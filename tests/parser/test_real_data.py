import json
import os
import bisect
import csv
import ijson
from collections import Counter, defaultdict
from rl_insight.parser import MemoryClusterParser
from rl_insight.utils.schema import Constant

profiler_data_path = os.path.join("data", "memory_data", "mem_profile", "ASCEND_PROFILER_OUTPUT")

parser = MemoryClusterParser(
    {Constant.INPUT_PATH: ".", Constant.RANK_LIST: "all"}
)

events = parser.parse_analysis_data(profiler_data_path, rank_id=7, role="actor_update")

matched = [e for e in events if e["call_stack"] != ""]
unmatched = [e for e in events if e["call_stack"] == ""]
print(f"Total: {len(events)}, Matched: {len(matched)}, Unmatched: {len(unmatched)}")

trace_with_cs = defaultdict(list)
trace_without_cs = defaultdict(list)

with open(os.path.join(profiler_data_path, "trace_view.json"), "rb") as f:
    for ev in ijson.items(f, "item"):
        if ev.get("cat") != "cpu_op":
            continue
        name = ev.get("name", "")
        ts = float(ev["ts"])
        args = ev.get("args", {})
        has_cs = isinstance(args, dict) and "Call stack" in args
        if has_cs:
            trace_with_cs[name].append(ts)
        else:
            trace_without_cs[name].append(ts)

for name in trace_with_cs:
    trace_with_cs[name].sort()
for name in trace_without_cs:
    trace_without_cs[name].sort()

reason_name_not_found = []
reason_no_cs_before = []
reason_ts_too_late = []

for e in unmatched:
    name = e["name"]
    alloc_us = e["allocation_time_ms"] * 1000

    if name not in trace_with_cs and name not in trace_without_cs:
        reason_name_not_found.append(e)
    elif name in trace_without_cs and name not in trace_with_cs:
        ts_list = trace_without_cs[name]
        idx = bisect.bisect_right(ts_list, alloc_us) - 1
        if idx >= 0:
            reason_no_cs_before.append(e)
        else:
            reason_ts_too_late.append(e)
    elif name in trace_with_cs:
        ts_list = trace_with_cs[name]
        idx = bisect.bisect_right(ts_list, alloc_us) - 1
        if idx < 0:
            reason_ts_too_late.append(e)

total_unmatched = len(unmatched)
print(f"\n=== 未匹配原因分析 ===")
print(f"未匹配总数: {total_unmatched}")
print(f"  算子名在 trace_view 中完全不存在: {len(reason_name_not_found)} ({len(reason_name_not_found)/total_unmatched*100:.1f}%)")
print(f"  算子存在但无 Call stack (时间上有前置): {len(reason_no_cs_before)} ({len(reason_no_cs_before)/total_unmatched*100:.1f}%)")
print(f"  算子有 Call stack 但时间上无前置 (ts > alloc_time): {len(reason_ts_too_late)} ({len(reason_ts_too_late)/total_unmatched*100:.1f}%)")
print(f"  其他 (有CS但时间无前置 + 无CS且时间无前置): {total_unmatched - len(reason_name_not_found) - len(reason_no_cs_before) - len(reason_ts_too_late)}")

counter_name_not_found = Counter(e["name"] for e in reason_name_not_found)
counter_no_cs = Counter(e["name"] for e in reason_no_cs_before)
counter_ts_late = Counter(e["name"] for e in reason_ts_too_late)

if counter_name_not_found:
    print(f"\n--- 算子名完全不存在 (共 {len(counter_name_not_found)} 种) ---")
    for name, cnt in counter_name_not_found.most_common(20):
        print(f"  {name}: {cnt} 条")

if counter_no_cs:
    print(f"\n--- 算子存在但无 Call stack (共 {len(counter_no_cs)} 种) ---")
    for name, cnt in counter_no_cs.most_common(20):
        print(f"  {name}: {cnt} 条")

if counter_ts_late:
    print(f"\n--- 算子有 CS 但时间无前置 (共 {len(counter_ts_late)} 种) ---")
    for name, cnt in counter_ts_too_late.most_common(20):
        print(f"  {name}: {cnt} 条")