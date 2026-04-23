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

import bisect
import csv
import ijson
from loguru import logger
import os
from collections import defaultdict

from .ascend_parser import AscendClusterParser
from .parser import register_cluster_parser
from rl_insight.utils.schema import Constant, MemoryEventRow
from rl_insight.data import DataEnum


@register_cluster_parser("memory")
class MemoryClusterParser(AscendClusterParser):
    input_type: DataEnum = DataEnum.ASCEND_MEMORY
    sort_key: str = "allocation_time_ms"

    def __init__(self, params) -> None:
        super().__init__(params)

    @staticmethod
    def extract_call_stack_top(call_stack: str) -> str:
        return call_stack.split(";\r\n")[0] if call_stack else ""

    def parse_analysis_data(
        self, profiler_data_path: str, rank_id: int, role: str
    ) -> list[MemoryEventRow]:
        if not profiler_data_path:
            logger.warning(f"Rank {rank_id}: profiler_data_path is empty")
            return []

        trace_view_path = os.path.join(profiler_data_path, "trace_view.json")
        operator_memory_path = os.path.join(profiler_data_path, "operator_memory.csv")

        if not os.path.exists(trace_view_path):
            logger.warning(
                f"Rank {rank_id}: trace_view.json not found at {trace_view_path}"
            )
            return []

        if not os.path.exists(operator_memory_path):
            logger.warning(
                f"Rank {rank_id}: operator_memory.csv not found at {operator_memory_path}"
            )
            return []

        call_stack_index = self._build_call_stack_index(trace_view_path)

        results = self._parse_operator_memory(
            operator_memory_path, call_stack_index, rank_id, role
        )

        logger.info(
            f"Rank {rank_id} Role {role}: parsed {len(results)} memory events"
        )
        return results

    def _build_call_stack_index(self, trace_view_path: str) -> dict:
        # 索引结构: name → {"ts_list": [...], "entries": [...]}
        # ts_list 预计算并按 ts 升序排列，后续匹配时直接用于二分查找
        # entries 与 ts_list 一一对应，通过索引关联
        index: dict[str, dict] = defaultdict(lambda: {"ts_list": [], "entries": []})

        # 使用 ijson 流式解析，避免将整个 JSON 一次性加载到内存
        # trace_view.json 可能达到数百 MB，全量加载会导致内存溢出
        with open(trace_view_path, "rb") as f:
            for event in ijson.items(f, "item"):
                # 只关注 cpu_op 类型事件，其他类型（如 communication、kernel）不含调用栈
                if event.get("cat") != "cpu_op":
                    continue

                # 只有 args 中包含 "Call stack" 字段的事件才有调用栈信息
                # 部分 cpu_op 事件可能缺少该字段，需跳过
                args = event.get("args", {})
                if not isinstance(args, dict) or "Call stack" not in args:
                    continue

                name = event.get("name", "")
                # ts 在 JSON 中是字符串类型（如 "1755143611835441.990"），需转为 float
                ts = float(event["ts"])
                dur = float(event.get("dur", 0))
                call_stack = args["Call stack"]

                # 按 name 分组追加，同一算子每次调用产生一条记录
                index[name]["entries"].append(
                    {"ts": ts, "dur": dur, "call_stack": call_stack}
                )

        # 组内按 ts 升序排序，确保后续二分查找的正确性
        # 排序后，对任意 allocation_time，可以用 bisect_right 快速定位
        # ts ≤ allocation_time 的最后一条即为最近匹配
        # 同时预计算 ts_list，避免 _match_call_stack 每次调用时重复构建
        for name in index:
            entries = index[name]["entries"]
            entries.sort(key=lambda x: x["ts"])
            index[name]["ts_list"] = [e["ts"] for e in entries]

        return dict(index)

    def _parse_operator_memory(
        self,
        csv_path: str,
        call_stack_index: dict,
        rank_id: int,
        role: str,
    ) -> list[MemoryEventRow]:
        results: list[MemoryEventRow] = []
        us_to_ms = Constant.US_TO_MS

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Size(KB): 正数表示内存申请，负数表示内存释放，均需保留
                size_kb = float(row["Size(KB)"].strip())

                # Allocation Time(us): CSV 中该字段末尾可能含制表符，需 strip
                allocation_time_us = float(row["Allocation Time(us)"].strip())

                # 调用栈匹配: 在 call_stack_index 中查找同名算子中
                # ts ≤ allocation_time 的最近一条记录
                # 未匹配到时返回空字符串
                call_stack, call_stack_top = self._match_call_stack(
                    row["Name"].strip(), allocation_time_us, call_stack_index
                )

                # Duration(us) 可能为空（内存尚未释放），空值时 duration_ms = 0
                duration_us = row.get("Duration(us)", "").strip()
                duration_ms = float(duration_us) / us_to_ms if duration_us else 0.0

                # 构建 MemoryEventRow，时间统一转为毫秒
                results.append(
                    MemoryEventRow(
                        name=row["Name"].strip(),
                        role=role,
                        rank_id=rank_id,
                        call_stack=call_stack,
                        call_stack_top=call_stack_top,
                        size_kb=size_kb,
                        allocation_time_ms=allocation_time_us / us_to_ms,
                        duration_ms=duration_ms,
                        total_allocated_mb=float(
                            row["Allocation Total Allocated(MB)"].strip()
                        ),
                        total_reserved_mb=float(
                            row["Allocation Total Reserved(MB)"].strip()
                        ),
                        total_active_mb=float(
                            row["Allocation Total Active(MB)"].strip()
                        ),
                        device_type=row["Device Type"].strip(),
                    )
                )

        return results

    def _match_call_stack(
        self,
        name: str,
        allocation_time_us: float,
        call_stack_index: dict,
    ) -> tuple[str, str]:
        """
        返回 (call_stack, call_stack_top)
        未命中返回 ("", "")
        """
        if name not in call_stack_index:
            return "", ""

        group = call_stack_index[name]
        ts_list = group["ts_list"]
        entries = group["entries"]

        # 二分查找: 找 ts ≤ allocation_time 的最近一条
        # bisect_right 返回第一个大于 allocation_time 的位置，
        # 减 1 即为最后一个小于等于 allocation_time 的位置
        idx = bisect.bisect_right(ts_list, allocation_time_us) - 1
        if idx < 0:
            # 所有记录的 ts 都大于 allocation_time，无匹配
            return "", ""

        entry = entries[idx]
        call_stack = entry["call_stack"]
        call_stack_top = self.extract_call_stack_top(call_stack)
        return call_stack, call_stack_top