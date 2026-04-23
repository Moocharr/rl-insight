# 内存分析 Parser 需求分析与设计文档

## 1. 背景与目标

当前 `rl-insight` 已有时序分析 Parser（`mstx` / `torch`），基于 `EventRow` 数据模型输出算子级时间线事件。现需新增**内存分析 Parser**，从 Ascend Profiler 输出的内存相关文件中提取内存分配信息，为 RL 训练的内存瓶颈分析提供数据支撑。

**核心目标**：在每条内存申请记录中补全以下信息：

- 调用栈（Call Stack）
- 内存申请大小（Size）
- 内存申请时间（Allocation Time）
- 内存占用时长（Duration）

---

## 2. 数据源分析

### 2.1 数据目录结构

```text
<role_name>_ascend_pt/
  ├── profiler_info_<rank_id>.json    ← 提取 rank_id
  ├── profiler_metadata.json          ← 提取 role
  └── ASCEND_PROFILER_OUTPUT/
        ├── operator_memory.csv        ← 【主数据源】算子级内存分配记录
        └── trace_view.json            ← 【调用栈来源】完整时间线事件（含 Call stack）
```

### 2.2 operator_memory.csv（主数据源）

| 字段 | 类型 | 说明 |
|------|------|------|
| Name | str | 算子名称 |
| Size(KB) | float | 正数=申请，负数=释放 |
| Allocation Time(us) | float | 申请/释放时间戳（微秒） |
| Release Time(us) | float | 释放时间（可能为空） |
| Duration(us) | float | 占用时长（可能为空） |
| Allocation Total Allocated(MB) | float | 申请时刻累计已分配 |
| Allocation Total Reserved(MB) | float | 申请时刻累计预留 |
| Allocation Total Active(MB) | float | 申请时刻累计活跃 |
| Release Total Allocated/Reserved/Active(MB) | float | 释放时刻累计指标 |
| Stream Ptr | str | 流指针 |
| Device Type | str | 设备类型 |

**关键特征**：

- Size > 0 为内存申请，Size < 0 为内存释放
- Release Time / Duration 可能为空（内存尚未释放）
- 时间戳精度为微秒（us）
- CSV 中时间戳字段末尾含制表符，解析时需 strip

### 2.3 trace_view.json（调用栈来源）

| 字段 | 类型 | 说明 |
|------|------|------|
| ph | str | 事件类型（X=Complete Event） |
| name | str | 算子名称 |
| ts | str | 开始时间戳（微秒） |
| dur | float | 持续时间（微秒） |
| cat | str | 事件类别 |
| args.Call stack | str | Python 调用栈 |

**关键特征**：

- 文件可能较大，**必须流式解析**
- `ts` 是算子开始执行时间，`Allocation Time` 是算子内触发内存分配的时间，因此 `Allocation Time ≥ ts`
- 同一算子名可能有多条记录（每次调用一条），可通过 ts 精确匹配
- 只有 `cat=="cpu_op"` 且 `args` 中含 `"Call stack"` 的事件才有调用栈信息

---

## 3. 数据模型设计

### 3.1 MemoryEventRow

```python
class MemoryEventRow(TypedDict):
    name: str                    # 算子名称
    role: str                    # RL 角色
    rank_id: int                 # 进程 rank
    call_stack: str              # 完整调用栈
    call_stack_top: str          # 调用栈顶层（用户代码入口）
    size_kb: float               # 内存大小（KB），正数=申请，负数=释放
    allocation_time_ms: float    # 内存申请时间（ms）
    duration_ms: float           # 内存占用时长（ms），0 表示未释放
    total_allocated_mb: float    # 申请时刻累计已分配内存（MB）
    total_reserved_mb: float     # 申请时刻累计预留内存（MB）
    total_active_mb: float       # 申请时刻累计活跃内存（MB）
    device_type: str             # 设备类型
```

**设计决策**：

- `size_kb`：正数表示内存申请，负数表示内存释放
- `duration_ms`：若 `Duration(us)` 有值则转换，无值则为 0
- `call_stack_top`：取调用栈第一行（用户代码入口），便于快速定位
- 时间单位统一为毫秒（ms），与现有 EventRow 保持一致

---

## 4. 数据关联策略

### 4.1 调用栈关联（核心）

**匹配方式**：按 `name` 查找同名 trace_view 记录，取 `ts ≤ Allocation Time` 中最近的一条

```text
operator_memory.csv:  name="aten::empty", Allocation Time=T_alloc
trace_view.json:      name="aten::empty", ts=T_start

匹配条件: name 相同 AND ts ≤ Allocation Time
匹配策略: 在满足条件的记录中，取 ts 最接近 Allocation Time 的一条
```

**匹配语义**：`ts` 是算子开始执行时间，`Allocation Time` 是算子内触发内存分配的时间。对于同名的 trace_view 记录，找到在 `Allocation Time` 之前最近开始的那条，即为触发该次内存分配的算子调用。

**允许多对一**：一个算子可能分配多次内存，因此多条 operator_memory 记录可以匹配到同一条 trace_view 记录。

### 4.2 未命中处理

未匹配到调用栈的记录，`call_stack` 和 `call_stack_top` 字段填空字符串。

---

## 5. Parser 设计

### 5.1 类结构

```text
BaseClusterParser (已有)
  └── MemoryClusterParser (新增, @register_cluster_parser("memory"))
        ├── allocate_prof_data()          → 扫描目录，构建 DataMap
        ├── parse_analysis_data()         → 主解析流程
        ├── _build_call_stack_index()     → 流式解析 trace_view.json，构建调用栈索引
        ├── _parse_operator_memory()      → 解析 operator_memory.csv，输出 MemoryEventRow
        └── _match_call_stack()           → name + ts 匹配调用栈
```

### 5.2 注册与数据类型

```python
@register_cluster_parser("memory")
class MemoryClusterParser(BaseClusterParser):
    input_type: DataEnum = DataEnum.ASCEND_MEMORY
```

DataEnum 新增：

```python
ASCEND_MEMORY = "ascend_memory"
SUMMARY_MEMORY_EVENT = "summary_memory_event"
```

### 5.3 allocate_prof_data 逻辑

复用 MstxClusterParser 的目录扫描逻辑：

1. 遍历 `input_path`，找到所有 `*_ascend_pt` 目录
2. 从 `profiler_metadata.json` 提取 `role`，从 `profiler_info_*.json` 提取 `rank_id`
3. `profiler_data_path` 指向 `ASCEND_PROFILER_OUTPUT` 目录

### 5.4 parse_analysis_data 核心流程

```text
输入: profiler_data_path (ASCEND_PROFILER_OUTPUT 目录), rank_id, role

步骤1: 构建调用栈索引
       流式解析 trace_view.json (ijson)
       → 筛选 cat=="cpu_op" 且含 "Call stack" 的事件
       → 按 name 分组，组内按 ts 排序
       → 只保留 name, ts, dur, call_stack 四个字段

步骤2: 解析 operator_memory.csv
       → 逐行构建 MemoryEventRow:
          a. 时间转换: us → ms (÷ 1000)
          b. 调用栈匹配: name + ts 匹配 trace_view 索引
             未命中 → call_stack = "", call_stack_top = ""
          c. 提取 call_stack_top: 取调用栈第一行
          d. duration_ms: Duration(us) 有值则转换，无值则为 0

步骤3: 返回 list[MemoryEventRow]
```

### 5.5 trace_view.json 流式解析设计

```python
import ijson

def _build_call_stack_index(self, trace_view_path: str) -> dict:
    # 索引结构: name → [{ts, dur, call_stack}, ...]
    # 按 name 分组，组内按 ts 排序，后续匹配时可用二分查找
    index: dict[str, list] = {}

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
            if name not in index:
                index[name] = []
            index[name].append({"ts": ts, "dur": dur, "call_stack": call_stack})

    # 组内按 ts 升序排序，确保后续二分查找的正确性
    # 排序后，对任意 allocation_time，可以用 bisect_right 快速定位
    # ts ≤ allocation_time 的最后一条即为最近匹配
    for name in index:
        index[name].sort(key=lambda x: x["ts"])

    return index
```

### 5.6 operator_memory.csv 解析

```python
def _parse_operator_memory(
    self,
    csv_path: str,
    call_stack_index: dict,
    rank_id: int,
    role: str,
) -> list[MemoryEventRow]:
    results: list[MemoryEventRow] = []
    us_to_ms = Constant.US_TO_MS  # 微秒转毫秒的除数: 1000

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
            # 释放事件的 Release Total 系列字段为空，
            # 申请事件的 Allocation Total 系列字段有值
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
                    total_allocated_mb=float(row["Allocation Total Allocated(MB)"].strip()),
                    total_reserved_mb=float(row["Allocation Total Reserved(MB)"].strip()),
                    total_active_mb=float(row["Allocation Total Active(MB)"].strip()),
                    device_type=row["Device Type"].strip(),
                )
            )

    return results
```

### 5.7 调用栈匹配算法

```python
import bisect

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

    entries = call_stack_index[name]
    ts_list = [e["ts"] for e in entries]

    # 二分查找: 找 ts ≤ allocation_time 的最近一条
    idx = bisect.bisect_right(ts_list, allocation_time_us) - 1
    if idx < 0:
        return "", ""

    entry = entries[idx]
    call_stack = entry["call_stack"]
    call_stack_top = call_stack.split(";\r\n")[0] if call_stack else ""
    return call_stack, call_stack_top
```

---

## 6. 文件结构

```text
rl_insight/parser/
├── parser.py              # BaseClusterParser (已有)
├── torch_parser.py        # TorchClusterParser (已有)
├── mstx_parser.py         # MstxClusterParser (已有)
└── memory_parser.py       # MemoryClusterParser (新增)

rl_insight/utils/schema.py
├── EventRow               # (已有)
└── MemoryEventRow         # (新增)
```

---

## 7. 依赖

| 库 | 用途 | 说明 |
|----|------|------|
| `ijson` | 流式解析大 JSON | 需安装: `pip install ijson` |
| `csv` | 解析 CSV | 标准库 |
| `bisect` | 二分查找调用栈 | 标准库 |