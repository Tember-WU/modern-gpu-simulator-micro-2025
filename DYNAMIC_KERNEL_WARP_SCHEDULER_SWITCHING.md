# Dynamic Kernel Warp Scheduler Policy Switching

## 目标

本修改为 trace-driven、remodeled SM 模式增加按 dynamic kernel launch 选择 warp scheduler policy 的能力。例如，backprop 的两个 dynamic kernel 可以配置为：

```text
dynamic kernel launch 1 -> LRR
dynamic kernel launch 2 -> GTO
```

对应命令行参数为：

```bash
-gpgpu_scheduler gto \
-dynamic_kernel_scheduler_map "1=lrr;2=gto"
```

`-gpgpu_scheduler` 是 fallback policy。没有出现在 map 中的 dynamic kernel 使用该 policy。如果 map 包含 launch 1，则 launch 1 的 policy 同时作为 remodeled Subcore 的初始 policy，而不是先用 fallback policy 冷启动、再在 launch 前切换。

当前实现只在 dynamic kernel launch 边界切换 policy；尚未实现同一个 kernel 内按照 kernel-relative cycle 切换。

## Dynamic kernel ID

Map 中的 ID 是 dynamic trace 中从 1 开始的全局 kernel launch 顺序，不是：

- kernel name；
- static function ID；
- CUDA stream 内部的 kernel ID；
- simulator 过滤 kernel 后重新生成的 UID。

`trace_parser::parse_commandlist_file()` 在解析 kernel event 时为 `trace_command` 保存 `dynamic_kernel_launch_id`。该 ID 随后传递给 `trace_kernel_info_t`，因此即使使用 kernel filter，保留下来的 kernel 仍使用原始 dynamic launch ID。

主要文件：

- `simulator-remodeled/gpu-simulator/trace-parser/trace_parser.h`
- `simulator-remodeled/gpu-simulator/trace-parser/trace_parser.cc`
- `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.h`
- `simulator-remodeled/gpu-simulator/main.cc`

## 配置格式

新增选项：

```text
-dynamic_kernel_scheduler_map "1=gto;2=lrr"
```

每一项使用 `<dynamic_launch_id>=<policy>`，不同项使用分号分隔。Policy 自身可以包含冒号：

```text
1=two_level_active:4:0:1;2=warp_limiting:2:8
```

解析结果保存为：

```cpp
std::map<unsigned, warp_scheduler_spec>
```

其中 `warp_scheduler_spec` 同时保存 concrete scheduler enum 和原始配置字符串。原始字符串用于 two-level active 和 warp-limiting 的参数解析。

配置解析会检查：

- dynamic launch ID 必须大于 0；
- dynamic launch ID 不能重复；
- policy 名称必须受支持；
- per-kernel map 只能用于 remodeled SM；
- 当前不允许与 concurrent kernel 模式同时使用。

## Policy 切换调用链

在 trace-driven 主循环准备 launch kernel 时执行：

```text
main.cc
  -> gpgpu_sim::configure_scheduler_for_dynamic_kernel()
     -> simt_core_cluster::set_warp_scheduler_policy()
        -> SM::set_warp_scheduler_policy()
           -> Subcore::set_warp_scheduler_policy()
```

切换发生在新 kernel 被 launch 之前。当前要求 `-gpgpu_concurrent_kernel_sm 0`，因此切换时前一个 dynamic kernel 已经完成，不会出现同一 SM 同时需要两个不同 policy 的情况。

## Runtime scheduler state

### 设计原则

Policy 切换被建模为动态控制器改变 warp selection logic，而不是重新启动 SM。因此切换时保留：

- physical warp slot 和 warp execution state；
- 最近 issue 的 warp：`m_greedy_pointer_issue`；
- 最近 fetch 的 warp：`m_greedy_pointer_fetch`；
- pipeline、scoreboard、barrier、cache 和 memory 状态。

只初始化目标 policy 自己拥有的状态。

这与旧实现中“policy 一变化就统一重置全部 scheduler pointer”的行为不同。统一冷重置会让 `LRR -> GTO` 的 GTO 从 warp 0 开始，而连续运行的 `GTO -> GTO` 会继承上一个 kernel 的 pointer，从而引入额外的、由切换实现本身造成的时序差异。

### 冷启动

`Subcore::finilized_warps_assignation()` 调用：

```cpp
initialize_scheduler_cold_start_state();
```

`Subcore` 构造时按照以下顺序选择冷启动 policy：

1. 如果 map 包含 dynamic launch 1，使用 `map[1]`；
2. 否则使用 `-gpgpu_scheduler` fallback policy。

因此，参数 `-gpgpu_scheduler gto -dynamic_kernel_scheduler_map "1=lrr;2=lrr"` 会直接按照 LRR 冷启动。launch 1 前再次配置 LRR 时，因为 policy 没有变化，不会触发 runtime transition。

冷启动仍保持原 remodeled scheduler 的初始行为：

- GTO 和 warp-limiting 从 local warp 0 开始；
- 其他 policy 的公共 pointer 从最后一个 local warp slot 开始；
- 然后初始化当前 policy 的专属状态。

### Dynamic policy transition

`Subcore::set_warp_scheduler_policy()` 只有在 `warp_scheduler_spec` 实际变化时才执行 transition：

```cpp
m_active_scheduler = spec;
initialize_active_policy_state(false);
```

它不会修改 `m_greedy_pointer_issue` 或 `m_greedy_pointer_fetch`。

各 policy 的转换语义如下。

| 目标 policy | Runtime transition 行为 |
|---|---|
| GTO | 保留最近 issue/fetch pointer，直接使用 full warp set 排序 |
| LRR | 保留最近 issue pointer，从其下一个 warp 继续 round-robin |
| Oldest | 无专属持久状态，按照当前 dynamic warp age 排序 |
| GTHID | 保留 greedy pointer，按照当前 warp ID/age 排序 |
| RRR | 丢弃旧 RRR token，设置为 `last_issued + 1` |
| Warp Limiting | 重新解析 prioritization 和 limit，保留 greedy pointer；有限候选集每次排序时重新计算 |
| Two-level Active | 重建 active/pending warp 集合 |

### 为什么 Two-level Active 要重建

Two-level active 持久维护：

```cpp
m_two_level_active_warps
m_two_level_pending_warps
```

这些集合是 policy 专属状态。从其他 policy 进入 two-level active 时，旧集合不存在或者已经过期，因此必须调用 `initialize_two_level_active()`。

### 为什么 Warp Limiting 不重置公共 pointer

当前 remodeled warp-limiting 没有持久保存 limited subset。`order_warp_limiting()` 每次调度时都从当前 warp 集合重新生成候选顺序。因此 transition 只需要更新 limit 参数，最近 issue 的 warp 可以作为新的 greedy warp 继续使用。

### RRR token

RRR 有独立的 `m_rrr_current_turn_warp`。从其他 policy 进入 RRR 时，不恢复旧的 RRR token，因为 warps 在 RRR 未启用期间可能已经发生变化。新的 token 从公共 last-issued history 推导：

```cpp
m_rrr_current_turn_warp =
    (m_greedy_pointer_issue + 1) % m_warps_of_subcore.size();
```

## Timing 结果的解释

相同 policy 不保证某个后续 kernel 在不同 policy 序列中具有完全相同的 cycles。例如：

```text
LRR -> GTO
GTO -> GTO
```

虽然第二个 kernel 都使用 GTO，但第一个 kernel 的调度可能改变 memory timing、DRAM state、arbitration history 和其他跨 kernel 微架构状态。因此必须保持相同的是程序结果和 dynamic instruction trace，而不是 cycles。

合理的验收条件包括：

- `GTO -> GTO` 与 whole-workload GTO 基线一致；
- `LRR -> LRR` 与 whole-workload LRR 基线一致；
- 切换日志中的 dynamic launch ID 和 policy 正确；
- 指令数和 CTA 数保持一致；
- hybrid policy 序列允许出现 timing 差异。

验收脚本位于：

```text
test/run_per_kernel_policy_simulation_test.sh
```

## 当前限制

1. 只支持 remodeled SM：`-is_SM_remodeling_enabled 1`。
2. 只支持 non-concurrent kernel：`-gpgpu_concurrent_kernel_sm 0`。
3. Policy 在 dynamic kernel launch 前全 GPU 同步切换。
4. 尚未实现 kernel-relative cycle phase，例如前 2000 cycles 使用 GTO、之后切换为 LRR。
5. 尚未定义 concurrent kernels 在同一 SM 上使用不同 policy 时的跨 kernel warp 仲裁规则。

## 后续扩展方向

未来实现同一 kernel 内按 cycle 切换时，应继续沿用当前状态分层：

- 保留公共 last-issued/last-fetched 状态；
- 重建目标 policy 专属状态；
- 在 `gpgpu_sim::cycle()` 进入并行 cluster/core cycle 之前统一切换所有 SM；
- 明确定义 kernel-relative cycle 0 和 phase 边界。
