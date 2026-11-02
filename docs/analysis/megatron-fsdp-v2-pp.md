# Megatron FSDP v2 与 PP 混合并行源码分析

本文分析 Megatron 自研 FSDP v2，即 `--use-megatron-fsdp --megatron-fsdp-version 2`。
分析基于源码快照 [`bfab98f4a3ca7bfab11f7fee7672e95be54a70b6`](https://github.com/NVIDIA/Megatron-LM/commit/bfab98f4a3ca7bfab11f7fee7672e95be54a70b6)，整理日期为 2026-11-02。
文中的源码链接指向仓库内文件，行号以该快照为准；后续版本需要重新核对。

核心方案是：**PP 负责模型层切分和 microbatch 调度；FSDP v2 负责每个 PP stage 内的参数分片、按需 all-gather 和梯度 reduce-scatter。两者通过模型 hook 与梯度同步接口衔接。**

以下以 Dense、BF16、TP=CP=EP=1、单层 DP 分片以及 `optim_grads_params` 完整分片策略为例。
本文区分源码已实现的路径、独立 CPU 复现结果和尚未执行的多卡验证。

## 1. 模型与设备如何划分

假设模型有 24 层，使用 8 张卡，配置 PP=2、DP=4。在默认 rank 排列下：

| DP/FSDP rank | PP stage 0：第 0～11 层 | PP stage 1：第 12～23 层 |
|---|---|---|
| 0 | GPU 0 | GPU 4 |
| 1 | GPU 1 | GPU 5 |
| 2 | GPU 2 | GPU 6 |
| 3 | GPU 3 | GPU 7 |

- 每一列是一个 FSDP group，负责当前 stage 的参数 all-gather 和梯度 reduce-scatter。
- 每一行是一条 PP 链路，负责同一份数据在相邻 stage 间的激活与激活梯度传输。
- 每张卡常驻当前 stage 的权重、主权重、梯度及优化器状态分片；计算某个 FSDP unit 时临时恢复该 unit 的完整计算权重。

例如 GPU 0 计算 stage 0 的某层前，从 GPU 0～3 gather 该层权重；stage 0 完成后，GPU 0 将输出激活发送给 GPU 4。

PP 决定本地需要构造哪些层，FSDP v2 通过当前 stage 的 `dp_cp` 进程组建立 mesh：

```python
dp_mesh = DeviceMesh.from_group(
    pg_collection.dp_cp,
    device_type=device_type,
    mesh_dim_names=("dp",),
)
```

在此示例中，mesh 覆盖当前 PP stage 的 DP ranks；PP 维度不参与该 unit 的 FSDP gather。
启用 CP 时，`dp_cp` 将 CP ranks 一并纳入分片域，不能再将其简单理解为纯 DP group。

源码入口：[PP 层数切分](../../megatron/core/transformer/transformer_block.py#L157)、[默认 rank 排列](../../megatron/core/parallel_state.py#L645)、[v2 mesh 构造](../../megatron/core/distributed/fsdp/mcore_adapter_v2.py#L227)。

## 2. 初始化顺序：先切 PP，再按模块分片

```text
get_model()
  ├─ 根据 pp_rank / vp_stage 构造本地 model chunk
  └─ wrap_model_chunks_with_ddp()
       └─ FullyShardedDataParallel() 工厂函数
            └─ FullyShardedDataParallelV2
                 ├─ 构造当前 stage 的 DP mesh
                 ├─ FsdpUnitCollector.collect()
                 └─ 对选中的子模块逐个 fully_shard()
```

`get_model()` 根据 PP rank 设置 `pre_process/post_process`；启用 VPP 时，先创建多个本地 model chunk。
之后，各 chunk 交由 FSDP wrapper 包装，工厂函数根据 `megatron_fsdp_version` 选择 v1 或 v2。

Dense Transformer 默认以 `TransformerLayer` 为 FSDP unit。
collector 按子模块在前、根模块在后的顺序处理，根 unit 管理未被子 unit 接管的剩余参数，例如 embedding 或输出层参数。
这保证嵌套 unit 的参数具有明确归属，父 unit 不会重复接管子 unit 的参数。

`fully_shard()` 将 `FsdpModule` mixin 附加到原模块，并安装 forward/backward hooks；父模块继续引用原来的子模块对象。
不同于只包装整个 stage，按层划分 unit 允许逐层 gather、释放和预取，限制临时完整权重的占用。

源码入口：[模型构造](../../megatron/training/training.py#L2584)、[版本分派](../../megatron/core/distributed/fsdp/mcore_fsdp_adapter.py#L45)、[默认 unit 类型](../../megatron/core/distributed/fsdp/mcore_adapter_v2.py#L186)、[unit 收集器](../../megatron/core/distributed/fsdp/fsdp_unit_collector.py#L31)、[fully_shard](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/fully_shard.py#L91)、[参数归属](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L528)。

## 3. PP 调度与 FSDP 参数生命周期

PP 仍使用 Megatron 的流水调度：普通 PP 为 warmup → 1F1B → cooldown；启用 VPP 时选择交错流水调度。
调度器决定何时执行某个 chunk 的 forward/backward，chunk 内各 FSDP unit 的 hooks 决定何时恢复与释放参数。

同一个 unit 在同一个 microbatch 上的主要生命周期如下：

| 时机 | FSDP v2 动作 |
|---|---|
| `pre_forward()` | gather 当前 unit 的计算权重，等待权重就绪，并按策略预取后续 unit |
| 层 forward | 使用完整计算权重执行 |
| `post_forward()` | 切回分片参数，释放临时完整权重存储 |
| `pre_backward()` | 再次 gather backward 所需的计算权重 |
| 层 backward 完成 | 得到当前 microbatch 的完整局部梯度 |
| `post_backward()` | 释放完整权重，reduce-scatter 梯度并累积到分片 buffer |

核心实现：

```python
def post_backward(self) -> None:
    self.reshard()
    self.context.validate_grad_sync()
    self._reduce_gradient_groups()
    self.phase = Phase.RESTING
    torch.cuda.nvtx.range_pop()
```

对于有可训练参数的 unit，`post_backward` 由参数梯度完成计数触发：所有归属参数的梯度都完成累积后，才启动该 unit 的归约。
延迟权重梯度计算使用对应的完成回调，避免仅凭 module backward 返回就过早释放或归约。

warmup 期间可以保留多个 microbatch 的激活，但每次 forward 后仍可释放完整权重。
后续 backward 再次 gather 权重，所有 microbatch 的反向完成后才执行 optimizer step，因此同一训练步内使用同一版本的权重。

源码入口：[调度选择](../../megatron/core/pipeline_parallel/schedules.py#L161)、[1F1B](../../megatron/core/pipeline_parallel/schedules.py#L2408)、[hook 注册与梯度完成计数](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L156)、[参数生命周期](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L371)、[训练步中的 forward/backward](../../megatron/training/training.py#L3450)、[optimizer step](../../megatron/training/training.py#L3538)。

## 4. 多 microbatch 的梯度累积与同步契约

在普通完整分片模式下，每个 microbatch 都会发起梯度 reduce-scatter，并将结果写入或累加到常驻的 `main_grad`。
完整局部梯度被打包进临时 `Partial("avg")` buffer，归约后绑定到优化器可见的分片参数 `.grad`。

以下是累积逻辑的简化，省略了直接写入 buffer、dtype 转换和缩放分支：

```python
reduced_grad = partial_grad.redistribute(self.main_grad.placements)
if has_sharded_grads:
    self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
else:
    self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)
```

因此，PP 调度器与 v2 的同步接口具有以下含义：

| 接口 | v2 中的实际行为 |
|---|---|
| `no_sync()` | 通过 `microbatch(context, is_last=False)` 标记非最终 microbatch，不停止普通完整分片模式的逐 microbatch reduce-scatter |
| `start_param_sync()` | 空操作；参数 gather 由模块 hooks 发起 |
| `start_grad_sync()` | 空操作；梯度归约已经在 backward 中发起 |
| `finish_grad_sync()` | 在当前计算流上等待已经发起的梯度通信 |

训练入口对 v2 安装 `no_sync_func`，不依赖 DDP 的 `overlap_grad_reduce` 开关。
如果扩展为 HSDP/HFSDP，最终 microbatch 标记还控制 outer-DP 归约；该扩展需要单独核对与 PP/VPP 调度的组合，不能由单层 DP 分片的路径推定正确性。

源码入口：[局部梯度打包](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L619)、[分片梯度累积](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L656)、[同步接口](../../megatron/core/distributed/fsdp/mcore_adapter_v2.py#L465)、[训练入口接入](../../megatron/training/training.py#L4808)。

## 5. 通信重叠与流水结束后的收尾

`FsdpContext` 管理通信流、预取顺序与 microbatch 状态。
默认情况下，all-gather 和梯度 reduction 使用不同 CUDA stream，模块计算运行于当前计算流。
`pre_forward/pre_backward` 先确保当前 unit 的权重就绪，再按预取策略发起后续 unit 的 gather，允许通信与当前 unit 的计算重叠。

Megatron 为 context 设置 `RuntimePolicy(caller_managed_grad_sync=True)`，由 PP 调度结束后的统一流程负责等待：

```text
全部流水 backward 与延迟权重梯度计算完成
  → finalize_model_grads()
  → 各 model_chunk.finish_grad_sync()
  → context.finish_grad_sync()
  → 其余必要的跨组梯度同步
  → 梯度统计、裁剪及 optimizer step
```

`context.finish_grad_sync()` 通过 `current_stream().wait_stream(...)` 建立 CUDA stream 依赖。
它不会阻塞 CPU，也不会替尚未发起的归约补发通信；调用者必须先完成所有 backward 工作。

如果输入 embedding 与输出权重共享，`finalize_model_grads` 还包含跨 PP 阶段的 embedding 梯度同步：恢复完整逻辑梯度，在 embedding group 上 all-reduce，再写回分片。

源码入口：[通信流策略](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/runtime_policy.py#L20)、[参数等待与预取](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L290)、[context 等待](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/context.py#L210)、[梯度收尾](../../megatron/core/distributed/finalize_model_grads.py#L731)、[embedding 梯度处理](../../megatron/core/distributed/finalize_model_grads.py#L287)。

## 6. 优化器与检查点

v2 直接拥有优化器可见的参数和梯度分片，主要持久 buffer 如下：

| Buffer | 用途 |
|---|---|
| `model_weight` | 计算精度的权重分片 |
| `main_weight` | 优化器更新的主权重分片，默认 FP32 |
| `main_grad` | 梯度累积分片 |

当 dtype 与 placement 一致时，部分 buffer 可以共享存储。
计算期间模块切换到完整计算参数；计算结束后，模块恢复为由 `main_weight` 支撑的分片 DTensor 参数。

Megatron 选择 `FullyShardedOptimizer`，保留梯度统计、裁剪和更新流程，由底层 Adam/SGD 更新分片。
更新后调用 `sync_model_weights_from_main_weights()`，刷新计算精度的权重。
v2 不再叠加 `--use-distributed-optimizer`。

检查点要求使用 `--ckpt-format fsdp_dtensor`。
优化器通过 `sharded_state_dict()` 输出以全局参数名组织的状态，并保留 DTensor 分片信息；普通 `state_dict()` 路径明确不支持。

源码入口：[buffer 构造](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L277)、[参数切换](../../megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L434)、[优化器选择](../../megatron/core/optimizer/__init__.py#L700)、[更新后刷新](../../megatron/core/optimizer/fully_sharded_optimizer.py#L415)、[优化器检查点](../../megatron/core/optimizer/fully_sharded_optimizer.py#L264)。

## 7. VPP 的适配与已定位的问题

### 7.1 多 chunk 共享 context

例如 8 层、PP=2、VPP=2，默认层分配为：

```text
PP stage 0：chunk 0 = layers [0,1]，chunk 1 = layers [4,5]
PP stage 1：chunk 0 = layers [2,3]，chunk 1 = layers [6,7]
```

同一 rank 的各 chunk 有自己的 FSDP units，但共享一个 `FsdpContext`，从而共享通信流、预取顺序和 microbatch 状态。
`wrap_model_chunks_with_ddp()` 在整个包装循环外打开一次 `fully_shard_context`，各 adapter 加入已有 context，退出时统一 finalize。
同时，v2 强制所有 chunk 保持 bucketing 开启。

另一个 PP 适配是自动关闭 `deallocate_pipeline_outputs`。
v2 的 full-backward hook 会使 stage 输出成为 autograd view，原来的 pipeline output 伪释放优化不适用于该输出。
常规参数转换路径已经处理这一点；直接手写底层配置时也需要遵守这一约束。

源码入口：[VPP 层分配](../../megatron/core/transformer/transformer_block.py#L162)、[共享 context 包装](../../megatron/training/training.py#L2441)、[adapter 加入 context](../../megatron/core/distributed/fsdp/mcore_adapter_v2.py#L261)、[输出释放适配](../../megatron/training/argument_utils.py#L308)。

### 7.2 优化器检查点的参数名映射问题

当前快照的 `get_global_unique_param_name()` 找到参数后只跳出内层参数循环，没有跳出外层 chunk 循环。
之后执行全局层号转换时，变量 `model` 可能已经指向最后一个 chunk，而目标参数属于前面的 chunk，导致没有将局部层号转换为全局层号。

该函数用于构造 v2 optimizer 的 `_param_to_fqn`，因此 VPP 的共享 context 适配不能单独证明检查点保存与恢复正确。

抽取当前源码函数、用模拟 chunk 与 layer 对象做独立 CPU 复现，结果如下。这里使用 4 层、PP=2、VPP=2，表内层号从 0 开始：

| PP rank | chunk | 实际逻辑层号 | 期望参数名 | 函数返回 |
|---|---|---|---|---|
| 0 | 0 | 0 | `decoder.layers.0.weight` | `decoder.layers.0.weight` |
| 0 | 1 | 2 | `decoder.layers.2.weight` | `decoder.layers.2.weight` |
| 1 | 0 | 1 | `decoder.layers.1.weight` | `decoder.layers.0.weight` |
| 1 | 1 | 3 | `decoder.layers.3.weight` | `decoder.layers.3.weight` |

这验证了函数级参数名映射错误及跨 PP 阶段重名的可能性，没有验证实际多卡 DCP 保存/恢复的具体失败表现。

源码入口：[参数名转换函数](../../megatron/core/transformer/fsdp_dtensor_checkpoint.py#L1061)、[v2 optimizer 参数名映射](../../megatron/core/optimizer/fully_sharded_optimizer.py#L90)。

## 8. 普通 PP 的配置示例与成本

下面的参数片段添加到现有训练命令，模型、数据和启动器参数仍由原训练脚本提供。
使用 8 卡、PP=2、DP=4、micro batch size=1、global batch size=32 时，每步有 `32 / (4 * 1) = 8` 个 microbatch。

```bash
unset CUDA_DEVICE_MAX_CONNECTIONS

# 添加到现有训练命令；保持 VPP 关闭。
--use-megatron-fsdp \
--megatron-fsdp-version 2 \
--data-parallel-sharding-strategy optim_grads_params \
--pipeline-model-parallel-size 2 \
--tensor-model-parallel-size 1 \
--context-parallel-size 1 \
--micro-batch-size 1 \
--global-batch-size 32 \
--bf16 \
--no-gradient-accumulation-fusion \
--ckpt-format fsdp_dtensor
```

当前快照的主要约束：

- TP 必须为 1，配置值与实际 TP group 大小都会被检查。
- 关闭 gradient accumulation fusion；PP>1 时禁止 per-token loss。
- 不启用 `--use-distributed-optimizer`；检查点格式使用 `fsdp_dtensor`。
- 优化器路径不支持 FP16/loss scaling、optimizer CPU offload、layer-wise distributed optimizer，以及 optimizer-step parameter-gather overlap。
- `CUDA_DEVICE_MAX_CONNECTIONS` 不能为 `1`；示例通过 unset 使用默认值。

均匀切层且忽略 padding、共享参数等因素时，持久模型状态大致按以下规模分摊：

```text
每卡持久模型状态 ≈ 全模型持久状态 / (PP * FSDP)
```

峰值显存还包括临时完整 unit 权重、完整局部梯度、预取 buffer 和流水激活。
固定总卡数时，增大 PP 并不会让上述持久状态的理论分摊比例继续下降，但会改变每卡层数、FSDP group 大小、microbatch 数和流水气泡。

普通 BF16 完整分片中，一个 unit 每个 microbatch 通常经历前向、反向各一次参数 gather，以及一次梯度 reduce-scatter。
实际通信还受 unit 参数大小、预取、重计算和 buffer 复用影响，吞吐收益需要在目标硬件上测量。

源码入口：[并行配置约束](../../megatron/core/distributed/fsdp/mcore_adapter_v2.py#L361)、[优化器约束](../../megatron/core/optimizer/fully_sharded_optimizer.py#L107)、[训练参数约束](../../megatron/training/arguments.py#L1009)。

## 9. 验证边界与后续验证项

本次完成了源码调用链核对，以及 VPP 检查点参数名函数的独立 CPU 复现。
没有修改训练实现，没有执行多卡 GPU 训练、DCP 保存/恢复或性能测试。

现有 `TestMfsdpV2PipelineSharedContext.test_shared_context_across_virtual_pipeline_chunks` 使用 PP=1，构造两个模拟 chunk，仅检查共享 context、wrapper 类型和 context finalize 状态。
它没有执行真实 PP>1 的 forward/backward，也没有验证优化器更新或检查点。

实际落地需要补齐：

1. 普通 PP>1、DP>1、多 microbatch 的训练，并与相同数据和初始化的参考路径比较 loss、梯度和参数更新。
2. 多步训练，确认梯度在 microbatch 间正确累积，在 optimizer step 间正确清零。
3. `fsdp_dtensor` 保存/恢复，与连续训练比较恢复后的状态与更新结果。
4. 在修复并验证上述参数名映射后，覆盖 VPP 的真实交错调度和保存/恢复。
5. 如需共享 embedding、重计算、CP、EP、HSDP/HFSDP 或 CUDA Graph，再分别验证相应组合。
6. 测量峰值显存、step time，以及参数 gather、梯度归约和 PP P2P 的实际重叠情况。

测试入口：[共享 context 测试](../../tests/unit_tests/distributed/mfsdp_v2/test_pipeline_parallel.py#L56)。

## 附录：参数名映射的独立 CPU 复现

从仓库根目录执行以下命令。它只使用 Python 标准库，抽取真实函数体执行，但使用模拟的 layer/chunk 和 expert 名称转换函数；不导入完整 Megatron，不需要 GPU。

```bash
python - <<'PY'
import ast
import re
from pathlib import Path
from types import SimpleNamespace


class TransformerLayer:
    def __init__(self, number, parameter):
        self.layer_number = number
        self.parameter = parameter

    def parameters(self):
        return iter([self.parameter])


class Chunk:
    def __init__(self, number):
        self.parameter = object()
        self.layer = TransformerLayer(number, self.parameter)
        self.config = SimpleNamespace(num_moe_experts=None)

    def named_parameters(self):
        return iter([("decoder.layers.0.weight", self.parameter)])

    def modules(self):
        return iter([self, self.layer])


path = "megatron/core/transformer/fsdp_dtensor_checkpoint.py"
tree = ast.parse(Path(path).read_text())
node = next(
    item for item in tree.body
    if isinstance(item, ast.FunctionDef)
    and item.name == "get_global_unique_param_name"
)
namespace = {
    "re": re,
    "TransformerLayer": TransformerLayer,
    "handle_experts_in_state_dict": lambda state, num_experts: state,
}
exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
get_name = namespace[node.name]

# TransformerLayer.layer_number is one-based in production.
for pp_rank, layer_numbers in [(0, [1, 3]), (1, [2, 4])]:
    chunks = [Chunk(number) for number in layer_numbers]
    for chunk_id, chunk in enumerate(chunks):
        actual = get_name(chunks, chunk.parameter)
        expected = f"decoder.layers.{layer_numbers[chunk_id] - 1}.weight"
        print(
            f"PP={pp_rank}, chunk={chunk_id}: "
            f"actual={actual}; expected={expected}; match={actual == expected}"
        )
PY
```

当前快照中，`PP=1, chunk=0` 一项输出 `match=False`；其余三项为 `match=True`。
