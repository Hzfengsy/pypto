# 诊断系统：警告与性能提示

Pass 流水线统一的建议性诊断通道。通过同一套注册表、Instrument 与输出路径，承载警告（疑似用户错误）和性能提示（建议性调优提示）。

## 概述

| 组件 | 作用 |
| ---- | ---- |
| **`Diagnostic` 结构体** | 携带 severity（`Error` / `Warning` / `PerfHint`）、`rule_name`、`error_code`、`hint_code`、消息、span。 |
| **`DiagnosticCheck` 枚举** | 标识具体的检查项（如 `UnusedVariable`、`TileInnermostDimGranularity`）。 |
| **`DiagnosticCheckRegistry`** | 将检查项映射到 verifier 工厂；每次注册都声明 severity、phase 和 hint 码。 |
| **`DiagnosticInstrument`** | 运行已注册检查并分发输出的 `PassInstrument`。 |
| **`DiagnosticPhase`** | 检查触发时机：`PrePipeline`、`PostPass` 或 `PostPipeline`。每个检查独立声明。 |

Severity 与 phase 解耦：`Warning` 可以在 `PrePipeline` 触发，`PerfHint` 可以在 `PostPipeline` 触发——按检查注册，而非按 severity。

## Severity 等级

| Severity | 何时使用 | 输出 | 抑制方式 |
| -------- | -------- | ---- | -------- |
| `Error` | IR 不合法 | 抛出 `VerificationError` | 不可抑制 |
| `Warning` | 疑似用户错误或 pass bug | `LOG_WARN` 至 stderr | `disabled_diagnostics` 集合 |
| `PerfHint` | 建议性调优提示 | `LOG_INFO` 至 stderr（默认 release 日志级别可见），并在上下文中存在 `ReportInstrument` 时附加写入 `${ReportInstrument.output_dir}/perf_hints.log` | `disabled_diagnostics` 集合 |

`PYPTO_LOG_LEVEL` release 默认值为 `INFO`，因此 `[perf_hint ...]` 行开箱可见。设置 `PYPTO_LOG_LEVEL=warn` 可在 stderr 上静音（文件输出独立）。

## 触发流程

```text
PassPipeline::Run(program)
 ├─ phase != None：运行 PrePipeline 检查  → EmitDiagnostics
 ├─ 对每个 pass：
 │    ├─ 执行 pass
 │    ├─ phase != None：运行 PostPass 检查 → EmitDiagnostics
 ├─ phase != None：运行 PostPipeline 检查  → EmitDiagnostics
 └─ ctx.RunAfterPipeline(program)         （instrument 钩子）
```

`EmitDiagnostics` 将 Warning 路由到 `LOG_WARN`，PerfHint 路由到 `LOG_INFO`，并在上下文中存在 `ReportInstrument` 时把 PerfHint 行附加到 `perf_hints.log`。

## 注册新的检查项

```cpp
// 1. 实现 PropertyVerifier 子类（src/ir/verifier/...）
class MyCheck : public PropertyVerifier {
 public:
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diags) override;
  std::string GetName() const override { return "MyCheck"; }
};
PropertyVerifierPtr CreateMyCheckVerifier() { return std::make_shared<MyCheck>(); }

// 2. 添加枚举值（include/pypto/ir/verifier/diagnostic_check_registry.h）
enum class DiagnosticCheck : uint32_t { ..., MyCheck = N };

// 3. 在 DiagnosticCheckRegistry::DiagnosticCheckRegistry() 中注册
Register(DiagnosticCheck::MyCheck,
         DiagnosticSeverity::PerfHint,
         DiagnosticPhase::PostPipeline,
         "PHnnn",
         CreateMyCheckVerifier);
```

注册表会将注册时声明的 severity 和 hint 码盖到 verifier 产生的每个 diagnostic 上。

## 性能提示（issue #1180）

性能提示是对可能未充分利用目标硬件的代码模式的最佳努力建议。默认在 `DiagnosticPhase::PostPipeline` 启用，在 IR 完全 lower 后运行。

### 各 backend 阈值

`BackendHandler` 暴露：

| 方法 | Ascend910B | Ascend950 |
| ---- | ---------- | --------- |
| `GetGmAccessGranularityBytes()` | 512 | 128 |
| `GetL2CacheLineBytes()` | 512 | 512 |
| `GetRecommendedInnermostDimBytes()` | 512 | 128 |

新增 backend 时实现这些虚函数；perf-hint 检查通过 `PassContext::Current()->GetBackendHandler()` 读取。

### 第一项检查：`TileInnermostDimGranularity` (PH001)

检查每个 `tile.load` / `tile.store` 操作。当最内层维度的字节数（`shape[-1] * sizeof(dtype)`）低于 `GetRecommendedInnermostDimBytes()` 时，对每个 op 发一条 diagnostic，指向用户源代码 span。

示例输出（`examples/kernels/08_assemble.py`，Ascend950）：

```text
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B; recommended >= 128B for backend a5 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/kernels/08_assemble.py:60:4
```

## 用户面 API

```python
# 默认：perf hint 在 pipeline 末尾通过 stderr 输出
with passes.PassContext([]):
    pipeline.run(program)

# 同时持久化到磁盘——文件与其他 report 同目录
with passes.PassContext([passes.ReportInstrument("/tmp/build")]):
    pipeline.run(program)
# → /tmp/build/perf_hints.log

# 抑制单个 hint
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
with passes.PassContext([], disabled_diagnostics=disabled):
    pipeline.run(program)

# 关闭整个通道
with passes.PassContext([], diagnostic_phase=passes.DiagnosticPhase.NONE):
    pipeline.run(program)
```

`compile()` 与 `run()` 通过 `diagnostic_phase` / `disabled_diagnostics` 关键字参数接收同样的配置。

## 环境变量

| 变量 | 效果 | 默认 |
| ---- | ---- | ---- |
| `PYPTO_LOG_LEVEL` | stderr 输出阈值（`debug`/`info`/`warn`/`error`/`fatal`/`event`/`none`） | `info`（release）/ `debug`（非 release） |
| `PYPTO_WARNING_LEVEL` | 默认 `DiagnosticPhase`（`none`/`pre_pipeline`/`post_pass`/`post_pipeline`） | `pre_pipeline` |
| `PYPTO_VERIFY_LEVEL` | 校验级别——与诊断系统正交 | `basic` |
