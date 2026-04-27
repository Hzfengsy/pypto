# Diagnostics: Warnings and Performance Hints

Unified advisory diagnostic channel for the pass pipeline. Surfaces warnings (likely user mistakes) and performance hints (advisory tuning suggestions) through a single registry, instrument, and output path.

## Overview

| Component | Purpose |
| --------- | ------- |
| **`Diagnostic` struct** | Carries severity (`Error` / `Warning` / `PerfHint`), `rule_name`, `error_code`, `hint_code`, message, span. |
| **`DiagnosticCheck` enum** | Identifies a specific check (e.g. `UnusedVariable`, `TileInnermostDimGranularity`). |
| **`DiagnosticCheckRegistry`** | Maps checks to verifier factories; each registration declares severity, phase, and hint code. |
| **`DiagnosticInstrument`** | `PassInstrument` that runs registered checks and dispatches output. |
| **`DiagnosticPhase`** | When a check fires: `PrePipeline`, `PostPass`, or `PostPipeline`. Per-check, declared at registration. |

Severity is independent of phase. A `Warning` may run at `PrePipeline`; a `PerfHint` may run at `PostPipeline` — declared per check.

## Severities

| Severity | When | Output | Suppression |
| -------- | ---- | ------ | ----------- |
| `Error` | IR is invalid | `VerificationError` thrown | Cannot be suppressed |
| `Warning` | Likely mistake or pass bug | `LOG_WARN` to stderr | `disabled_diagnostics` set |
| `PerfHint` | Advisory tuning suggestion | `LOG_INFO` to stderr (visible at default release log level) **plus** `${ReportInstrument.output_dir}/perf_hints.log` if a report instrument is in the context | `disabled_diagnostics` set |

The release default for `PYPTO_LOG_LEVEL` is `INFO`, so `[perf_hint ...]` lines reach the console out of the box. Override with `PYPTO_LOG_LEVEL=warn` to mute them on stderr (the file output is independent).

## How a check fires

```text
PassPipeline::Run(program)
 ├─ if phase != None: run PrePipeline checks  → EmitDiagnostics
 ├─ for each pass:
 │    ├─ run pass
 │    ├─ if phase != None: run PostPass checks → EmitDiagnostics
 ├─ if phase != None: run PostPipeline checks  → EmitDiagnostics
 └─ ctx.RunAfterPipeline(program)            (instrument hooks)
```

`EmitDiagnostics` routes Warning to `LOG_WARN`, PerfHint to `LOG_INFO`, and additionally appends PerfHint lines to `perf_hints.log` when a `ReportInstrument` is in the active context.

## Registering a new check

```cpp
// 1. Implement a PropertyVerifier subclass (src/ir/verifier/...)
class MyCheck : public PropertyVerifier {
 public:
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diags) override;
  std::string GetName() const override { return "MyCheck"; }
};
PropertyVerifierPtr CreateMyCheckVerifier() { return std::make_shared<MyCheck>(); }

// 2. Add an enum value (include/pypto/ir/verifier/diagnostic_check_registry.h)
enum class DiagnosticCheck : uint32_t { ..., MyCheck = N };

// 3. Register in DiagnosticCheckRegistry::DiagnosticCheckRegistry()
Register(DiagnosticCheck::MyCheck,
         DiagnosticSeverity::PerfHint,
         DiagnosticPhase::PostPipeline,
         "PHnnn",
         CreateMyCheckVerifier);
```

The registry stamps the registered severity and hint code onto every diagnostic the verifier emits.

## Performance hints (issue #1180)

Performance hints are best-effort advisory diagnostics that flag patterns likely to under-utilise the target hardware. They are on by default at `DiagnosticPhase::PostPipeline` and run after the IR is fully lowered.

### Per-backend thresholds

`BackendHandler` exposes:

| Method | Ascend910B | Ascend950 |
| ------ | ---------- | --------- |
| `GetGmAccessGranularityBytes()` | 512 | 128 |
| `GetL2CacheLineBytes()` | 512 | 512 |
| `GetRecommendedInnermostDimBytes()` | 512 | 128 |

Adding a new backend implements these alongside the existing virtuals; perf-hint checks consult them via `PassContext::Current()->GetBackendHandler()`.

### First check: `TileInnermostDimGranularity` (PH001)

Inspects every `tile.load` and `tile.store` op. When the innermost-dimension byte size (`shape[-1] * sizeof(dtype)`) is below `GetRecommendedInnermostDimBytes()`, emits one diagnostic per op pointing at the user-source span.

Example output (from `examples/kernels/08_assemble.py` on Ascend950):

```text
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B; recommended >= 128B for backend a5 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/kernels/08_assemble.py:60:4
```

## User-facing API

```python
# Default: perf hints fire on stderr at end of pipeline.
with passes.PassContext([]):
    pipeline.run(program)

# Persist perf hints to disk too — file lives next to other reports.
with passes.PassContext([passes.ReportInstrument("/tmp/build")]):
    pipeline.run(program)
# → /tmp/build/perf_hints.log

# Suppress a specific hint.
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
with passes.PassContext([], disabled_diagnostics=disabled):
    pipeline.run(program)

# Disable the whole channel.
with passes.PassContext([], diagnostic_phase=passes.DiagnosticPhase.NONE):
    pipeline.run(program)
```

`compile()` and `run()` accept the same parameters via `diagnostic_phase` and `disabled_diagnostics` kwargs.

## Environment variables

| Variable | Effect | Default |
| -------- | ------ | ------- |
| `PYPTO_LOG_LEVEL` | Threshold for stderr output (`debug`/`info`/`warn`/`error`/`fatal`/`event`/`none`) | `info` (release) / `debug` (non-release) |
| `PYPTO_WARNING_LEVEL` | Default `DiagnosticPhase` (`none`/`pre_pipeline`/`post_pass`/`post_pipeline`) | `pre_pipeline` |
| `PYPTO_VERIFY_LEVEL` | Verification level — orthogonal to diagnostics | `basic` |
