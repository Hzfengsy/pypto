---
name: incore-profiling
description: Profile PyPTO kernels in-core with the Ascend msprof op-simulator — cycle-accurate per-kernel traces. Use when the user wants to profile a built case, inspect kernel timing or instruction streams, or generate MindStudio Insight traces.
---

# In-Core Kernel Profiling (msprof op-simulator)

Run cycle-accurate, single-AI-core profiling of every PTOAS kernel in a PyPTO
build via the Ascend `msprof op simulator`. For each kernel the tool generates a
standalone testcase, builds it, runs it on the op-simulator, and collects the
Insight trace artifacts.

This is **runtime** profiling — distinct from the compiler's
`report/perf_hints.log`, which records compile-time hints. Use that file for
"why is codegen suggesting X"; use this skill for "how does the kernel actually
execute".

## When to use

- The user wants per-kernel timing / instruction-level traces of a built case.
- The user asks for MindStudio Insight traces or `trace.json` artifacts.
- The user wants to compare kernel execution cost across changes.

## Prerequisites

- A built case under `build_output/<case>/` — it must contain a `ptoas/`
  directory of generated `.cpp` kernels.
- A working CANN installation. The tool auto-discovers `set_env.sh`; pass
  `--cann-set-env <path>` if discovery fails.
- `ptoas-bin` installed. If missing, run the `/pto-env-setup` skill.

No PTOAS source checkout is needed — the testcase generator is vendored under
`vendor/`.

## Quick start

```bash
python .claude/skills/incore-profiling/incore_profile.py \
  --build-dir build_output/<case> --target a2a3
```

- `--target a2a3` for Ascend A2/A3 devices, `--target a5` for A5. This sets the
  compile arch (`dav-c220` / `dav-c310`) and constrains camodel-SoC selection.
- `--case <model.py>` instead of `--build-dir` builds the case first, then
  profiles it. Arguments after `--` are forwarded to the case script.
- `--list-funcs --build-dir <dir>` previews the kernels without running anything
  (needs no toolchain).
- `--func <name>` profiles a single kernel; repeatable.

CANN, the camodel SoC, and the compile arch are auto-resolved from `--target`.
Override any of them with `--cann-set-env`, `--soc-version`, `--aicore-arch`.

## Output

Each run writes to `<build-dir>/kernel_insight_all_funcs_<timestamp>/`:

- `manifest_export.csv` and `summary.txt` — index and per-kernel status.
- `funcs/<kernel>/collect/out/OPPROF_*/simulator/`:
  - `trace.json` and `visualize_data.bin` — open these in MindStudio Insight.
  - `core0.*/` — per-core `trace.json`, `*_instr_exe.csv`, `*_code_exe.csv` for
    instruction-level analysis.

A final `EXPORTED N/M` line reports how many kernels succeeded.

## Troubleshooting

- **Build fails inside `pto/npu/a5/*.hpp`** — wrong target; pass `--target a2a3`
  for an A2/A3 device.
- **`ld: cannot find -lruntime_camodel`** — the chosen SoC has no camodel; pass
  `--soc-version` naming a real `simulator/<soc>/` directory in the CANN install.
- **`CANN set_env.sh not found`** — pass `--cann-set-env <path>`, or set
  `ASCEND_HOME_PATH` / `CANN_SET_ENV`.
- **`generate_testcase.py not found`** — the vendored `vendor/` directory is
  missing; restore it, or pass `--ptoas-root <PTOAS source checkout>`.
- **Export step reports "no dump file"** — handled automatically; newer `msprof`
  emits traces during `collect`, and the tool skips the redundant export pass.

## How it works

For each kernel `.cpp` the tool: (1) generates a standalone testcase via the
vendored `generate_testcase.py`, (2) builds a simulator binary with CMake, (3)
runs `golden.py` for reference data, (4) runs `msprof op simulator` to collect
traces, (5) records the artifacts. Steps run per kernel and continue on failure
unless `--no-keep-going` is given.

## Future work

`--target` must currently be passed explicitly. Once the build pipeline records
the backend arch into the build folder, `incore_profile.py` can auto-detect it
and `--build-dir` alone will suffice. `--target` will remain as an override.
