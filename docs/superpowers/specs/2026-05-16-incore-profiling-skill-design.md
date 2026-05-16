# In-Core Profiling Skill — Design

- **Date:** 2026-05-16
- **Status:** Approved (design)
- **Topic:** Package the `msprof op-simulator` kernel-profiling workflow as a
  portable, repo-committed skill.

## 1. Context

During this session a local tool, `export_all_kernel_insight.py`, was used to
run per-kernel **in-core** profiling of a PyPTO build via Ascend
`msprof op simulator` — cycle-accurate profiling of one kernel running inside a
single AI core. It works (17/17 kernels exported on the local A2 machine), but:

- It lives untracked at the repo root.
- It depends on PTOAS's `generate_testcase.py`, already vendored into
  `ptoas_testcase/` (verbatim copy + 4 templates).
- It has environment-specific defaults that are wrong on most machines:
  `DEFAULT_SOC_VERSION = "dav_2201"`, `DEFAULT_CANN_SET_ENV = "/data/CANN/..."`.
- Running it on the local A2 machine needs three non-obvious flags:
  `--cann-set-env <path>`, `--soc-version Ascend910B1`, `--aicore-arch dav-c220`.

The workflow is valuable and should be (a) committed as a project skill, and
(b) made environment-agnostic so any user with a working Ascend environment can
run it.

It is distinct from the compiler's `report/perf_hints.log` (compile-time hints):
this skill produces cycle-accurate runtime traces.

## 2. Goals / Non-goals

### Goals

- A committed project skill `incore-profiling` documenting the workflow.
- The script + vendored generator reorganized into a self-contained skill folder.
- The script generic across environments via a single `--target {a2a3,a5}` knob,
  CANN auto-discovery, and camodel-SoC auto-selection.

### Non-goals (deferred)

- Auto-detecting `--target` from the build folder (Approach A) — needs the build
  pipeline to dump the backend arch into a dump file. Tracked as future work.
- Unit tests for the CLI tool — verification is a real run (project rule
  discourages throwaway test scripts).
- Cleanup of unrelated session artifacts (`qwen3_decode*.py`, `trace_*.json`,
  the `~/ptoas` clone).

## 3. File layout

**Before** (untracked, repo root):

```text
export_all_kernel_insight.py
ptoas_testcase/
├── generate_testcase.py
└── templates/{main_template.cpp, golden_template.py, compare_template.py, run_sh_template.sh}
```

**After** (committed, self-contained skill folder):

```text
.claude/skills/incore-profiling/
├── SKILL.md            # workflow guide (~150 lines)
├── incore_profile.py   # renamed from export_all_kernel_insight.py
└── vendor/             # renamed from ptoas_testcase/
    ├── generate_testcase.py   # verbatim-vendored; provenance header kept
    └── templates/             # 4 template files, unchanged
```

The files are untracked, so the move is `mv` (not `git mv`), followed by
`git add` of the new tree. `incore_profile.py` already resolves the generator
relative to itself (`SCRIPT_DIR`) and `REPO_ROOT` already `.git`-walks, so the
relocation is safe.

## 4. Script genericity (Approach B)

### 4.1 `--target` knob

A single target abstraction maps to the compile arch and the camodel-SoC family:

```python
# incore_profile.py — module level
TARGET_PROFILES: dict[str, dict[str, object]] = {
    "a2a3": {"aicore_arch": "dav-c220", "soc_keywords": ("910b",)},
    "a5":   {"aicore_arch": "dav-c310", "soc_keywords": ("950", "a5")},
}
```

```python
# argparse — tool_group
tool_group.add_argument(
    "--target", choices=sorted(TARGET_PROFILES), default="a2a3",
    help="NPU target family: sets the compile arch (dav-c220 / dav-c310) and "
         "constrains camodel-SoC auto-selection. Override with --aicore-arch / --soc-version.",
)
```

### 4.2 CANN auto-discovery

Replace the hardcoded `DEFAULT_CANN_SET_ENV` with discovery:

```python
def discover_cann_set_env() -> Path | None:
    """Locate a CANN set_env.sh from the environment / standard install roots."""
    env_val = os.environ.get("CANN_SET_ENV")
    if env_val and Path(env_val).expanduser().is_file():
        return Path(env_val).expanduser()
    candidates: list[Path] = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates += [Path(ascend_home) / "set_env.sh",
                       Path(ascend_home).parent / "set_env.sh"]
    for root in ("/usr/local/Ascend", str(Path.home() / "Ascend"), "/opt/Ascend"):
        candidates += [Path(root) / "ascend-toolkit/set_env.sh",
                       Path(root) / "ascend-toolkit/latest/set_env.sh",
                       Path(root) / "set_env.sh"]
    return next((c for c in candidates if c.is_file()), None)
```

`--cann-set-env` default becomes `None`; `main()` falls back to
`discover_cann_set_env()`, and raises a clear `StepError` ("pass --cann-set-env")
when nothing is found.

### 4.3 Camodel-SoC auto-selection

After sourcing CANN, scan the install for simulator SoCs that ship the camodel
runtime, then pick one within the target family:

```python
def discover_camodel_socs(env: dict[str, str]) -> list[str]:
    ascend_home = Path(env.get("ASCEND_HOME_PATH", ""))
    if not ascend_home.is_dir():
        return []
    socs: set[str] = set()
    for pattern in ("*/simulator/*/lib/libruntime_camodel.so",
                    "simulator/*/lib/libruntime_camodel.so",
                    "tools/simulator/*/lib/libruntime_camodel.so"):
        for lib in ascend_home.glob(pattern):
            socs.add(lib.parent.parent.name)   # <ascend>/.../simulator/<SOC>/lib/...
    return sorted(socs)


def select_soc_version(target: str, available: list[str], explicit: str | None) -> str:
    if explicit:
        return explicit
    keywords = TARGET_PROFILES[target]["soc_keywords"]
    matches = [s for s in available if any(k in s.lower() for k in keywords)]
    if not matches:
        raise StepError(
            f"no camodel SoC found for --target {target}; available: {available}. "
            f"Pass --soc-version explicitly.")
    return matches[0]
```

### 4.4 `main()` wiring and override precedence

Order in `main()`, after `validate_toolchain()` returns the sourced `env`:

1. `args.aicore_arch = args.aicore_arch or TARGET_PROFILES[args.target]["aicore_arch"]`
   (`--aicore-arch` / `AICORE_ARCH` env still win — they are the existing default).
2. `socs = discover_camodel_socs(env)`
3. `args.soc_version = select_soc_version(args.target, socs, args.soc_version)`
   (`--soc-version` default changes from `"dav_2201"` to `None`).

The `--list-funcs` fast path needs no toolchain, so it runs before this block —
unchanged.

### 4.5 Path / docstring updates

- `VENDORED_GENERATE_TESTCASE = SCRIPT_DIR / "vendor" / "generate_testcase.py"`
  (was `"ptoas_testcase"`).
- The comment above it: `See ptoas_testcase/.` → `See vendor/.`
- Module docstring usage examples: `tools/export_all_kernel_insight.py` →
  `.claude/skills/incore-profiling/incore_profile.py`, and show `--target`.
- Remove `DEFAULT_SOC_VERSION` and `DEFAULT_CANN_SET_ENV` constants.

## 5. `SKILL.md` outline

YAML frontmatter (`name: incore-profiling`, `description:` covering "profile
PyPTO kernels in-core with msprof op-simulator"). Body sections:

- **Purpose** — cycle-accurate single-AI-core profiling via `msprof op simulator`;
  contrasted with compile-time `report/perf_hints.log`.
- **Prerequisites** — a built case under `build_output/`; CANN installed;
  `ptoas-bin` (cross-reference `/pto-env-setup`).
- **Quick start** — `python .claude/skills/incore-profiling/incore_profile.py
  --build-dir build_output/<case> --target a2a3`; `--case <model.py>` variant;
  `--list-funcs` preview.
- **Output** — per kernel: `trace.json` + `visualize_data.bin` (open in MindStudio
  Insight), per-core traces, `*_instr_exe.csv`; run-level `manifest_export.csv`
  and `summary.txt`.
- **Troubleshooting table** — `npu/a5` compile error → wrong `--target`;
  `-lruntime_camodel` link error → SoC/CANN mismatch; export "no dump file" →
  handled automatically; CANN not found → `--cann-set-env`.
- **Future** — one line: auto-detect `--target` once the build pipeline dumps the
  backend arch (Approach A).

Length target ≤200 lines (project `documentation-length` rule).

## 6. Implementation order

1. Create `.claude/skills/incore-profiling/`; `mv` + rename the script and the
   `ptoas_testcase/` → `vendor/` directory.
2. Update `incore_profile.py`: `VENDORED_GENERATE_TESTCASE` path, the `vendor/`
   comment, module docstring.
3. Add `TARGET_PROFILES` + `--target`.
4. Add `discover_cann_set_env()`; change `--cann-set-env` default to `None` and
   fall back in `main()`.
5. Add `discover_camodel_socs()` + `select_soc_version()`; change `--soc-version`
   default to `None`; wire arch/SoC resolution into `main()` (§4.4).
6. Write `SKILL.md`.
7. Verify (§7).
8. `git add` the skill folder.

## 7. Verification

No unit tests (CLI tool; project rule discourages throwaway test files).

- `python .claude/skills/incore-profiling/incore_profile.py --help` — parses.
- `... --list-funcs --build-dir build_output/Qwen3Decode_20260515_165750` — lists
  17 funcs with no toolchain.
- `... --build-dir build_output/Qwen3Decode_20260515_165750 --target a2a3` — with
  no `--cann-set-env` / `--soc-version` / `--aicore-arch`, expect CANN
  auto-discovered, SoC auto-selected, and **17/17 exported**.
- `ruff check incore_profile.py` — no new diagnostics from this change.

## 8. Future work — Approach A

Once the build pipeline records the backend arch into a dump file (e.g. under
`passes_dump/` or `kernel_config.py`), `incore_profile.py` can read it and
default `--target` automatically, making `--build-dir` the only required
argument. The `--target` flag remains as an override. Out of scope for this
change.
