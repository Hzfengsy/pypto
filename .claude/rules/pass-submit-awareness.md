# Call vs Submit Awareness in Passes

> **Status:** `Submit` is a first-class IR kind sibling to `Call`, representing
> a task launch from `pl.submit(...)` inside a `pl.manual_scope`. If the kind
> has not yet landed in the codebase you are reading, treat this rule as the
> required handling once it does — and treat any current attrs-based encoding
> (e.g. `Call::attrs_["manual_dep_edges"]`, return-type augmentation with
> `Scalar[TASK_ID]`) with the same "submit-aware" diligence in the interim.

## Core Principle

**PyPTO has two call-like Expr kinds: `Call` (plain function call) and `Submit`
(task launch). When you create or modify a pass that inspects calls, you MUST
consider both — never assume `Call` covers all call-like nodes.**

This is the same shape of issue as `Var` / `IterArg` covered in
`ir-kind-traits.md`: a single `ObjectKind` dispatch will silently skip submits,
and the bug only surfaces inside `pl.manual_scope` bodies.

## What Distinguishes Submit from Call

| Aspect | `Call` | `Submit` |
| ------ | ------ | -------- |
| Semantics | Synchronous function call | Asynchronous task launch |
| Where it appears | Anywhere | Inside `manual_scope` bodies |
| Return type | Callee's declared return | `Tuple[<callee return>..., Scalar[TASK_ID]]` |
| `args_` vs callee `params_` | Identity, full coverage: `args_.size() == params_.size()` | Identity, *bounded*: `args_.size() <= params_.size()` (see rule 5) |
| Has `deps` | No | First-class `deps_` field — TaskId `Var`s / `Array`s |
| Use-def chain | `args_` only | `args_` **and** `deps_` |
| Python syntax | `out = self.foo(...)` | `out, tid = pl.submit(self.foo, ...)` |

## What Pass Authors Must Do

### 1. When walking calls, walk Submit too

```cpp
// ❌ Silently skips submits — bug inside manual_scope
void MyPass::VisitExpr_(const CallPtr& op) {
  for (const auto& arg : op->args()) { Visit(arg); }
}

// ✅ Handle both — visitor dispatches on kind
void MyPass::VisitExpr_(const CallPtr& op)   { VisitCallLike(op); }
void MyPass::VisitExpr_(const SubmitPtr& op) { VisitCallLike(op); }

void MyPass::VisitCallLike(const ExprPtr& op) {
  auto view = AsCallLike(op);              // unified accessor
  for (const auto& arg : view.args()) Visit(arg);
  for (const auto& dep : view.deps()) Visit(dep);  // empty for Call
}
```

### 2. Treat `deps_` as part of the use-def chain

A `Submit::deps_` entry is a TaskId `Var` or `Array` — a real SSA value, not
metadata. Passes that:

- Collect uses → must include `deps_`
- Substitute variables → must rewrite `deps_` too
- Validate SSA dominance → must check `deps_` are defined before the `Submit`
- Run DCE / liveness → TaskId vars are live through `deps_`

```cpp
// ❌ Missing deps_ — TaskId vars look unused, may get DCE'd
std::vector<VarPtr> CollectUses(const SubmitPtr& op) {
  return CollectVarUses(op->args());
}

// ✅ Submit uses include deps_
std::vector<VarPtr> CollectUses(const SubmitPtr& op) {
  auto uses = CollectVarUses(op->args());
  for (const auto& dep : op->deps()) AppendVarUses(&uses, dep);
  return uses;
}
```

### 3. When transforming, preserve Submit-ness

A pass that rewrites a `Submit` must produce another `Submit`, not a plain
`Call` — even if `deps_` becomes empty after the rewrite. The `TASK_ID` return
shape and the structural property "submit appears inside manual_scope" must be
preserved.

```cpp
// ❌ Loses the Submit kind — return type no longer matches the binding LHS
auto new_call = std::make_shared<Call>(op->callee(), new_args, ...);

// ✅ Preserve kind through rewrite
auto new_submit = std::make_shared<Submit>(
    op->callee(), new_args, op->deps(), op->return_type(), op->span());
```

### 4. When examining return types

`Submit`'s return type is **always** augmented with `Scalar[TASK_ID]` at the
tail. If your pass inspects return types of call-like nodes (for tuple
projection, type inference, etc.), strip / account for the trailing `TASK_ID`
before comparing against the callee's declared signature.

### 5. When matching `args_` against callee `params_`

**Canonical statement: `Submit::args_` in `include/pypto/ir/expr.h`.** Read it
there — this section is the pass-author summary, not a second source of truth.

Mapping is positional identity (`args_[i] ↔ params_[i]`) for both kinds; they
differ only in *coverage*. `Call` is exact (`args_.size() == params_.size()`);
`Submit` is **bounded** (`args_.size() <= params_.size()`). A Submit's args
split into up to three regions, in callee param order (`ctx` = number of
trailing `CommCtxType` params):

| Region | Callee param indices | Carried in `args_`? |
| ------ | -------------------- | ------------------- |
| Caller-supplied | `[0, args_.size() - ctx)` | Yes — **any** direction, In / InOut / **and Out** |
| Runtime-allocated outputs | `[args_.size() - ctx, params_.size() - ctx)` | No — must be declared `Out` |
| CommCtx suffix | `[params_.size() - ctx, params_.size())` | Yes — appended to signature and call site together by `MaterializeDistTensorCtx` |

Two consequences that are easy to get wrong:

- **Caller-allocated `Out` params are NOT excluded from `args_`.** Coverage is
  bounded, not direction-filtered. Orchestration codegen has a live branch for
  exactly this case.
- **`args_` is a plain prefix only while `ctx == 0`.** After
  `MaterializeDistTensorCtx` it is a prefix *plus* a suffix with the
  runtime-allocated region as the gap between them, so a consumer indexing
  `args_` against `params_` must subtract `ctx` first.

The gap region comes from **user source**, not from a pass: writing
`pl.submit(self.kernel, x)` against a kernel that declares a trailing `pl.Out`
param parses fine and carries the gap through the whole pipeline. (No pass
opens it — the two that append a tail `Out` param, `ConvertTensorToTileOps` and
`InjectGMPipeBuffer`, both forward the matching arg.) So bailing on `!=`
mishandles ordinary DSL input, not just a hypothetical future rewrite.

The args-side asymmetry does **not** change the return shape (rule 4): an
appended `Out` param mirrors an existing declared return rather than adding
one, so caller-allocated and runtime-allocated params surface through the
*same* return-tuple elements. Only the aliasing target differs.

**Runtime caveat for orchestration codegen:** `TaskOutputTensors` (returned by
`rt_submit_*_task`) stores only `add_output(TensorCreateInfo)` entries —
runtime-created outputs (see `runtime/.../pto_types.h:72`). `add_inout(Tensor&)`
and caller-allocated `add_output(Tensor&)` slots do **not** appear in
`task_<n>_outs`. So aliasing a tuple-return element must dispatch on the region:

- Caller-supplied (`param_idx < args_.size() - ctx`): alias to the arg's emit
  name (`ext_<arg>` / the user-passed variable).
- Runtime-allocated: alias to `task_<n>_outs.get_ref(param_idx -
  (args_.size() - ctx))` — one synth `add_output` per gap param, appended in
  callee param order, *before* the CommCtx args.

A pass that hard-codes `args_.size() == params_.size()` (typical size guard)
will silently bail on Submit and leave attrs empty. Use a kind-aware bound:

```cpp
// ❌ Wrong for Submit — bails whenever the callee has uncovered tail params
if (call->args_.size() != callee->params_.size()) return call;

// ✅ Identity mapping; relax the size check for Submit, keep it exact for Call.
const bool size_ok = is_submit ? (call->args_.size() <= callee->params_.size())
                               : (call->args_.size() == callee->params_.size());
if (!size_ok) return call;
for (size_t i = 0; i < call->args_.size(); ++i) {
  auto dir = callee->param_directions_[i];   // identity — always safe
  ...
}
// Callee params in the gap region are runtime-allocated outputs — route them
// through add_output / TaskOutputTensors rather than args_-indexed code.
```

This is the args-side dual of the return-type asymmetry in rule 4. If your
pass uses a `SubmitToCallView`–style adapter to share code with the Call
path, the adapter does **not** fix this — `args_` is copied through
unchanged, so the size-vs-params guard will still mis-fire.

### 6. Verifier hooks

If a pass produces a new IR property that involves `Submit` (e.g. "all submit
dependencies are dominated by their definitions"), add a `SubmitVerifier` and
register it in the `PropertyVerifierRegistry` — same pattern as other
property verifiers (see `pass-context-config.md` and `documentation.md`).

## Decision Guide: Unified vs Separate Handlers

```text
Does the pass treat Call and Submit identically?
├─ YES → Use AsCallLike() / a shared VisitCallLike helper — one code path
│
└─ NO  → Submit needs special handling (visits deps_, checks scope, etc.)
    └─ Override VisitExpr_(CallPtr) and VisitExpr_(SubmitPtr) separately
```

When in doubt, separate handlers are safer — the unified `CallLike` view is an
optimization for passes that genuinely do not care about deps or task-launch
semantics.

## Audit Checklist

Before merging a new or updated pass that touches calls, verify:

- [ ] If the pass inspects `Call`, it also inspects `Submit` (or uses `CallLike`)
- [ ] If the pass collects variable uses, `Submit::deps_` is included
- [ ] If the pass rewrites or clones calls, Submit-ness is preserved
- [ ] If the pass examines call return types, the `TASK_ID` suffix on `Submit` is accounted for
- [ ] If the pass matches `args_` against callee `params_`, it allows the Submit coverage bound (size_ok = `args_.size() <= params_.size()`, not `==`) and does not assume Out params are absent from `args_`
- [ ] If the pass produces a structural invariant, the verifier covers `Submit` too
- [ ] Tests cover at least one `pl.manual_scope` / `pl.submit` case if the pass is reachable from there

## See Also

- `ir-kind-traits.md` — broader pattern for sibling-kind dispatch (`Var`/`IterArg`, `Call`/`Submit`)
- `pass-complexity.md` — O(N log N) complexity rule still applies
- `pass-context-config.md` — verifier registration via `PassContext`
- `pass-doc-ordering.md` — doc numbering when adding a new pass
