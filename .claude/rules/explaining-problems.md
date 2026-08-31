# Explaining a Bug, Issue, or Failure

## Core Principle

**Explain a defect as a causal chain the reader can follow down the layers —
written in the native syntax or artifact of each layer, and grounded at every
step in output you actually produced. Never in prose where an artifact would do,
and never in an inference presented as an observation.**

This rule governs *explanations* (root-cause writeups, issue analyses, failure
reports). `plans-and-proposals.md` governs proposals; `problem-handling.md`
governs logging an issue to `KNOWN_ISSUES.md`.

## 1. ALWAYS explain in DSL / IR syntax — never describe it in prose

**Every claim about what the program does must be shown as an artifact, in the
native form of the layer it belongs to.** Prose is connective tissue between the
artifacts; it is never the evidence itself. A sentence describing IR is a claim.
The IR is a fact. (The bottom layers are error text and stack frames rather than
code — paste those verbatim too; §5.)

```text
❌ "The constant's buffer allocation is left behind in the cube half, and the
    cube then pushes a value it never defines."

✅ [bad_aic]
   mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)   # UB alloc in the CUBE kernel
   pl.tile.tpush_to_aiv(bias_full__tile__FREE_VAR, ...)   # pushes an UNDEFINED var
```

The explanation switches vocabulary as it descends — user DSL (`pl.aiv_shard`),
printed IR (`pl.tile.tpush_to_aiv`), PTOAS/MLIR (`pto.tpush_to_aiv ...
!pto.tile_buf<loc=acc, ...>`), then the C++ assertion. Use each layer's real
spelling; never paraphrase one layer in another's terms.

**Trim and annotate.** Paste the offending lines, not the dump. Comment them
inline, and state the evidence that makes the point provable:

> `mem_vec_9` appears exactly once in the function — its own definition.

## 2. Calibrate to the reader, and enter at their layer

Ask (or infer) how expert the reader is in *this* subsystem. Default to
non-expert. A non-expert reader enters at the **DSL**, not at a pass name.

Open by establishing the mental model in their vocabulary — what the API means,
what the hardware/runtime actually is — before naming a single pass, class, or
file. Introduce internal names only at the point where they are needed.

```text
❌ "SplitReshapeDirection hard-codes the push lane, so BuildCoreBody's boundary
    arm emits a tpush on AIC."           <- opens inside the compiler
✅ "An Ascend cluster is one cube core plus two vector lanes. `pl.aiv_shard(x)`
    means 'x crosses cube -> vector, and halve it'. That is only meaningful for a
    cube-produced value."                <- opens where the user writes code
```

## 3. Reproduce it — never narrate from source alone

**Build and run a minimal repro before writing a single claim about mechanism.**
Reading the code yields a plausible story; running it yields the true one. In
practice the two differ, and the plausible one names the wrong culprit.

Every artifact you paste must be real output you captured. If you could not run
it, say so explicitly and mark the mechanism as *inferred, unverified*.

## 4. Pair the broken case with a working one

Reduce to the smallest DSL kernel that fails, then put a variant that **works**
beside it, differing by one line. The contrast is the explanation:

```python
acc_h  = pl.aiv_shard(acc)         # cube-produced -> correct
bias_h = pl.aiv_shard(bias_full)   # vector-produced -> THE BUG
```

Where a second, correct code path already solves the same problem elsewhere in
the tree, show its output too. It proves the fix is not speculative — the
codebase already contains the answer.

## 5. Walk one layer per step, following the same value

Structure the body as a descent, each step showing the *same* value getting
worse, in that layer's syntax (§1). For this project that is usually:

| Step | Shows |
| ---- | ----- |
| DSL source | what the author wrote and meant |
| IR after the offending pass | the first corrupt artifact |
| PTOAS / `.pto` / MLIR | what the backend was asked to emit |
| The assertion / error | where it finally dies, with the C++ stack |

Skip a layer only when it adds nothing. Never jump from source to stack trace.

## 6. Separate the trigger from the symptom

An error message names whatever the failing code happened to be holding. That is
usually the symptom. Say which is which — it is most of the value you add.

```text
Reported: "no MLIR mapping for MemRef base 'mem_vec_9'"   <- the symptom (an alloc)
Trigger:  the tpush of an undefined free variable          <- what actually fails
Cause:    direction derived from the op name, not the operand's memory
```

## 7. Name the root cause as a location plus the wrong decision

Not an area — a function, a file:line, and the specific decision it gets wrong.
Quote the few lines and say what they should have consulted instead.

```text
❌ "The bug is in the pass's statement partitioning."
✅ "`<Fn>` (<file>:<line>) picks the transfer direction from the operator NAME,
    while the filter deciding which lane keeps the producer reads the operand's
    MEMORY SPACE. They agree for a cube-produced operand and diverge for a
    vector-produced one, which is the reported input."
```

When two decisions disagree, name both sources of truth and the exact input on
which they diverge. Cite live code only where you verified it at that commit —
a stale claim in an always-on rule misleads every later session.

## 8. Report reachability honestly

A guard can make a trigger unreachable. **Say which it did: mitigated the
observed failure, or removed the root cause.** Name any path that still reaches
the unsafe state, and never let a diagnostic's existence imply a repair.

Report faithfully in both directions: if the issue text you were handed
mis-describes the mechanism, correct it and show the evidence — do not restate a
claim your own repro contradicted.

## 9. Close with what the reader does next

End with actions, not a summary of what you just said:

- **The real fix** — concrete, at the location from §7.
- **Defence in depth** — the check that would have caught it early.
- **Today's workaround**, if any — *including workarounds that look plausible
  but do not work, and why*, so the reader does not burn a day on them.
- Where the repro files live.

## Format

- **Chapters, each with a one-line takeaway** — the reader can stop at any one.
- **Tables for two things differing along one axis** (auto vs explicit, before vs
  after, trigger vs symptom).
- **Present the finished causal chain, not your investigation.** No "first I
  looked at..."; dead ends belong in it only per §9.

## Anti-Patterns

| Don't | Do |
| ----- | -- |
| Describe behaviour in prose | Show the layer's own artifact (§1) |
| Paraphrase one layer in another's terms | Use each layer's real spelling |
| Paste the whole IR / log dump | Paste the offending lines, annotated |
| Lead with the stack trace | Lead with what the author wrote |
| Explain in pass names to a non-expert | Enter at the DSL, introduce names as needed |
| Assert a mechanism you only read | Run it; label inference as inference |
| Stop at the error message's own noun | Trace trigger vs symptom |
| Say "guarded" and leave "fixed?" open | Say mitigated or root-caused, and why |
