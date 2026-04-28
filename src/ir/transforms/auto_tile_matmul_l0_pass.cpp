/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/// AutoTileMatmulL0
/// ----------------
/// For each ``tile.matmul`` whose operands live in ``MemorySpace::Mat`` with
/// static 2D shape, picks an L0 tile shape ``(m, n, k)`` from the active
/// ``BackendHandler``'s L0 capacities (via ``utils::ChooseL0Tile``) and rewrites
/// the call into a peeled first matmul plus a K-loop of ``tile.matmul_acc``
/// over Mat-resident slices.  When the K loop has at least two iterations the
/// loop is marked ``ForKind::Pipeline`` with ``pipeline_stages=2`` so the
/// downstream ``LowerPipelineLoops`` pass produces a 2-deep ping-pong on the
/// auto-inserted Mat→Left/Right moves.
///
/// V1 scope:
///   * Only ``tile.matmul`` (no ``tile.matmul_acc``).  ``matmul_acc`` is
///     handled in a follow-up.
///   * Only K tiling (``m == M`` and ``n == N``).  Cases where the chooser
///     picks ``m < M`` or ``n < N`` would require an additional Mat-resident
///     output buffer + per-iter assemble, also handled in a follow-up.
///   * Requires ``K % k == 0``.  Boundary handling is deferred.
///
/// Cases outside this scope (``matmul_acc``, M/N tiling, K-not-divisible) emit
/// a ``DiagnosticSeverity::PerfHint`` describing why no rewrite happened and
/// leave the matmul untouched so the rest of the pipeline still runs.

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kPassName = "AutoTileMatmulL0";

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(const std::vector<int64_t>& values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (auto v : values) elements.push_back(MakeIndex(v, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

/// True if `tile`'s 2D shape is static and `tile.memory_space_ == Mat`.
bool IsMatResidentStatic2D(const TileTypePtr& tile, int64_t& out_d0, int64_t& out_d1) {
  if (!tile || tile->shape_.size() != 2) return false;
  auto mem = tile->GetMemorySpace();
  if (!mem.has_value() || *mem != MemorySpace::Mat) return false;
  auto a = As<ConstInt>(tile->shape_[0]);
  auto b = As<ConstInt>(tile->shape_[1]);
  if (!a || !b) return false;
  out_d0 = a->value_;
  out_d1 = b->value_;
  return true;
}

/// Element width in bytes for a tile dtype.  Returns 0 for sub-byte types
/// (INT4, FP4 et al.) which the cube path does not support; the caller should
/// emit a PerfHint and skip in that case.
uint32_t DTypeBytes(const DataType& dt) {
  size_t bits = dt.GetBit();
  if (bits == 0 || bits % 8 != 0) return 0;
  return static_cast<uint32_t>(bits / 8);
}

/// Build a ``tile.slice(source, [shape], [offset])`` AssignStmt with a fresh
/// result Var.  Inherits the source's memory_space (Mat).
AssignStmtPtr BuildSliceStmt(const VarPtr& source_var, const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& offset, const std::string& name_hint,
                             const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<ExprPtr> args = {source_var, MakeIndexTuple(shape, span), MakeIndexTuple(offset, span)};
  auto call = reg.Create("tile.slice", args, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build ``tile.matmul(lhs, rhs)`` AssignStmt with a fresh result Var.
AssignStmtPtr BuildMatmulStmt(const VarPtr& lhs, const VarPtr& rhs, const std::string& name_hint,
                              const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto call = reg.Create("tile.matmul", {lhs, rhs}, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build ``tile.matmul_acc(acc, lhs, rhs)`` AssignStmt with a fresh result Var.
AssignStmtPtr BuildMatmulAccStmt(const ExprPtr& acc, const VarPtr& lhs, const VarPtr& rhs,
                                 const std::string& name_hint, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto call = reg.Create("tile.matmul_acc", {acc, lhs, rhs}, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Shape of an ``L0TileResult``-backed K-loop rewrite for a static 2D
/// ``M x K @ K x N`` matmul whose operands live in Mat.  Built up incrementally
/// by ``MaybeRewriteMatmul`` so the actual statement emission is a single
/// straight-line block.
struct KLoopRewrite {
  // Original AssignStmt and operand Vars.
  AssignStmtPtr original;
  VarPtr lhs_mat;  // x_mat: TileType([M, K], FP*, Mat)
  VarPtr rhs_mat;  // y_mat: TileType([K, N], FP*, Mat)
  // Problem dimensions.
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  // Chosen L0 tile shape.
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

/// True when the loop has at least two iterations after peeling, i.e. when the
/// K-tiled matmul should be marked as a pipeline loop so LowerPipelineLoops can
/// body-clone it for ping-pong execution.
bool ShouldMarkPipeline(int64_t K, int64_t k) {
  // After peeling, the loop runs (K/k - 1) iterations.  Pipeline marking only
  // helps when there are at least two iterations to overlap.
  return (K / k) >= 3;
}

/// Build the replacement statements for a single Mat-resident matmul.
///
/// Layout (for K = 4*k, peeled-first form):
///   slice_a_0 = tile.slice(x_mat, [M, k], [0, 0])
///   slice_b_0 = tile.slice(y_mat, [k, N], [0, 0])
///   c_init = tile.matmul(slice_a_0, slice_b_0)
///   for ko in <Pipeline?>(k, K, k) iter_args=[c=c_init] return_vars=[out_acc]:
///     slice_a = tile.slice(x_mat, [M, k], [0, ko])
///     slice_b = tile.slice(y_mat, [k, N], [ko, 0])
///     c_new = tile.matmul_acc(c, slice_a, slice_b)
///     yield c_new
///
/// Reuses the original AssignStmt's Var (``out_acc``) as the loop's
/// ``return_var`` so downstream uses remain valid without substitution.
std::vector<StmtPtr> BuildKLoopRewrite(const KLoopRewrite& r) {
  const Span sp = r.original->span_;
  const std::string base = r.original->var_->name_hint_;

  std::vector<StmtPtr> out;
  out.reserve(4);

  // Peeled first iteration: slice + tile.matmul producing a fresh Acc tile.
  auto slice_a_0 = BuildSliceStmt(r.lhs_mat, {r.M, r.k}, {0, 0}, base + "_l0_a0", sp);
  auto slice_b_0 = BuildSliceStmt(r.rhs_mat, {r.k, r.N}, {0, 0}, base + "_l0_b0", sp);
  auto c_init = BuildMatmulStmt(slice_a_0->var_, slice_b_0->var_, base + "_l0_c_init", sp);
  out.push_back(slice_a_0);
  out.push_back(slice_b_0);
  out.push_back(c_init);

  // K loop runs (K/k - 1) iterations: ko from k to K (step k).  When K == k
  // the caller has already filtered this case out (no rewrite needed), so we
  // expect at least one loop iteration here.
  INTERNAL_CHECK(r.K > r.k) << "Internal error: BuildKLoopRewrite expected K > k";

  auto ko_var = std::make_shared<Var>(base + "_l0_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);

  // Loop body: slice + matmul_acc + yield.
  auto c_iter_arg = std::make_shared<IterArg>(base + "_l0_c", c_init->var_->GetType(), c_init->var_, sp);
  auto slice_a_body = BuildSliceStmt(r.lhs_mat, {r.M, r.k}, /*offset=*/{0, 0}, base + "_l0_a", sp);
  auto slice_b_body = BuildSliceStmt(r.rhs_mat, {r.k, r.N}, /*offset=*/{0, 0}, base + "_l0_b", sp);
  // Patch slice offsets to use the loop var.  We rebuild with the var-based
  // offset since BuildSliceStmt only accepts static offsets.
  {
    auto& reg = OpRegistry::GetInstance();
    auto a_off = std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(0, sp), ko_var}, sp);
    auto a_call = reg.Create("tile.slice", {r.lhs_mat, MakeIndexTuple({r.M, r.k}, sp), a_off}, sp);
    auto a_var = std::make_shared<Var>(base + "_l0_a", a_call->GetType(), sp);
    slice_a_body = std::make_shared<AssignStmt>(a_var, a_call, sp);

    auto b_off = std::make_shared<MakeTuple>(std::vector<ExprPtr>{ko_var, MakeIndex(0, sp)}, sp);
    auto b_call = reg.Create("tile.slice", {r.rhs_mat, MakeIndexTuple({r.k, r.N}, sp), b_off}, sp);
    auto b_var = std::make_shared<Var>(base + "_l0_b", b_call->GetType(), sp);
    slice_b_body = std::make_shared<AssignStmt>(b_var, b_call, sp);
  }
  auto c_new = BuildMatmulAccStmt(c_iter_arg, slice_a_body->var_, slice_b_body->var_, base + "_l0_c_new", sp);
  auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_new->var_}, sp);
  std::vector<StmtPtr> body_stmts = {slice_a_body, slice_b_body, c_new, yield};
  auto body = SeqStmts::Flatten(std::move(body_stmts), sp);

  // Loop kind / attrs.
  ForKind kind = ForKind::Sequential;
  std::vector<std::pair<std::string, std::any>> attrs;
  if (ShouldMarkPipeline(r.K, r.k)) {
    kind = ForKind::Pipeline;
    attrs.emplace_back(kPipelineStagesAttr, /*pipeline_stages=*/2);
  }

  // Return-var reuses the original matmul's output Var so downstream uses
  // remain unchanged.  Iter-arg init is c_init (the peeled matmul result).
  auto for_stmt = std::make_shared<ForStmt>(ko_var, MakeIndex(r.k, sp), MakeIndex(r.K, sp),
                                            MakeIndex(r.k, sp), std::vector<IterArgPtr>{c_iter_arg}, body,
                                            std::vector<VarPtr>{r.original->var_}, sp, kind,
                                            /*chunk_config=*/std::nullopt, std::move(attrs));
  out.push_back(for_stmt);
  return out;
}

/// Decide whether `assign` is a Mat-resident matmul that we know how to tile.
/// On success, returns the rewritten statements.  Otherwise returns nullopt and
/// (when relevant) appends a PerfHint diagnostic.
std::optional<std::vector<StmtPtr>> MaybeRewriteMatmul(const AssignStmtPtr& assign,
                                                       std::vector<Diagnostic>& hints) {
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return std::nullopt;

  const std::string& op_name = call->op_->name_;
  // V1 scope: only plain ``tile.matmul``.  ``tile.matmul_acc`` (an extension
  // that threads a caller-provided accumulator) and ``tile.matmul_bias``
  // (with bias add) are out of scope and should be left untouched.  Skipping
  // them here also keeps this pass idempotent — the pass itself emits
  // ``tile.matmul_acc`` bodies inside the K-loop, and we don't want a
  // re-run to flag those.
  if (op_name != "tile.matmul") return std::nullopt;

  // Operands must be Var with TileType.
  if (call->args_.size() != 2) return std::nullopt;
  auto lhs = As<Var>(call->args_[0]);
  auto rhs = As<Var>(call->args_[1]);
  if (!lhs || !rhs) return std::nullopt;
  auto lhs_tile = As<TileType>(lhs->GetType());
  auto rhs_tile = As<TileType>(rhs->GetType());
  if (!lhs_tile || !rhs_tile) return std::nullopt;

  // Both operands must be Mat-resident with static 2D shape [M,K] / [K,N].
  // If either constraint fails, the matmul is out of scope for this pass
  // (Vec/Acc operands, dynamic shapes) — return silently without a perf hint
  // since it isn't a missed optimization opportunity.
  int64_t M = 0, K_lhs = 0, K_rhs = 0, N = 0;
  if (!IsMatResidentStatic2D(lhs_tile, M, K_lhs) || !IsMatResidentStatic2D(rhs_tile, K_rhs, N)) {
    return std::nullopt;
  }
  if (K_lhs != K_rhs) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-002",
                       "tile.matmul: K dimensions don't match (lhs K=" + std::to_string(K_lhs) +
                           ", rhs K=" + std::to_string(K_rhs) + ") — left untouched",
                       assign->span_);
    return std::nullopt;
  }
  const int64_t K = K_lhs;

  // Read element widths from operand dtypes.
  uint32_t bytes_a = DTypeBytes(lhs_tile->dtype_);
  uint32_t bytes_b = DTypeBytes(rhs_tile->dtype_);
  // Result is FP32 / INT32 per matmul deduction; bytes_c=4 in both cases.
  constexpr uint32_t kBytesC = 4;
  if (bytes_a == 0 || bytes_b == 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-003",
                       "tile.matmul: unsupported operand dtype (zero element width) — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // Read L0 capacities from the active backend handler.
  auto* ctx = PassContext::Current();
  if (!ctx) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-004",
                       "tile.matmul: no PassContext active; cannot read L0 capacities — left untouched",
                       assign->span_);
    return std::nullopt;
  }
  const auto* handler = ctx->GetBackendHandler();
  INTERNAL_CHECK(handler) << "Internal error: PassContext returned a null BackendHandler";

  utils::L0TileConfig cfg;
  cfg.M = static_cast<int>(M);
  cfg.N = static_cast<int>(N);
  cfg.K = static_cast<int>(K);
  cfg.l0a_bytes = handler->GetL0aCapacityBytes();
  cfg.l0b_bytes = handler->GetL0bCapacityBytes();
  cfg.l0c_bytes = handler->GetL0cCapacityBytes();
  cfg.bytes_a = bytes_a;
  cfg.bytes_b = bytes_b;
  cfg.bytes_c = kBytesC;
  cfg.align_m = handler->GetL0FractalAlignment();
  cfg.align_n = handler->GetL0FractalAlignment();
  cfg.align_k = handler->GetL0FractalAlignment();
  cfg.min_m = handler->GetMinL0TileDim();
  cfg.min_n = handler->GetMinL0TileDim();
  cfg.min_k = handler->GetMinL0TileDim();
  // Schedule: lean on LowerPipelineLoops body cloning for L0a/L0b ping-pong;
  // the chooser must reserve half of each so two splits' tiles fit.
  cfg.double_buffer_a = true;
  cfg.double_buffer_b = true;
  cfg.double_buffer_c = false;
  cfg.c_read = false;
  cfg.allow_padding = false;

  utils::L0TileResult res;
  try {
    res = utils::ChooseL0Tile(cfg);
  } catch (const pypto::ValueError& e) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-005",
        std::string("tile.matmul: ChooseL0Tile rejected configuration — left untouched. ") + e.what(),
        assign->span_);
    return std::nullopt;
  }

  // Already L0-sized — nothing to do.
  if (res.m == M && res.n == N && res.k == K) return std::nullopt;

  // V1 scope: K-tiling only.  Defer M/N tiling to a follow-up.
  if (res.m != M || res.n != N) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "tile.matmul: chooser picked m=" + std::to_string(res.m) + ", n=" +
                           std::to_string(res.n) + " (M=" + std::to_string(M) + ", N=" + std::to_string(N) +
                           "); M/N tiling not yet supported in this pass — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // V1 scope: require K % k == 0.  Boundary handling is deferred.
  if (K % res.k != 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-007",
                       "tile.matmul: chooser picked k=" + std::to_string(res.k) + " not dividing K=" +
                           std::to_string(K) + "; K-boundary handling not yet supported — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // K must be split into at least two iterations (K/k >= 2) for any rewrite
  // to be useful.  K == k means already L0-sized; handled above.  K/k == 1
  // (with K != k) is impossible since k divides K and K > k iff K/k >= 2.
  INTERNAL_CHECK(K / res.k >= 2) << "Internal error: K=" << K << " not properly tiled by k=" << res.k;

  if (!res.perf_hint.empty()) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-008",
                       "tile.matmul: ChooseL0Tile fallback. " + res.perf_hint, assign->span_);
  }

  KLoopRewrite r;
  r.original = assign;
  r.lhs_mat = lhs;
  r.rhs_mat = rhs;
  r.M = M;
  r.N = N;
  r.K = K;
  r.m = res.m;
  r.n = res.n;
  r.k = res.k;
  return BuildKLoopRewrite(r);
}

/// Mutator that flattens any tile.matmul rewrite into the enclosing SeqStmts.
class AutoTileMutator : public IRMutator {
 public:
  std::vector<Diagnostic> hints;

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& child : op->stmts_) {
      auto visited = VisitStmt(child);
      if (visited.get() != child.get()) changed = true;
      auto assign = std::dynamic_pointer_cast<const AssignStmt>(visited);
      if (!assign) {
        out.push_back(visited);
        continue;
      }
      auto rewrite = MaybeRewriteMatmul(assign, hints);
      if (!rewrite) {
        out.push_back(visited);
        continue;
      }
      changed = true;
      for (auto& s : *rewrite) out.push_back(std::move(s));
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  /// AssignStmt outside any SeqStmts (e.g. as a function body's single
  /// statement) — wrap the rewrite in a SeqStmts when emission produces more
  /// than one statement.
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto rewritten = IRMutator::VisitStmt_(op);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(rewritten);
    if (!assign) return rewritten;
    auto rewrite = MaybeRewriteMatmul(assign, hints);
    if (!rewrite) return rewritten;
    return SeqStmts::Flatten(std::move(*rewrite), op->span_);
  }
};

FunctionPtr TransformFunction(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  if (!func || !func->body_) return func;
  if (!IsInCoreType(func->func_type_)) return func;
  AutoTileMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  for (auto& d : mutator.hints) hints.push_back(std::move(d));
  if (new_body == func->body_) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass AutoTileMatmulL0() {
  auto run = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool any_change = false;
    std::vector<Diagnostic> hints;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func, hints);
      if (new_func != func) any_change = true;
      new_functions.emplace(gvar, new_func);
    }
    if (!hints.empty()) EmitDiagnostics(hints, kPassName);
    if (!any_change) return program;
    auto new_program = MutableCopy(program);
    new_program->functions_ = std::move(new_functions);
    return new_program;
  };
  return CreateProgramPass(run, kPassName, kAutoTileMatmulL0Properties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
