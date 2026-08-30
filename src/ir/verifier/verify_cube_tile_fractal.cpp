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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// The two operand roles whose outer tile index is M: `Left` is L0A-backed and
/// `Acc` is L0C-backed. The right operand is absent by construction -- its outer
/// index is K, which this M-axis rule does not describe.
enum class CubeRole : std::uint8_t { kLeft, kAcc };

const char* RoleName(CubeRole role) { return role == CubeRole::kLeft ? "left operand" : "accumulator"; }

/// One cube matmul call, decomposed into the operands this verifier checks.
/// Two operands are deliberately absent. The right operand's outer index is K,
/// not M. A bias tile is one row by construction and lives in the architectural
/// Bias table rather than L0A/L0B/L0C, so the M0 rule does not describe it.
struct CubeCall {
  ExprPtr left;
  ExprPtr acc;   ///< null for the fresh (non-accumulating) forms
  bool is_gemv;  ///< GEMV spells its left operand as exactly one physical row
};

/// Decompose a matmul-family call, or return false when `call` is not one.
///
/// `tile.matmul_mx*` is out of scope: the MX block-scale path carries its own
/// paired scale-tile fractal contract (A5 only), which this rule does not
/// describe.
bool MatchCubeCall(const CallPtr& call, CubeCall* out) {
  if (!call || !call->op_) return false;
  const auto& args = call->args_;
  auto at = [&args](size_t i) -> ExprPtr { return i < args.size() ? args[i] : nullptr; };

  // (lhs, rhs)
  if (IsOp(call, "tile.matmul") || IsOp(call, "tile.batch_matmul")) {
    *out = CubeCall{at(0), nullptr, /*is_gemv=*/false};
    return true;
  }
  // (lhs, rhs, bias) -- bias intentionally unchecked, see CubeCall.
  if (IsOp(call, "tile.matmul_bias")) {
    *out = CubeCall{at(0), nullptr, /*is_gemv=*/false};
    return true;
  }
  // (acc, lhs, rhs[, init_cond])
  if (IsOp(call, "tile.matmul_acc") || IsOp(call, "tile.batch_matmul_acc")) {
    *out = CubeCall{at(1), at(0), /*is_gemv=*/false};
    return true;
  }
  if (IsOp(call, "tile.gemv")) {
    *out = CubeCall{at(0), nullptr, /*is_gemv=*/true};
    return true;
  }
  if (IsOp(call, "tile.gemv_bias")) {
    *out = CubeCall{at(0), nullptr, /*is_gemv=*/true};
    return true;
  }
  if (IsOp(call, "tile.gemv_acc")) {
    *out = CubeCall{at(1), at(0), /*is_gemv=*/true};
    return true;
  }
  return false;
}

/// Rejects a cube matmul operand whose *physical* tile extent is not a whole
/// number of NZ fractal boxes.
///
/// pto-isa keeps two independent notions of size for a cube operand, and only
/// one of them is constrained:
///
///  * The **logical** extent is what the instruction computes over.
///    `pto.tmatmul` reads it from the operands' valid region
///    (`docs/isa/tile/matrix-and-matrix-vector.md`: "For M = a.GetValidRow(),
///    K = a.GetValidCol(), and N = b.GetValidCol()"), and A2A3 bounds it at
///    [1, 4095]. `m = 1` is a documented case, not an edge -- `pto.mad` carries
///    a `disable_gemv` clause whose whole purpose is selecting the L0A
///    organization when `%m = 1`.
///
///  * The **physical** extent is how many bytes the tile occupies, and it must
///    be a whole number of fractal boxes. From
///    `docs/isa/cube/nz-fractal-layout.md`: the inner box is `M0 x K0` with
///    `M0 = 16` and `K0 = N0 = C0 / sizeof(dtype)` for `C0 = 32` bytes.
///    Hardware already handles a logical extent narrower than the physical one
///    -- `docs/isa/state-and-types/layout.md`: "Compact mode is automatic in
///    matmul when valid region < physical tile size; the fractal address
///    generator handles it transparently".
///
/// A sub-fractal *physical* tile has no such fallback: it cannot hold even one
/// box. Nothing downstream repairs it -- AutoTileMatmulL0 declines to tile it
/// (issue: the decline was reported as a perf hint), ptoas accepts the shape,
/// and the emitted `TMATMUL` names an L0A/L0C tile the cube cannot address. The
/// failure reached device silently, which is why this is an error at pipeline
/// input on the user's own Span rather than a diagnostic later.
///
/// The fix is always available to the author and costs nothing: declare the
/// tile at fractal size and carry the true extent in `valid_shape`. A load only
/// moves the valid extent, so the padded region costs no DMA and is never read.
///
/// **Scope: the M axis only.** `M0 = 16` is stated flat in the layout table,
/// repeated as the `FRACTAL_NZ_ROW = 16` machine constant, and corroborated in
/// this repository by `BuildGemvResultType`, which gives a one-row GEMV result
/// exactly 16 physical accumulator rows. The K and N granularities are *not*
/// enforced here: the manual gives `K0 = N0 = C0 / E`, which for a 2-byte dtype
/// agrees with the 512-byte `CUBE_BLOCK_SIZE` box but for int8 and f32 does not
/// (int8 would make the L0B box 32x32 = 1024 bytes), so the per-dtype rule for
/// those axes is ambiguous as documented and guessing it would reject kernels
/// the target accepts. Extending the check to K and N is a follow-up, gated on
/// confirming the rule with the ISA owners.
class CubeTileFractalVisitor : public IRVisitor {
 public:
  CubeTileFractalVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name,
                         const backend::BackendHandler* handler)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), handler_(handler) {}

  /// Record every SSA definition before its uses. The IR is in SSA form and
  /// visited in program order, so a matmul's operands are already in the map by
  /// the time the call is checked; an operand that is not (a parameter, an
  /// IterArg, a forward reference) simply resolves to no defining call.
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      if (auto call = As<Call>(op->value_)) defs_[op->var_.get()] = call;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    CheckCubeCall(op);
    IRVisitor::VisitExpr_(op);
  }

  // Every matmul-family node is an operator call and a Submit launches a
  // Function, so a Submit cannot carry one today. Routing it through the Call
  // view anyway keeps this verifier correct if that ever changes.
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) CheckCubeCall(SubmitToCallView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  void CheckCubeCall(const CallPtr& call) {
    CubeCall cube;
    if (!MatchCubeCall(call, &cube)) return;
    const std::string& op_name = call->op_->name_;

    // GEMV's left operand is *required* to be exactly one physical row -- that
    // is the A-vector organization, not a sub-fractal tile -- so it is exempt.
    // Its accumulator is not: BuildGemvResultType gives that 16 physical rows,
    // and the shared path below holds it to exactly that.
    if (!cube.is_gemv) CheckRows(cube.left, CubeRole::kLeft, op_name, call->span_);
    CheckRows(cube.acc, CubeRole::kAcc, op_name, call->span_);
  }

  /// Is `operand` a *freshly allocated* tile, i.e. one whose physical shape the
  /// author wrote down, rather than a window carved out of a larger buffer?
  ///
  /// Only an allocation can be sub-fractal in the sense this rule cares about.
  /// A window such as `tile.slice` addresses bytes inside a buffer that is
  /// already whole boxes, and whether a given window is a legal MAD operand is
  /// the windowing op's contract, not this one's -- `FlattenTileNdTo2D` row-packs
  /// `[B, M, N]` into one `[B*M, N]` accumulator and then hands each batch an
  /// `M`-row `tile.slice` of it, which `CanonicalizeTileSlice` whitelists for a
  /// page at most one L0C block column wide. Checking the window would reject
  /// that shipped lowering.
  ///
  /// This is an allowlist, so it fails open: an operand produced by an op not
  /// named here is skipped rather than rejected. A new op therefore cannot make
  /// this verifier refuse a legal program, only miss an illegal one.
  [[nodiscard]] bool IsFreshAllocation(const ExprPtr& operand) const {
    auto var = AsVarLike(operand);
    if (!var) return false;
    auto it = defs_.find(var.get());
    // No defining call: a parameter or an IterArg. A parameter's physical shape
    // is written in the signature, so it is checked; an IterArg inherits its
    // init's type, which was checked at its own definition.
    if (it == defs_.end()) return As<IterArg>(operand) == nullptr;
    return IsOp(it->second, "tile.load") || IsOp(it->second, "tile.create");
  }

  /// Check the M extent of a 2D, freshly allocated cube operand.
  ///
  /// **Rank > 2 is deliberately skipped.** A batched operand's middle dimension
  /// is the *per-batch* M, which is not the row count of any cube tile:
  /// `FlattenTileNdTo2D` row-packs `[B, M, N]` into a single `[B*M, N]`
  /// accumulator, so the extent the cube actually sees is `B*M` and the legal
  /// constraint is `B*M % 16 == 0` -- weaker than `M % 16 == 0`. Rejecting the
  /// per-batch page would refuse legal programs such as a `[2, 8, N]` batched
  /// accumulator that flattens to a whole `[16, N]` box. The flattened form is
  /// covered instead: this property is re-checked before every later pass that
  /// requires it, by which point every cube operand is 2D.
  void CheckRows(const ExprPtr& operand, CubeRole role, const std::string& op_name, const Span& span) {
    if (!operand) return;
    auto tile_type = As<TileType>(operand->GetType());
    if (!tile_type || tile_type->shape_.size() != 2) return;
    if (!IsFreshAllocation(operand)) return;

    const int granularity = handler_->GetCubeFractalRows();
    if (granularity <= 0) return;

    // A dynamic extent is resolved to a constant well before codegen; there is
    // nothing to divide here and guessing would reject legal programs.
    auto constant = As<ConstInt>(tile_type->shape_[0]);
    if (!constant) return;
    const int64_t value = constant->value_;
    if (value > 0 && value % granularity == 0) return;

    // Round up rather than quote the bare granularity: the actionable number is
    // the physical extent to declare, which for e.g. 24 rows at 16 is 32, not 16.
    const int64_t padded = ((value + granularity - 1) / granularity) * granularity;
    const bool suggest_gemv = role == CubeRole::kLeft && value == 1;
    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "CubeTileFractalValid", /*error_code=*/1,
        op_name + ": the " + RoleName(role) + " tile has " + std::to_string(value) +
            " physical rows, which is not a multiple of the cube NZ fractal granularity " +
            std::to_string(granularity) + " for dtype " + tile_type->dtype_.ToString() + " (function '" +
            func_name_ +
            "'). The cube addresses L0A/L0B/L0C in whole fractal boxes, so a sub-fractal physical tile "
            "cannot be loaded; the logical extent is unconstrained, only the declared tile size is. "
            "Declare " +
            std::to_string(padded) +
            " physical rows and keep the true extent in the tile's valid_shape -- a load moves only "
            "the valid extent, so the padding costs no extra DMA" +
            (suggest_gemv ? ". For a single-row left operand, pl.tile.gemv is the dedicated cube "
                            "instruction and avoids the padding entirely."
                          : "."),
        span);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  const backend::BackendHandler* handler_;
  /// SSA definitions seen so far in this function, keyed by the defined Var.
  std::map<const Var*, CallPtr> defs_;
};

}  // namespace

class CubeTileFractalValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CubeTileFractalValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // Fractal granularity is a backend fact (C0 and the element width). Several
    // codegen tests drive passes with no backend configured; there is nothing to
    // verify against then, and guessing a profile would reject programs the real
    // target accepts. Both lookups below CHECK-fail when unconfigured, so probe
    // first.
    if (!backend::BackendConfig::IsConfigured()) return;
    const auto* ctx = PassContext::Current();
    const backend::BackendHandler* handler =
        ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
    if (handler == nullptr) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      CubeTileFractalVisitor visitor(diagnostics, func->name_, handler);
      visitor.VisitFunction(func);
    }
  }
};

PropertyVerifierPtr CreateCubeTileFractalValidPropertyVerifier() {
  return std::make_shared<CubeTileFractalValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
