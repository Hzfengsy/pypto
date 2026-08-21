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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
using transform_utils::FlattenToStmts;

namespace {

class WindowWriteLocalizer : public IRMutator {
 public:
  WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                       const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                       WindowRewriteContext& rewrite_context)
      : out_info_by_var_(out_info_by_var), new_out_vars_(new_out_vars), rewrite_context_(rewrite_context) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto remap_it = result_var_remap_.find(op.get());
    if (remap_it != result_var_remap_.end()) return remap_it->second;
    auto out_it = new_out_vars_.find(op.get());
    if (out_it != new_out_vars_.end()) return out_it->second;
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto out_it = new_out_vars_.find(op.get());
    if (out_it != new_out_vars_.end()) return out_it->second;
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto visited_value = VisitExpr(op->value_);
    auto assign = MutableCopy(op);
    assign->value_ = visited_value;
    auto call = As<Call>(assign->value_);
    if (!call) return assign;

    ExprPtr rewritten_target_expr;
    const Var* target_var = nullptr;
    MakeTuplePtr offsets;
    size_t offset_arg_index = SIZE_MAX;
    size_t target_arg_index = SIZE_MAX;

    if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[2];
      auto out_var = AsVarLike(rewritten_target_expr);
      if (!out_var) return assign;
      target_var = out_var.get();
      offsets = As<MakeTuple>(call->args_[1]);
      offset_arg_index = 1;
      target_arg_index = 2;
    } else if (IsOp(call, "tensor.assemble") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[0];
      auto parent_var = AsVarLike(rewritten_target_expr);
      if (!parent_var) return assign;
      target_var = parent_var.get();
      offsets = As<MakeTuple>(call->args_[2]);
      offset_arg_index = 2;
      target_arg_index = 0;
    } else if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[0];
      auto parent_var = AsVarLike(rewritten_target_expr);
      if (!parent_var) return assign;
      target_var = parent_var.get();
      offsets = As<MakeTuple>(call->args_[1]);
      offset_arg_index = 1;
      target_arg_index = 0;
    } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
      rewritten_target_expr = call->args_[0];
      auto parent_var = AsVarLike(rewritten_target_expr);
      if (!parent_var) return assign;
      target_var = parent_var.get();
      offsets = As<MakeTuple>(call->args_[2]);
      offset_arg_index = 2;
      target_arg_index = 0;
    } else {
      return assign;
    }

    const OutputRewriteInfo* info = nullptr;
    auto info_it = out_info_by_var_.find(target_var);
    if (info_it != out_info_by_var_.end()) {
      info = &info_it->second;
    } else {
      auto result_info_it = result_var_output_info_.find(target_var);
      if (result_info_it != result_var_output_info_.end()) info = result_info_it->second;
    }
    if (!info) return assign;
    if (!offsets) return assign;
    if (offsets->elements_.size() != info->callsite_offsets.size()) return assign;

    arith::Analyzer analyzer;
    std::vector<ExprPtr> local_offsets;
    local_offsets.reserve(offsets->elements_.size());
    std::vector<StmtPtr> prelude_stmts;
    for (size_t i = 0; i < offsets->elements_.size(); ++i) {
      auto local_offset = analyzer.Simplify(
          MakeSub(offsets->elements_[i], info->callsite_offsets[i], offsets->elements_[i]->span_));
      local_offsets.push_back(
          FlattenGeneratedScalarExpr(local_offset, assign->var_->name_hint_, assign->span_, &prelude_stmts));
    }
    auto new_offset_tuple = std::make_shared<MakeTuple>(std::move(local_offsets), offsets->span_);
    std::vector<ExprPtr> new_args = call->args_;
    new_args[offset_arg_index] = new_offset_tuple;
    auto new_out_it = new_out_vars_.find(target_var);
    if (new_out_it != new_out_vars_.end()) new_args[target_arg_index] = new_out_it->second;
    auto new_type = (IsOp(call, "tile.store") || IsOp(call, "tensor.assemble"))
                        ? new_args[target_arg_index]->GetType()
                        : call->GetType();
    auto new_call =
        std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_type, call->span_);

    auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
    result_var_remap_[assign->var_.get()] = new_result_var;
    result_var_output_info_[new_result_var.get()] = info;
    assign->var_ = new_result_var;
    assign->value_ = new_call;
    if (!prelude_stmts.empty()) {
      prelude_stmts.push_back(assign);
      return SeqStmts::Flatten(std::move(prelude_stmts), assign->span_);
    }
    return assign;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_loop = MutableCopy(op);
    new_loop->start_ = VisitExpr(op->start_);
    new_loop->stop_ = VisitExpr(op->stop_);
    new_loop->step_ = VisitExpr(op->step_);

    std::unordered_map<const Var*, OutputRewriteInfo> nested_out_info(out_info_by_var_.begin(),
                                                                      out_info_by_var_.end());
    std::unordered_map<const Var*, ExprPtr> nested_new_out_vars(new_out_vars_.begin(), new_out_vars_.end());
    bool changed = false;

    for (size_t i = 0; i < new_loop->iter_args_.size() && i < new_loop->return_vars_.size(); ++i) {
      auto old_iter_arg = new_loop->iter_args_[i];
      auto old_return_var = new_loop->return_vars_[i];
      auto init_expr = VisitExpr(old_iter_arg->initValue_);
      auto init_var = AsVarLike(init_expr);
      if (!init_var) {
        if (init_expr.get() != old_iter_arg->initValue_.get()) {
          auto new_iter_arg = std::make_shared<IterArg>(old_iter_arg->name_hint_, old_iter_arg->GetType(),
                                                        init_expr, old_iter_arg->span_);
          new_loop->iter_args_[i] = new_iter_arg;
          changed = true;
        }
        continue;
      }

      const OutputRewriteInfo* info = nullptr;
      auto direct_info_it = out_info_by_var_.find(init_var.get());
      if (direct_info_it != out_info_by_var_.end()) {
        info = &direct_info_it->second;
      } else {
        auto result_info_it = result_var_output_info_.find(init_var.get());
        if (result_info_it != result_var_output_info_.end()) info = result_info_it->second;
      }

      if (!info) {
        if (init_expr.get() != old_iter_arg->initValue_.get()) {
          auto new_iter_arg = std::make_shared<IterArg>(old_iter_arg->name_hint_, old_iter_arg->GetType(),
                                                        init_expr, old_iter_arg->span_);
          new_loop->iter_args_[i] = new_iter_arg;
          changed = true;
        }
        continue;
      }

      auto narrowed_type = init_expr->GetType();
      auto new_iter_arg =
          std::make_shared<IterArg>(old_iter_arg->name_hint_, narrowed_type, init_expr, old_iter_arg->span_);
      auto new_return_var =
          std::make_shared<Var>(old_return_var->name_hint_, narrowed_type, old_return_var->span_);

      nested_out_info[old_iter_arg.get()] = *info;
      nested_out_info[new_iter_arg.get()] = *info;
      nested_new_out_vars[old_iter_arg.get()] = new_iter_arg;
      nested_new_out_vars[new_iter_arg.get()] = new_iter_arg;
      result_var_remap_[old_return_var.get()] = new_return_var;
      result_var_output_info_[new_return_var.get()] = info;

      new_loop->iter_args_[i] = new_iter_arg;
      new_loop->return_vars_[i] = new_return_var;
      changed = true;
    }

    if (!changed) return IRMutator::VisitStmt_(op);

    WindowWriteLocalizer nested_localizer(nested_out_info, nested_new_out_vars, result_var_remap_,
                                          result_var_output_info_, rewrite_context_);
    new_loop->body_ = nested_localizer.VisitStmt(new_loop->body_);
    return new_loop;
  }

 private:
  ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                     std::vector<StmtPtr>* stmts) {
    return FlattenGeneratedScalarExprWithLocalTemps(expr, name_prefix, span, stmts, rewrite_context_);
  }

  WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                       const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                       std::unordered_map<const Var*, VarPtr> result_var_remap,
                       std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_output_info,
                       WindowRewriteContext& rewrite_context)
      : out_info_by_var_(out_info_by_var),
        new_out_vars_(new_out_vars),
        result_var_remap_(std::move(result_var_remap)),
        result_var_output_info_(std::move(result_var_output_info)),
        rewrite_context_(rewrite_context) {}

  const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var_;
  const std::unordered_map<const Var*, ExprPtr>& new_out_vars_;
  std::unordered_map<const Var*, VarPtr> result_var_remap_;
  std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_output_info_;
  WindowRewriteContext& rewrite_context_;
};

class WindowReadLocalizer : public IRMutator {
 public:
  WindowReadLocalizer(const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var,
                      WindowRewriteContext& rewrite_context)
      : in_info_by_var_(in_info_by_var), rewrite_context_(rewrite_context) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto visited_value = VisitExpr(op->value_);
    auto assign = MutableCopy(op);
    assign->value_ = visited_value;

    auto call = As<Call>(assign->value_);
    if (!call || call->args_.empty()) return assign;

    size_t offset_arg_index = SIZE_MAX;
    if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
      offset_arg_index = 1;
    } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
      // Keep the localizer aligned with AnalyzeInputWindows(): only window
      // reads that are already proven as a fixed tile.load/tensor.slice are
      // rewritten, and tensor.slice only localizes the matched offset.
      offset_arg_index = 2;
    } else {
      return assign;
    }

    auto parent = AsVarLike(call->args_[0]);
    auto info_it = parent ? in_info_by_var_.find(parent.get()) : in_info_by_var_.end();
    if (info_it == in_info_by_var_.end()) return assign;

    auto old_offsets = As<MakeTuple>(call->args_[offset_arg_index]);
    if (!old_offsets) return assign;
    if (old_offsets->elements_.size() != info_it->second.callsite_offsets.size()) return assign;

    arith::Analyzer analyzer;
    std::vector<ExprPtr> local_offsets;
    local_offsets.reserve(old_offsets->elements_.size());
    std::vector<StmtPtr> prelude_stmts;
    for (size_t i = 0; i < old_offsets->elements_.size(); ++i) {
      ExprPtr base_offset = info_it->second.callsite_offsets[i];
      auto local_offset = analyzer.Simplify(
          MakeSub(old_offsets->elements_[i], base_offset, old_offsets->elements_[i]->span_));
      local_offsets.push_back(
          FlattenGeneratedScalarExpr(local_offset, assign->var_->name_hint_, assign->span_, &prelude_stmts));
    }

    std::vector<ExprPtr> new_args = call->args_;
    new_args[offset_arg_index] = std::make_shared<MakeTuple>(std::move(local_offsets), old_offsets->span_);
    assign->value_ = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, call->GetType(),
                                            call->span_);
    if (!prelude_stmts.empty()) {
      prelude_stmts.push_back(assign);
      return SeqStmts::Flatten(std::move(prelude_stmts), assign->span_);
    }
    return assign;
  }

 private:
  ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                     std::vector<StmtPtr>* stmts) {
    return FlattenGeneratedScalarExprWithLocalTemps(expr, name_prefix, span, stmts, rewrite_context_);
  }

  const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var_;
  WindowRewriteContext& rewrite_context_;
};

}  // namespace

StmtPtr LocalizeWindowWrites(const StmtPtr& body,
                             const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                             const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                             WindowRewriteContext& rewrite_context) {
  WindowWriteLocalizer localizer(out_info_by_var, new_out_vars, rewrite_context);
  return localizer.VisitStmt(body);
}

StmtPtr LocalizeWindowReads(const StmtPtr& body,
                            const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var,
                            WindowRewriteContext& rewrite_context) {
  WindowReadLocalizer localizer(in_info_by_var, rewrite_context);
  return localizer.VisitStmt(body);
}

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto
