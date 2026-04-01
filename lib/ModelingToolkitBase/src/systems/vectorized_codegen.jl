"""
Vectorized code generation for systems with ArrayOp block equations.

When a compiled system has block_eqs metadata (from block tearing), the generated
IIP function can replace scalar assignments for block equations with for-loops,
reducing Julia compilation time from O(N) to O(1) in the number of grid points.
"""

"""Metadata key for storing block equation info on a compiled system."""
struct BlockEquationsKey end

import Moshi.Match: @match
import SymbolicUtils as _SU_VC
using Symbolics: SymbolicT

"""Find the first ArrayOp in an expression, looking inside D() wrappers. (Local copy for MTKBase.)"""
function _find_arrayop_local(expr)
    expr isa SymbolicT || return nothing
    @match expr begin
        _SU_VC.BSImpl.ArrayOp(;) => return expr
        _SU_VC.BSImpl.Term(; f, args) => begin
            for arg in args
                ao = _find_arrayop_local(arg)
                ao !== nothing && return ao
            end
            return nothing
        end
        _ => return nothing
    end
end

"""Extract output_idx symbols, ranges dict, and shape from an ArrayOp. (Local copy for MTKBase.)"""
function _get_arrayop_index_info_local(ao)
    @match ao begin
        _SU_VC.BSImpl.ArrayOp(; output_idx, ranges, shape = sh) => begin
            sym_idxs = [(dim_i, ii) for (dim_i, ii) in enumerate(output_idx) if !(ii isa Int)]
            return [ii for (_, ii) in sym_idxs], ranges, sh
        end
        _ => return SymbolicT[], Dict{SymbolicT, StepRange{Int,Int}}(), UnitRange{Int}[]
    end
end

"""
    _inline_block_observed_into_rhss(rhss, eqs, sys, block_eqs_meta)

Inline observed variable definitions into block representative RHSs so they become
self-contained (no observed variable references). This is critical for loop codegen:
the loop body must not reference N separate observed variable locals.

Only block representative RHSs are modified. Scalar equation RHSs are unchanged
(their observed deps are handled normally by build_function_wrapper).
"""
function _inline_block_observed_into_rhss(rhss, eqs, sys, block_eqs_meta)
    obs = observed(sys)
    isempty(obs) && return rhss

    # Build observed substitution dict
    obs_dict = Dict{SymbolicT, Any}()
    for eq in obs
        lhs_uw = unwrap(eq.lhs)
        obs_dict[lhs_uw] = unwrap(eq.rhs)
    end
    isempty(obs_dict) && return rhss

    new_rhss = collect(rhss)
    for (i, rhs) in enumerate(rhss)
        block = get(block_eqs_meta, i, nothing)
        block === nothing && continue
        isdiffeq(block.representative_eq) || continue
        # Inline all observed into this block representative RHS
        new_rhss[i] = Symbolics.fixpoint_sub(unwrap(rhs), obs_dict; maxiters=100)
    end
    return new_rhss
end

"""
    _build_block_outputidxs(eqs, sys)

Build outputidxs vector mapping each equation to its du[] position.
For block representatives: variable_index of the LHS derivative variable.
For scalar equations: variable_index of the LHS derivative variable.
"""
function _build_block_outputidxs(eqs, sys)
    outputidxs = Int[]
    for eq in eqs
        lhs = unwrap(eq.lhs)
        if isdiffeq(eq)
            # D(u(t)[k]) → variable_index(sys, u(t)[k])
            inner = arguments(lhs)[1]
            pos = variable_index(sys, inner)
            push!(outputidxs, pos)
        elseif _iszero(lhs)
            # Algebraic equation 0 ~ rhs — shouldn't appear in block systems
            # but handle gracefully
            push!(outputidxs, length(outputidxs) + 1)
        else
            # v(t)[k] ~ rhs — variable_index of v(t)[k]
            pos = variable_index(sys, lhs)
            push!(outputidxs, pos)
        end
    end
    return outputidxs
end

"""
    _expand_rhss_for_codegen(rhss, eqs, sys, block_eqs)

Expand M representative RHS expressions to N scalar RHS expressions for code generation.
This is done at codegen time (not during mtkcompile) to keep the structural analysis O(1).
The expansion is pure index-shifting on symbolic expressions — fast O(N) with no
structural analysis or tearing.

Returns `(expanded_rhss, expanded_eqs)`.
"""
function _expand_rhss_for_codegen(rhss, eqs, sys, block_eqs)
    new_rhss = SymbolicT[]
    new_eqs = Equation[]

    for (i, (rhs_val, eq)) in enumerate(zip(rhss, eqs))
        # Check if this equation is a block representative
        block = get(block_eqs, i, nothing)
        if block === nothing || !isa(block, Any) || (block isa Pair ? block.first : 0) < 0
            # Regular scalar equation — keep as-is
            push!(new_rhss, rhs_val)
            push!(new_eqs, eq)
            continue
        end

        # Block equation — expand to N scalar equations
        ao = _find_arrayop_local(unwrap(block.original_eq.lhs))
        if ao === nothing
            ao = _find_arrayop_local(unwrap(block.original_eq.rhs))
        end
        if ao === nothing
            push!(new_rhss, rhs_val)
            push!(new_eqs, eq)
            continue
        end

        output_idx, ranges, sh = _get_arrayop_index_info_local(ao)
        isempty(output_idx) && (push!(new_rhss, rhs_val); push!(new_eqs, eq); continue)

        # Get representative's base indices
        rep_idx_vals = Int[]
        for (dim_i, ii) in enumerate(output_idx)
            if haskey(ranges, ii)
                push!(rep_idx_vals, first(ranges[ii]))
            else
                push!(rep_idx_vals, first(sh[dim_i]))
            end
        end

        iter_ranges = [haskey(ranges, ii) ? ranges[ii] : sh[dim_i]
                       for (dim_i, ii) in enumerate(output_idx)]

        # For each point in the iteration space, shift the RHS and LHS
        lhs_uw = unwrap(eq.lhs)
        rhs_uw = unwrap(rhs_val)
        for idx_tuple in Iterators.product(iter_ranges...)
            shifts = [idx_tuple[d] - rep_idx_vals[d] for d in eachindex(output_idx)]
            if all(iszero, shifts)
                push!(new_rhss, rhs_val)
                push!(new_eqs, eq)
            else
                new_lhs = _shift_array_indices_multidim_sym(lhs_uw, shifts)
                new_rhs = _shift_array_indices_multidim_sym(rhs_uw, shifts)
                push!(new_rhss, new_rhs)
                push!(new_eqs, new_lhs ~ new_rhs)
            end
        end
    end
    return new_rhss, new_eqs
end

"""
Shift array indices in a symbolic expression by per-dimension shifts.
Local copy for MTKBase (mirrors MTK's _shift_array_indices_multidim).
"""
function _shift_array_indices_multidim_sym(expr, shifts::Vector{Int})
    all(iszero, shifts) && return expr
    expr isa SymbolicT || return expr
    if SU.iscall(expr)
        f = operation(expr)
        args = arguments(expr)
        if f === getindex && length(args) >= 2
            n_idx = length(args) - 1
            new_args = Any[args[1]]
            for k in 1:n_idx
                if k <= length(shifts) && shifts[k] != 0
                    push!(new_args, args[k+1] + shifts[k])
                else
                    push!(new_args, args[k+1])
                end
            end
            return new_args[1][new_args[2:end]...]
        elseif f isa Differential
            new_inner = _shift_array_indices_multidim_sym(args[1], shifts)
            return f(new_inner)
        else
            new_args = [_shift_array_indices_multidim_sym(a, shifts) for a in args]
            return SU.maketerm(SymbolicT, f, new_args, SU.metadata(expr))
        end
    end
    return expr
end

# ============================================================================
# IR-Level ForLoop Generation
# ============================================================================

using SymbolicUtils.Code: ForLoop, SetArray, AtIndex, Let, Func, Assignment

"""
    _make_block_forloop_wrap_code(block_eqs_meta, sys, eqs)

Create a `wrap_code` IIP transform that generates `ForLoop` IR objects for block
equations. This operates at the symbolic IR level (before CSE and `toexpr`),
producing O(M) code instead of O(N).

Returns a function `iip_transform(func::Func) -> Func`.
"""
function _make_block_forloop_wrap_code(block_eqs_meta, sys, eqs)
    # Pre-compute block metadata
    block_info_list = []  # (eq_idx, du_range, base_idx, du_offset)
    for (eq_idx, block) in block_eqs_meta
        eq_idx < 0 && continue
        isdiffeq(block.representative_eq) || continue
        rep_lhs = unwrap(block.representative_eq.lhs)
        rep_var = arguments(rep_lhs)[1]  # u(t)[k]
        rep_pos = variable_index(sys, rep_var)
        rep_pos === nothing && continue

        # Get the concrete array index from the representative variable
        base_idx = rep_pos  # default: du position = array index
        if _SU_VC.iscall(rep_var) && operation(rep_var) === getindex
            idx_arg = arguments(rep_var)[2]
            if _SU_VC.isconst(idx_arg)
                base_idx = Int(_SU_VC.unwrap_const(idx_arg))
            end
        end

        du_range = rep_pos:(rep_pos + block.scalar_count - 1)
        du_offset = rep_pos - base_idx
        push!(block_info_list, (eq_idx=eq_idx, du_range=du_range, base_idx=base_idx, du_offset=du_offset))
    end

    function iip_transform(func::Func)
        isempty(block_info_list) && return func

        # Find the Let containing the SetArray
        let_body, set_array = _find_let_with_setarray(func.body)
        (let_body === nothing || set_array === nothing) && return func

        arr_sym = set_array.arr

        # Collect all du positions belonging to blocks
        block_du_positions = Set{Int}()
        for bi in block_info_list
            union!(block_du_positions, bi.du_range)
        end

        # Separate scalar and block AtIndex entries
        scalar_entries = AtIndex[]
        rep_rhs_map = Dict{Int, Any}()  # rep_du_pos => symbolic RHS
        for entry in set_array.elems
            entry isa AtIndex || continue
            if entry.i isa Integer && entry.i in block_du_positions
                rep_rhs_map[Int(entry.i)] = entry.elem
            else
                push!(scalar_entries, entry)
            end
        end

        # Build ForLoops for each block
        forloops = []
        for bi in block_info_list
            rep_pos = first(bi.du_range)
            rep_rhs = get(rep_rhs_map, rep_pos, nothing)
            rep_rhs === nothing && continue

            # Create symbolic loop variable (needs type=Int for integer index)
            loop_var = _SU_VC.Sym{_SU_VC.SymReal}(Symbol("__blk_k_$(bi.eq_idx)");
                type = Int, shape = _SU_VC.ShapeVecT())

            # Parameterize: replace concrete getindex(arr, int) with arr[loop_var + offset]
            param_rhs = _parameterize_symbolic_rhs(rep_rhs, bi.base_idx, loop_var)

            # Compute du index expression
            du_idx = bi.du_offset == 0 ? loop_var : loop_var + bi.du_offset

            # Build ForLoop IR
            inner_set = SetArray(true, arr_sym, [AtIndex(du_idx, param_rhs)], false)
            push!(forloops, ForLoop(loop_var, bi.du_range, inner_set))
        end

        # Rebuild body: scalar SetArray + ForLoops, return arr
        new_scalar_set = SetArray(set_array.inbounds, arr_sym, scalar_entries, false)
        inner_pairs = Union{Assignment, Any}[Assignment(Symbol("##scalar_out##"), new_scalar_set)]
        for (fi, fl) in enumerate(forloops)
            push!(inner_pairs, Assignment(Symbol("##loop_$(fi)##"), fl))
        end
        new_inner = Let(inner_pairs, arr_sym, false)

        # Replace SetArray in the Let chain
        new_body = _replace_setarray_body(func.body, let_body, new_inner)
        return Func(func.args, func.kwargs, new_body, func.pre)
    end

    return iip_transform
end

"""
    _parameterize_symbolic_rhs(expr, base_idx, loop_var)

Replace all concrete-integer `getindex(arr, Const(int))` in a symbolic expression
with parameterized loop-variable expressions. Converts the symbolic RHS to a Julia
Expr via `toexpr`, then parameterizes integer array indices at the Expr level.
Returns a `LiteralExpr` wrapping the parameterized Expr.

This approach avoids symbolic `getindex` type issues by operating at the Expr level
after `toexpr` has resolved all variable name mappings.
"""
function _parameterize_symbolic_rhs(expr, base_idx::Int, loop_var)
    # The expr is symbolic. We need to parameterize it.
    # Strategy: collect all getindex(arr, Const(int)) → term(getindex, arr, loop_var + offset)
    # using term() with explicit type/shape to avoid _getindex dispatch issues.
    sub_dict = Dict{SymbolicT, Any}()
    _collect_symbolic_getindex_subs!(sub_dict, expr, base_idx, loop_var)
    isempty(sub_dict) && return expr
    # Apply substitution with allow-all filter to penetrate Differential
    allow_all = (_) -> true
    sub = _SU_VC.Substituter{false}(sub_dict, allow_all)
    return sub(expr)
end

"""
Collect getindex substitution rules, building replacements with `term()` directly.
"""
function _collect_symbolic_getindex_subs!(sub_dict, expr, base_idx, loop_var)
    expr isa SymbolicT || return
    _SU_VC.iscall(expr) || return
    f = operation(expr)
    args = arguments(expr)
    if f === getindex && length(args) >= 2
        idx = args[2]
        if _SU_VC.isconst(idx)
            val = Int(_SU_VC.unwrap_const(idx))
            offset = val - base_idx
            new_idx = offset == 0 ? loop_var : loop_var + offset
            # Build replacement using term() with explicit type/shape
            expr_type = _SU_VC.symtype(expr)
            replacement = _SU_VC.term(getindex, args[1], new_idx;
                type = expr_type, shape = _SU_VC.ShapeVecT())
            sub_dict[expr] = replacement
            return  # Don't recurse
        end
    end
    for a in args
        _collect_symbolic_getindex_subs!(sub_dict, a, base_idx, loop_var)
    end
end

"""Find the Let containing a SetArray as its body."""
function _find_let_with_setarray(body)
    body isa Let || return (nothing, nothing)
    if body.body isa SetArray
        return (body, body.body)
    elseif body.body isa Let
        return _find_let_with_setarray(body.body)
    end
    return (nothing, nothing)
end

"""Replace the SetArray body in a Let chain with a new body."""
function _replace_setarray_body(outer, target_let, replacement)
    outer isa Let || return replacement
    if outer === target_let
        return Let(outer.pairs, replacement, outer.let_block)
    elseif outer.body isa Let
        new_inner = _replace_setarray_body(outer.body, target_let, replacement)
        return Let(outer.pairs, new_inner, outer.let_block)
    end
    return Let(outer.pairs, replacement, outer.let_block)
end

# Legacy Expr-level function — no longer called, kept for API compat:
function vectorize_iip_expr!(iip_expr, sys, block_eqs)
    return iip_expr  # No-op
end


