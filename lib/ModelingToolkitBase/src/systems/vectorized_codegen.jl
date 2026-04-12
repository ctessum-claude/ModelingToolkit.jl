"""
Vectorized code generation for systems with ArrayOp block equations.

When a compiled system has block_eqs metadata (from block tearing), the generated
IIP function can replace scalar assignments for block equations with for-loops,
reducing Julia compilation time from O(N) to O(1) in the number of grid points.
"""

"""Metadata key for storing block equation info on a compiled system."""
struct BlockEquationsKey end

"""Metadata key for storing eliminated (pre-substituted) block equations on a compiled system."""
struct EliminatedBlockEquationsKey end

import Moshi.Match: @match
using Symbolics: SymbolicT

"""Find the first ArrayOp in an expression, looking inside D() wrappers. (Local copy for MTKBase.)"""
function _find_arrayop_local(expr)
    expr isa SymbolicT || return nothing
    @match expr begin
        SU.BSImpl.ArrayOp(;) => return expr
        SU.BSImpl.Term(; f, args) => begin
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
        SU.BSImpl.ArrayOp(; output_idx, ranges, shape = sh) => begin
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
    # Build observed substitution dict from TWO sources:
    # 1. Scalar observed equations in sys.observed (O(M_scalar))
    # 2. Block algebraic representatives from eliminated_block_eqs (O(M_block))
    # This is O(M) total, NOT O(N).
    obs_dict = Dict{SymbolicT, Any}()

    # Source 1: scalar observed
    for eq in observed(sys)
        lhs_uw = unwrap(eq.lhs)
        obs_dict[lhs_uw] = unwrap(eq.rhs)
    end

    # Source 2: eliminated block algebraic representatives
    # The representative defines v[k0] ~ rhs_at_k0. We add this single entry.
    # The parameterized ForLoop will shift the indices automatically.
    elim_blocks = getmetadata(sys, EliminatedBlockEquationsKey, nothing)
    if elim_blocks !== nothing
        for block in elim_blocks
            rep = block.representative_eq
            rep_lhs = unwrap(rep.lhs)
            SU._iszero(rep_lhs) && continue  # Skip algebraic constraints (0 ~ expr)
            obs_dict[rep_lhs] = unwrap(rep.rhs)
        end
    end

    isempty(obs_dict) && return rhss

    new_rhss = collect(rhss)
    for (i, rhs) in enumerate(rhss)
        block = get(block_eqs_meta, i, nothing)
        block === nothing && continue
        isdiffeq(block.representative_eq) || continue
        # Inline all observed into this block representative RHS.
        # maxiters=100: observed chains are typically shallow (1-3 levels for
        # algebraic elimination), but we allow headroom for deeply nested cases.
        new_rhss[i] = Symbolics.fixpoint_sub(unwrap(rhs), obs_dict; maxiters=100)
    end
    return new_rhss
end

"""
    _resolve_block_observed_expr(sys, sym, block_eqs_meta)

For block-observed variables (eliminated algebraic blocks), resolve `sym` to concrete
expressions computable from unknowns. Returns resolved expression(s), or `nothing`
if `sym` is not a block-observed variable.

- `sym = v(t)[i]` → returns scalar expression for v[i]
- `sym = v(t)` (array) → returns array of expressions [v[1], ..., v[N]]

This enables O(M) storage: block algebraic equations are NOT expanded to N scalar
observed equations. Instead, the representative equation + index shifting generates
the expression on demand.
"""
function _resolve_block_observed_expr(sys, sym, block_eqs_meta)
    sym_uw = unwrap(sym)

    # Build lookup: base_variable => (block, rep_idx_vals, iter_ranges)
    # Includes both namespaced and un-namespaced keys for matching compiled.v[i]
    block_lookup = _build_block_observed_lookup(block_eqs_meta, sys)
    isempty(block_lookup) && return nothing

    # Case 1: sym = v(t)[i] — indexed access to a single element
    if SU.iscall(sym_uw) && operation(sym_uw) === getindex
        base_var = arguments(sym_uw)[1]
        info = get(block_lookup, base_var, nothing)
        info === nothing && return nothing

        block, rep_idx_vals, _, _ = info
        target_idxs = [Int(SU.unwrap_const(arguments(sym_uw)[k])) for k in 2:length(arguments(sym_uw))]
        shifts = [target_idxs[d] - rep_idx_vals[d] for d in eachindex(rep_idx_vals)]

        rep_rhs = unwrap(block.representative_eq.rhs)
        return _shift_array_indices_multidim_sym(rep_rhs, shifts)
    end

    # Case 2: sym = v(t) — full array access
    for (_, (block, rep_idx_vals, iter_ranges, orig_base_var)) in block_lookup
        if isequal(sym_uw, orig_base_var) || isequal(sym_uw, renamespace(sys, orig_base_var))
            rep_rhs = unwrap(block.representative_eq.rhs)
            results = SymbolicT[]
            for idx_tuple in Iterators.product(iter_ranges...)
                shifts = [idx_tuple[d] - rep_idx_vals[d] for d in eachindex(rep_idx_vals)]
                push!(results, _shift_array_indices_multidim_sym(rep_rhs, shifts))
            end
            return results
        end
    end

    return nothing
end

"""Build lookup from base variable to block info for eliminated observed blocks.
Stores both namespaced and un-namespaced keys to handle compiled.v[i] access."""
function _build_block_observed_lookup(block_eqs_meta, sys)
    BlockObsInfo = Tuple{Any, Vector{Int}, Vector, SymbolicT}
    lookup = Dict{SymbolicT, BlockObsInfo}()
    elim_blocks = getmetadata(sys, EliminatedBlockEquationsKey, nothing)
    elim_blocks === nothing && return lookup
    for block in elim_blocks
        rep = block.representative_eq
        rep_lhs = unwrap(rep.lhs)
        SU._iszero(rep_lhs) && continue

        SU.iscall(rep_lhs) && operation(rep_lhs) === getindex || continue
        base_var = arguments(rep_lhs)[1]

        # Extract representative index values and iteration ranges from original ArrayOp
        ao = _find_arrayop_local(unwrap(block.original_eq.lhs))
        ao === nothing && (ao = _find_arrayop_local(unwrap(block.original_eq.rhs)))
        ao === nothing && continue

        output_idx, ranges, sh = _get_arrayop_index_info_local(ao)
        isempty(output_idx) && continue

        rep_idx_vals = Int[]
        for (dim_i, ii) in enumerate(output_idx)
            push!(rep_idx_vals, haskey(ranges, ii) ? first(ranges[ii]) : first(sh[dim_i]))
        end

        iter_ranges = [haskey(ranges, ii) ? ranges[ii] : sh[dim_i]
                       for (dim_i, ii) in enumerate(output_idx)]

        info = (block, rep_idx_vals, iter_ranges, base_var)
        lookup[base_var] = info
        # Also store namespaced key for matching compiled.v[i]
        lookup[renamespace(sys, base_var)] = info
    end
    return lookup
end

"""
    _block_jacobian_sparsity(sys, block_eqs)

Compute Jacobian sparsity from block stencil structure. For each block equation,
the representative RHS has a fixed stencil pattern (which unknowns it references).
This pattern is tiled across all N elements of the block, producing a banded/sparse
matrix instead of a dense N×N pattern.

For scalar (non-block) equations, sparsity is computed via `Symbolics.jacobian_sparsity`.

Uses `_build_block_outputidxs` for consistent equation→du row mapping (same mapping
used by `generate_rhs` for code generation).

!!! note
    This assumes a spatially uniform stencil — every element in a block has the same
    dependency offsets as the representative. This is correct for standard finite
    difference discretizations on uniform grids but would be wrong for spatially-varying
    stencils or non-uniform grids.
"""
function _block_jacobian_sparsity(sys, block_eqs)
    N = length(unknowns(sys))
    dvs = [unwrap(dv) for dv in unknowns(sys)]
    eqs = equations(sys)

    # Get the consistent equation→du row mapping
    outputidxs = _build_block_outputidxs(eqs, sys)

    row_idxs = Int[]
    col_idxs = Int[]

    for (i, eq) in enumerate(eqs)
        row = outputidxs[i]
        block = get(block_eqs, i, nothing)

        if block !== nothing && block.scalar_count > 1
            # Block equation: compute column offsets from the representative's RHS
            # by finding which unknowns appear in it and computing their index offsets.
            rep_rhs = unwrap(eq.rhs)

            # Get column indices of unknowns referenced in the representative RHS
            rep_sp = Symbolics.jacobian_sparsity([rep_rhs], dvs)
            _, rep_cols, _ = SparseArrays.findnz(rep_sp)

            # Compute offsets relative to the representative's row position
            offsets = [col - row for col in rep_cols]

            # Tile the offsets across all N elements of this block
            n = block.scalar_count
            for k in 0:(n - 1)
                tile_row = row + k
                tile_row > N && continue
                for offset in offsets
                    col = tile_row + offset
                    if 1 <= col <= N
                        push!(row_idxs, tile_row)
                        push!(col_idxs, col)
                    end
                end
            end
        else
            # Scalar equation: compute sparsity directly via Symbolics
            scalar_sp = Symbolics.jacobian_sparsity([unwrap(eq.rhs)], dvs)
            _, cols, _ = SparseArrays.findnz(scalar_sp)
            for col in cols
                push!(row_idxs, row)
                push!(col_idxs, col)
            end
        end
    end

    return SparseArrays.sparse(row_idxs, col_idxs, true, N, N)
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
            # Algebraic equation 0 ~ rhs — block systems should not have these
            # (algebraic blocks are eliminated during tearing). Fall back to
            # sequential indexing as a safe default.
            @warn "Unexpected algebraic equation (0 ~ rhs) in block system at position $(length(outputidxs)+1)"
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
    # Pre-compute block metadata for both 1D and multi-dimensional blocks
    block_info_list = []
    for (eq_idx, block) in block_eqs_meta
        eq_idx < 0 && continue
        isdiffeq(block.representative_eq) || continue
        rep_lhs = unwrap(block.representative_eq.lhs)
        rep_var = arguments(rep_lhs)[1]  # u(t)[k] or u(t)[i,j]
        rep_pos = variable_index(sys, rep_var)
        rep_pos === nothing && continue

        # Get the ArrayOp's iteration info for this block
        ao = _find_arrayop_local(unwrap(block.original_eq.lhs))
        ao === nothing && (ao = _find_arrayop_local(unwrap(block.original_eq.rhs)))
        ao === nothing && continue
        output_idx, ranges, sh = _get_arrayop_index_info_local(ao)
        isempty(output_idx) && continue

        # Get per-dimension iteration ranges
        iter_ranges = [haskey(ranges, ii) ? ranges[ii] : sh[dim_i]
                       for (dim_i, ii) in enumerate(output_idx)]

        # Get per-dimension base indices from the representative variable
        base_idxs = Int[]
        if SU.iscall(rep_var) && operation(rep_var) === getindex
            rep_args = arguments(rep_var)
            for k in 2:length(rep_args)
                idx_arg = rep_args[k]
                if SU.isconst(idx_arg)
                    push!(base_idxs, Int(SU.unwrap_const(idx_arg)))
                else
                    push!(base_idxs, k <= length(iter_ranges) ? first(iter_ranges[k-1]) : 1)
                end
            end
        else
            base_idxs = [rep_pos]
        end

        du_range = rep_pos:(rep_pos + block.scalar_count - 1)
        ndims = length(output_idx)

        push!(block_info_list, (eq_idx=eq_idx, du_range=du_range, rep_pos=rep_pos,
            base_idxs=base_idxs, iter_ranges=iter_ranges, ndims=ndims))
    end

    function iip_transform(func::Func)
        isempty(block_info_list) && return func

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
        rep_rhs_map = Dict{Int, Any}()
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
            rep_rhs = get(rep_rhs_map, bi.rep_pos, nothing)
            rep_rhs === nothing && continue

            if bi.ndims == 1
                # 1D block: single ForLoop
                loop_var = SU.Sym{SU.SymReal}(Symbol("__blk_k_$(bi.eq_idx)");
                    type = Int, shape = SU.ShapeVecT())
                param_rhs = _parameterize_symbolic_rhs(rep_rhs, bi.base_idxs[1], loop_var)
                du_offset = bi.rep_pos - bi.base_idxs[1]
                du_idx = du_offset == 0 ? loop_var : loop_var + du_offset
                inner = SetArray(true, arr_sym, [AtIndex(du_idx, param_rhs)], false)
                push!(forloops, ForLoop(loop_var, bi.du_range, inner))
            else
                # Multi-dimensional block: nested ForLoops
                # Create per-dimension loop variables
                loop_vars = [SU.Sym{SU.SymReal}(
                    Symbol("__blk_$(d)_$(bi.eq_idx)"); type = Int, shape = SU.ShapeVecT())
                    for d in 1:bi.ndims]

                # Parameterize RHS with multi-dimensional loop variables
                param_rhs = _parameterize_symbolic_rhs_multidim(rep_rhs, bi.base_idxs, loop_vars)

                # Compute linear du index from multi-dimensional loop variables
                # Unknowns are in column-major order from Iterators.product(shape...)
                # Linear index = rep_pos + (i - base_i) + stride_1 * (j - base_j) + ...
                strides = _compute_strides(bi.iter_ranges)
                du_idx_expr = _build_linear_index_expr(loop_vars, bi.base_idxs, strides, bi.rep_pos)

                # Build nested ForLoops via foldl (innermost first)
                inner = SetArray(true, arr_sym, [AtIndex(du_idx_expr, param_rhs)], false)
                loop = foldl(reverse(collect(enumerate(zip(loop_vars, bi.iter_ranges)))); init=inner) do body, (d, (lv, rng))
                    ForLoop(lv, rng, body)
                end
                push!(forloops, loop)
            end
        end

        # Rebuild body: scalar SetArray + ForLoops, return arr
        new_scalar_set = SetArray(set_array.inbounds, arr_sym, scalar_entries, false)
        inner_pairs = Union{Assignment, Any}[Assignment(Symbol("##scalar_out##"), new_scalar_set)]
        for (fi, fl) in enumerate(forloops)
            push!(inner_pairs, Assignment(Symbol("##loop_$(fi)##"), fl))
        end
        new_inner = Let(inner_pairs, arr_sym, false)

        new_body = _replace_setarray_body(func.body, let_body, new_inner)
        return Func(func.args, func.kwargs, new_body, func.pre)
    end

    return iip_transform
end

"""Compute strides for column-major indexing from ranges."""
function _compute_strides(iter_ranges)
    strides = Int[1]
    for i in 1:(length(iter_ranges)-1)
        push!(strides, strides[end] * length(iter_ranges[i]))
    end
    return strides
end

"""Build a symbolic linear index expression from multi-dim loop variables."""
function _build_linear_index_expr(loop_vars, base_idxs, strides, rep_pos)
    # linear = rep_pos + sum((loop_var_d - base_idx_d) * stride_d for d in dims)
    expr = rep_pos
    for d in eachindex(loop_vars)
        offset_d = loop_vars[d] - base_idxs[d]
        if strides[d] == 1
            expr = expr + offset_d
        else
            expr = expr + offset_d * strides[d]
        end
    end
    return expr
end

"""
    _parameterize_symbolic_rhs(expr, base_idx, loop_var)

Replace all concrete-integer `getindex(arr, Const(int))` in a symbolic expression
with parameterized loop-variable expressions: `getindex(arr, loop_var + offset)`.

Only handles 1D getindex (single index dimension). For multi-dimensional blocks,
use `_parameterize_symbolic_rhs_multidim` instead.
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
    sub = SU.Substituter{false}(sub_dict, allow_all)
    return sub(expr)
end

"""
Collect getindex substitution rules, building replacements with `term()` directly.
"""
function _collect_symbolic_getindex_subs!(sub_dict, expr, base_idx, loop_var)
    expr isa SymbolicT || return
    SU.iscall(expr) || return
    f = operation(expr)
    args = arguments(expr)
    if f === getindex && length(args) >= 2
        idx = args[2]
        if SU.isconst(idx)
            val = Int(SU.unwrap_const(idx))
            offset = val - base_idx
            new_idx = offset == 0 ? loop_var : loop_var + offset
            # Build replacement using term() with explicit type/shape
            expr_type = SU.symtype(expr)
            replacement = SU.term(getindex, args[1], new_idx;
                type = expr_type, shape = SU.ShapeVecT())
            sub_dict[expr] = replacement
            return  # Don't recurse
        end
    end
    for a in args
        _collect_symbolic_getindex_subs!(sub_dict, a, base_idx, loop_var)
    end
end

"""
    _parameterize_symbolic_rhs_multidim(expr, base_idxs, loop_vars)

Multi-dimensional version of _parameterize_symbolic_rhs. Replaces all concrete-integer
`getindex(arr, Const(i1), Const(i2), ...)` with `getindex(arr, lv1+off1, lv2+off2, ...)`.
"""
function _parameterize_symbolic_rhs_multidim(expr, base_idxs::Vector{Int}, loop_vars)
    sub_dict = Dict{SymbolicT, Any}()
    _collect_symbolic_getindex_subs_multidim!(sub_dict, expr, base_idxs, loop_vars)
    isempty(sub_dict) && return expr
    allow_all = (_) -> true
    sub = SU.Substituter{false}(sub_dict, allow_all)
    return sub(expr)
end

"""
Collect multi-dimensional getindex substitution rules.
For getindex(arr, Const(i1), Const(i2), ...), builds a replacement with
loop variables: getindex(arr, lv1 + (i1-base1), lv2 + (i2-base2), ...).
"""
function _collect_symbolic_getindex_subs_multidim!(sub_dict, expr, base_idxs, loop_vars)
    expr isa SymbolicT || return
    SU.iscall(expr) || return
    f = operation(expr)
    args = arguments(expr)
    if f === getindex && length(args) >= 2
        n_idx = length(args) - 1
        # Check if all indices are concrete integers
        all_const = all(2:length(args)) do k
            SU.isconst(args[k])
        end
        if all_const && n_idx == length(base_idxs) && n_idx == length(loop_vars)
            new_idx_args = Any[]
            for d in 1:n_idx
                val = Int(SU.unwrap_const(args[d+1]))
                offset = val - base_idxs[d]
                new_idx = offset == 0 ? loop_vars[d] : loop_vars[d] + offset
                push!(new_idx_args, new_idx)
            end
            expr_type = SU.symtype(expr)
            replacement = SU.term(getindex, args[1], new_idx_args...;
                type = expr_type, shape = SU.ShapeVecT())
            sub_dict[expr] = replacement
            return  # Don't recurse into matched getindex
        end
    end
    for a in args
        _collect_symbolic_getindex_subs_multidim!(sub_dict, a, base_idxs, loop_vars)
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

