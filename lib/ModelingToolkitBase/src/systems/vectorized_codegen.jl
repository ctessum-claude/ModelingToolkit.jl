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

"""
    _find_arrayop(expr)

Find the first `ArrayOp` node in a symbolic expression, descending through
operator wrappers like `Differential`. Returns the `ArrayOp` or `nothing`.

Implementation: uses `SU.search_variables!` with `SU.isarrayop` as the
atomicity predicate — this treats `ArrayOp` nodes as leaves and returns the
first one encountered.

This is the MTKBase-level canonical definition. `ModelingToolkitTearing`
re-exports it so tearing code can call it unqualified.
"""
function _find_arrayop(expr)
    expr isa SymbolicT || return nothing
    buffer = Set{SymbolicT}()
    SU.search_variables!(buffer, expr; is_atomic = SU.isarrayop)
    isempty(buffer) && return nothing
    return first(buffer)
end

"""
    _get_arrayop_index_info(ao)

Extract the iteration metadata from an `ArrayOp` node: returns
`(output_idx_symbols, ranges_dict, shape)` where `output_idx_symbols` is a
vector of the symbolic output indices (`Int` slots are dropped), `ranges_dict`
maps each output index symbol to its iteration range, and `shape` is the
ArrayOp's declared shape.

This is the MTKBase-level canonical definition. `ModelingToolkitTearing`
re-exports it so tearing code can call it unqualified.
"""
function _get_arrayop_index_info(ao)
    @match ao begin
        BSImpl.ArrayOp(; output_idx, ranges, shape = sh) => begin
            sym_idxs = [(dim_i, ii) for (dim_i, ii) in enumerate(output_idx) if !(ii isa Int)]
            return SymbolicT[ii for (_, ii) in sym_idxs], ranges, sh
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
    # Build substitution dict from TWO sources:
    # 1. Scalar observed equations in sys.observed — reuse `get_substitutions`,
    #    which already flattens observed-to-observed references.
    # 2. Block algebraic representatives from eliminated_block_eqs — a single
    #    entry per block; the parameterized ForLoop shifts indices automatically.
    # This is O(M) total, NOT O(N).
    obs_dict = Dict{SymbolicT, Any}(get_substitutions(sys))

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

"""
    _register_block_observed_base_timeseries!(dict, sys)

Register the base array of each eliminated block (e.g. `v(t)` when `v[i]` has
been eliminated algebraically) in `dict` with an empty `TimeseriesSetType()`.
This is used to populate `IndexCache.observed_syms_to_timeseries` so that
`_all_ts_idxs!` can recognize block-observed variables as continuous-timeseries
via the same lookup path used for ordinary observed variables.

Only the base array (and its namespaced variant) is registered, not every
`v[i]` element — callers must fall back to the base key via `split_indexed_var`
when looking up a specific element. This keeps the cache O(M_block) instead
of O(N) in the grid size.
"""
function _register_block_observed_base_timeseries!(dict, sys)
    elim_blocks = getmetadata(sys, EliminatedBlockEquationsKey, nothing)
    elim_blocks === nothing && return dict
    for block in elim_blocks
        rep = block.representative_eq
        rep_lhs = unwrap(rep.lhs)
        SU._iszero(rep_lhs) && continue
        SU.iscall(rep_lhs) && operation(rep_lhs) === getindex || continue
        base_var = arguments(rep_lhs)[1]
        dict[base_var] = TimeseriesSetType()
        dict[renamespace(sys, base_var)] = TimeseriesSetType()
    end
    return dict
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
        ao = _find_arrayop(unwrap(block.original_eq.lhs))
        ao === nothing && (ao = _find_arrayop(unwrap(block.original_eq.rhs)))
        ao === nothing && continue

        output_idx, ranges, sh = _get_arrayop_index_info(ao)
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
            # (algebraic blocks are eliminated during tearing). If one appears here
            # we can't produce a correct du-row mapping, so fail loudly rather than
            # silently computing wrong Jacobian sparsity.
            error("Unexpected algebraic equation (0 ~ rhs) at position $(length(outputidxs)+1) \
                   in block system. This should have been eliminated during tearing. \
                   Please file a bug report.")
        else
            # v(t)[k] ~ rhs — variable_index of v(t)[k]
            pos = variable_index(sys, lhs)
            push!(outputidxs, pos)
        end
    end
    return outputidxs
end

"""
Shift array indices in a symbolic expression by per-dimension shifts using
`Postwalk`. Every `getindex(arr, i, j, ...)` node in `expr` (including those
inside `Differential` operators) is rewritten to `getindex(arr, i+shifts[1],
j+shifts[2], ...)`. The `filter = Returns(true)` kwarg makes the walker
descend through operator subtrees.

`getindex` replacements are built with `SU.term(...)` rather than `arr[i]`
to bypass the `_getindex` overload that collapses nested getindex on
constant-indexed parents — see `SymbolicUtils/src/symbolic_ops/getindex.jl`.
"""
function _shift_array_indices_multidim_sym(expr, shifts::Vector{Int})
    all(iszero, shifts) && return expr
    expr isa SymbolicT || return expr
    rw = let shifts = shifts
        x -> begin
            (SU.iscall(x) && operation(x) === getindex) || return x
            args = arguments(x)
            length(args) >= 2 || return x
            new_idxs = Any[]
            for k in 1:(length(args) - 1)
                idx = args[k + 1]
                push!(new_idxs, (k <= length(shifts) && shifts[k] != 0) ? idx + shifts[k] : idx)
            end
            return SU.term(getindex, args[1], new_idxs...;
                type = SU.symtype(x), shape = SU.ShapeVecT())
        end
    end
    return Postwalk(rw; filter = Returns(true))(expr)
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
        ao = _find_arrayop(unwrap(block.original_eq.lhs))
        ao === nothing && (ao = _find_arrayop(unwrap(block.original_eq.rhs)))
        ao === nothing && continue
        output_idx, ranges, sh = _get_arrayop_index_info(ao)
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

        # Walk the Func body, transforming every SetArray we encounter. The walker
        # preserves enclosing `Let`/`ForLoop` structure so it's robust to whatever
        # wrapper layers have been composed into the body.
        transformed = Ref(false)
        new_body = _map_setarrays(func.body) do set_array
            transformed[] = true
            return _rewrite_setarray_for_blocks(set_array, block_info_list)
        end

        # If we never found a SetArray, the IR shape changed underneath us (e.g.
        # sparse output, a new wrapper layer, or a CSE pass). Fail loudly so
        # regressions show up in tests instead of silently skipping the
        # O(M) codegen optimization.
        transformed[] || error(
            "`_make_block_forloop_wrap_code`: no `SetArray` found in generated IIP \
             function body. The Symbolics/MTKBase code-generation IR has changed \
             shape; the block-tearing ForLoop transform must be updated. Body \
             type: $(typeof(func.body)).")

        return Func(func.args, func.kwargs, new_body, func.pre)
    end

    return iip_transform
end

"""
Rewrite a `SetArray` so that entries at block-equation du positions are replaced
with `ForLoop` assignments that parameterize the block representative RHS. The
return value is a `Let` that first runs the remaining scalar `SetArray`, then
each generated `ForLoop`, then returns `arr_sym`.
"""
function _rewrite_setarray_for_blocks(set_array::SetArray, block_info_list)
    arr_sym = set_array.arr

    # Collect all du positions belonging to blocks.
    block_du_positions = Set{Int}()
    for bi in block_info_list
        union!(block_du_positions, bi.du_range)
    end

    # Separate scalar and block AtIndex entries.
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

    # Build ForLoops for each block.
    forloops = []
    for bi in block_info_list
        rep_rhs = get(rep_rhs_map, bi.rep_pos, nothing)
        rep_rhs === nothing && continue

        # Allocate fresh symbolic loop variables (one per block dimension).
        loop_vars = [SU.Sym{SU.SymReal}(
            Symbol("__blk_$(d)_$(bi.eq_idx)"); type = Int, shape = SU.ShapeVecT())
            for d in 1:bi.ndims]
        param_rhs = _parameterize_symbolic_rhs(rep_rhs, bi.base_idxs, loop_vars)

        if bi.ndims == 1
            # 1D block: the du index is just a shift of the single loop variable.
            loop_var = loop_vars[1]
            du_offset = bi.rep_pos - bi.base_idxs[1]
            du_idx = du_offset == 0 ? loop_var : loop_var + du_offset
            inner = SetArray(true, arr_sym, [AtIndex(du_idx, param_rhs)], false)
            push!(forloops, ForLoop(loop_var, bi.du_range, inner))
        else
            # Multi-dimensional block: linear du index is
            #   rep_pos + sum_d (loop_var_d - base_idx_d) * stride_d
            # where strides are column-major from Iterators.product(shape...).
            strides = _compute_strides(bi.iter_ranges)
            du_idx_expr = _build_linear_index_expr(loop_vars, bi.base_idxs, strides, bi.rep_pos)

            # Build nested ForLoops via foldl (innermost first).
            inner = SetArray(true, arr_sym, [AtIndex(du_idx_expr, param_rhs)], false)
            loop = foldl(reverse(collect(enumerate(zip(loop_vars, bi.iter_ranges)))); init=inner) do body, (d, (lv, rng))
                ForLoop(lv, rng, body)
            end
            push!(forloops, loop)
        end
    end

    # Rebuild the SetArray site: scalar SetArray first, then each ForLoop, then
    # return arr_sym. Each step is wrapped in a Let Assignment so toexpr emits
    # them as a sequence.
    new_scalar_set = SetArray(set_array.inbounds, arr_sym, scalar_entries, false)
    inner_pairs = Union{Assignment, Any}[Assignment(Symbol("##scalar_out##"), new_scalar_set)]
    for (fi, fl) in enumerate(forloops)
        push!(inner_pairs, Assignment(Symbol("##loop_$(fi)##"), fl))
    end
    return Let(inner_pairs, arr_sym, false)
end

"""
    _map_setarrays(f, node)

Recursively walk a `SymbolicUtils.Code` IR tree and apply `f::SetArray -> Any`
to every `SetArray` encountered, reconstructing the enclosing `Let`/`ForLoop`/
`Func`/`Assignment` structure around the transformed result. Non-IR subtrees
(symbolic expressions, literals) are returned unchanged.

This replaces the earlier pattern-matching chain walker that assumed a specific
`Let -> Let -> ... -> Let -> SetArray` nesting — that assumption was brittle
and silently failed on sparse outputs and any future wrapper layer.
"""
function _map_setarrays(f, node::SetArray)
    return f(node)
end
function _map_setarrays(f, node::Let)
    new_pairs = map(node.pairs) do pair
        if pair isa Assignment
            new_rhs = _map_setarrays(f, pair.rhs)
            new_rhs === pair.rhs ? pair : Assignment(pair.lhs, new_rhs)
        else
            pair
        end
    end
    new_body = _map_setarrays(f, node.body)
    return Let(new_pairs, new_body, node.let_block)
end
function _map_setarrays(f, node::ForLoop)
    new_body = _map_setarrays(f, node.body)
    return ForLoop(node.itervar, node.range, new_body)
end
function _map_setarrays(f, node::Func)
    new_body = _map_setarrays(f, node.body)
    return Func(node.args, node.kwargs, new_body, node.pre)
end
function _map_setarrays(f, node::Assignment)
    new_rhs = _map_setarrays(f, node.rhs)
    return Assignment(node.lhs, new_rhs)
end
# Fallback for any other node (symbolic expressions, literals, Symbols, etc.).
_map_setarrays(f, node) = node

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
    _parameterize_symbolic_rhs(expr, base_idxs, loop_vars)

Rewrite every `getindex(arr, Const(i1), Const(i2), ...)` node in `expr` with
fully-constant indices to `getindex(arr, loop_vars[1] + (i1 - base_idxs[1]),
loop_vars[2] + (i2 - base_idxs[2]), ...)`, producing a parameterized body for
`ForLoop` codegen. Works for any number of index dimensions: `base_idxs` and
`loop_vars` must have the same length.

Uses `Postwalk(...; filter = Returns(true))` to descend through `Differential`
and other operator subtrees. Only full-rank getindex nodes whose indices are
all `Const` are rewritten — partially-symbolic indices are preserved.

Replacements are built with `SU.term(...)` to avoid the `_getindex` overload
that collapses nested getindex on constant-indexed parents.
"""
function _parameterize_symbolic_rhs(expr, base_idxs::Vector{Int}, loop_vars::Vector)
    length(base_idxs) == length(loop_vars) ||
        throw(ArgumentError("base_idxs and loop_vars must have the same length"))
    ndims = length(base_idxs)
    rw = let base_idxs = base_idxs, loop_vars = loop_vars, ndims = ndims
        x -> begin
            (SU.iscall(x) && operation(x) === getindex) || return x
            args = arguments(x)
            length(args) - 1 == ndims || return x
            # Only rewrite if every index is a concrete integer.
            all(k -> SU.isconst(args[k + 1]), 1:ndims) || return x
            new_idxs = Any[]
            for d in 1:ndims
                val = Int(SU.unwrap_const(args[d + 1]))
                offset = val - base_idxs[d]
                push!(new_idxs, offset == 0 ? loop_vars[d] : loop_vars[d] + offset)
            end
            return SU.term(getindex, args[1], new_idxs...;
                type = SU.symtype(x), shape = SU.ShapeVecT())
        end
    end
    return Postwalk(rw; filter = Returns(true))(expr)
end

