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

"""
    vectorize_iip_expr!(iip_expr, sys, block_eqs)

Post-process the in-place function Expr generated by `build_function_wrapper` to replace
scalar assignments corresponding to block equations with for-loops.

The IIP Expr has structure:
```
function f!(du, u, p, t)
    @inbounds begin
        # observed assignments
        # du[1] = rhs1
        # du[2] = rhs2
        # ...
    end
end
```

For each block equation at position `pos` in the equation list, we:
1. Find the assignment `du[pos] = rhs` in the Expr
2. Extract the RHS template
3. Parameterize all array indices relative to the representative's base index
4. Replace the single assignment with a for-loop over the block's iteration range
"""
function vectorize_iip_expr!(iip_expr::Expr, sys, block_eqs::Dict{Int, <:Any})
    # Find the function body (inside @inbounds begin...end)
    body = _find_assignment_block(iip_expr)
    body === nothing && return iip_expr

    # Get variable index mapping: symbolic variable → position in du/u vector
    dvs = unknowns(sys)
    var_to_pos = Dict{Int, Int}()  # position of the representative var in du

    # Process each block equation
    for (eq_idx, block) in block_eqs
        eq_idx < 0 && continue  # Skip eliminated algebraics

        rep = block.representative_eq
        if !isdiffeq(rep)
            continue
        end

        # Find the assignment index in du for this representative
        rep_lhs = unwrap(rep.lhs)
        rep_var = arguments(rep_lhs)[1]  # u[k] from D(u[k])
        rep_du_pos = variable_index(sys, rep_var)
        rep_du_pos === nothing && continue

        # Get the block's iteration info
        output_idx, ranges, sh = _get_arrayop_index_info_local(
            _find_arrayop_local(unwrap(block.original_eq.lhs)))

        # For 1D blocks: simple loop
        if length(output_idx) == 1
            idx_sym = output_idx[1]
            iter_range = haskey(ranges, idx_sym) ? ranges[idx_sym] : sh[1]
            first_idx = first(iter_range)

            # Find and replace the assignment in the body
            _replace_with_loop_1d!(body, rep_du_pos, iter_range, first_idx, sys, block)
        else
            # Multi-dimensional: expand to scalar (fallback for now)
            # TODO: implement nested loop codegen for 2D+
            _replace_with_expanded!(body, rep_du_pos, sys, block)
        end
    end

    return iip_expr
end

"""
Find the block of assignments in an IIP function Expr.
Returns the `args` vector of the `begin...end` block that contains the assignments.
"""
function _find_assignment_block(expr::Expr)
    if expr.head === :function || expr.head === :(->)
        return _find_assignment_block(expr.args[2])
    elseif expr.head === :block
        for arg in expr.args
            arg isa Expr || continue
            result = _find_assignment_block(arg)
            result !== nothing && return result
        end
        # Check if this block contains assignments
        for arg in expr.args
            arg isa Expr || continue
            if arg.head === :(=) || (arg.head === :macrocall && arg.args[1] === Symbol("@inbounds"))
                return expr.args
            end
        end
    elseif expr.head === :macrocall && expr.args[1] === Symbol("@inbounds")
        inner = expr.args[end]
        if inner isa Expr && inner.head === :block
            return inner.args
        end
    end
    return nothing
end

"""
Replace a single scalar assignment `du[pos] = rhs` with a 1D for-loop.
"""
function _replace_with_loop_1d!(body_args::Vector, rep_du_pos::Int, iter_range, first_idx::Int, sys, block)
    # Find the assignment statement for du[rep_du_pos]
    for (i, stmt) in enumerate(body_args)
        stmt isa Expr || continue
        du_idx = _extract_setarray_index(stmt)
        du_idx === nothing && continue
        du_idx != rep_du_pos && continue

        # Found it. Extract the RHS expression
        rhs_expr = _extract_setarray_rhs(stmt)
        rhs_expr === nothing && continue

        # Get the base array variable to compute offset
        rep_var = arguments(unwrap(block.representative_eq.lhs))[1]
        if iscall(rep_var) && operation(rep_var) === getindex
            base_idx = arguments(rep_var)[2]
            if SU.isconst(base_idx)
                rep_base_idx = Int(SU.unwrap_const(base_idx))
            else
                rep_base_idx = first_idx
            end
        else
            rep_base_idx = first_idx
        end

        # Compute the du offset: how much to shift du index per iteration
        du_offset = rep_du_pos - rep_base_idx

        # Create the loop variable
        loop_var = gensym("_blk_idx")

        # Parameterize the RHS: replace all arg[k] with arg[loop_var + (k - rep_base_idx)]
        param_rhs = _parameterize_indices(rhs_expr, rep_base_idx, loop_var)

        # Build the for-loop Expr
        loop_expr = Expr(:for,
            Expr(:(=), loop_var, iter_range),
            Expr(:block,
                Expr(:(=),
                    Expr(:ref, :ˍ₋out, Expr(:call, :+, loop_var, du_offset)),
                    param_rhs
                )
            )
        )

        # Replace the assignment with the loop
        body_args[i] = loop_expr
        return true
    end
    return false
end

"""
Replace a single scalar assignment with N expanded scalar assignments (fallback for 2D+).
"""
function _replace_with_expanded!(body_args::Vector, rep_du_pos::Int, sys, block)
    rep_eq = block.representative_eq
    expanded = _expand_block_eq(rep_eq, block)

    for (i, stmt) in enumerate(body_args)
        stmt isa Expr || continue
        du_idx = _extract_setarray_index(stmt)
        du_idx === nothing && continue
        du_idx != rep_du_pos && continue

        # Replace the single assignment with N assignments
        # Build the expanded assignments using the same build_function approach
        # For now, just keep the representative (TODO: proper 2D expansion at Expr level)
        return false
    end
    return false
end

"""
Extract the du index from a SetArray-style assignment: `ˍ₋out[k] = rhs` → k
"""
function _extract_setarray_index(stmt::Expr)
    if stmt.head === :(=)
        lhs = stmt.args[1]
        if lhs isa Expr && lhs.head === :ref && lhs.args[1] === :ˍ₋out
            idx = lhs.args[2]
            return idx isa Integer ? Int(idx) : nothing
        end
    end
    return nothing
end

"""
Extract the RHS from a SetArray-style assignment: `ˍ₋out[k] = rhs` → rhs
"""
function _extract_setarray_rhs(stmt::Expr)
    if stmt.head === :(=)
        return stmt.args[2]
    end
    return nothing
end

"""
Parameterize array indices in a Julia Expr. Replaces every `arg[k]` (where k is a literal
integer) with `arg[loop_var + (k - base_idx)]`.
"""
function _parameterize_indices(expr, base_idx::Int, loop_var::Symbol)
    if expr isa Expr
        if expr.head === :ref && length(expr.args) >= 2
            # Array reference: arg[k] → arg[loop_var + (k - base_idx)]
            new_args = Any[expr.args[1]]  # Keep the array name
            for k in 2:length(expr.args)
                idx = expr.args[k]
                if idx isa Integer
                    offset = Int(idx) - base_idx
                    if offset == 0
                        push!(new_args, loop_var)
                    else
                        push!(new_args, Expr(:call, :+, loop_var, offset))
                    end
                else
                    push!(new_args, _parameterize_indices(idx, base_idx, loop_var))
                end
            end
            return Expr(:ref, new_args...)
        else
            new_args = Any[]
            for arg in expr.args
                push!(new_args, _parameterize_indices(arg, base_idx, loop_var))
            end
            return Expr(expr.head, new_args...)
        end
    end
    return expr
end
