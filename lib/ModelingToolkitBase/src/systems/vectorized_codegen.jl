"""
Post-processing utilities for vectorizing ArrayOp-covered equations in generated code.

When a system has ArrayEquationsCtx metadata (from PDE discretization with ArrayOp),
the generated IIP function can be post-processed to replace groups of scalar
assignments `du[i] = rhs_i` with compact `for` loops. This dramatically reduces
Julia compilation time for large PDE systems.
"""

"""
    find_arrayop_equation_indices(sys, arrayop_infos)

Given a compiled system and its ArrayOp metadata, determine which equation indices
in the compiled system correspond to each ArrayOp equation.

Returns a Vector of UnitRanges, one per ArrayOp equation.
"""
function find_arrayop_equation_indices(sys, arrayop_infos)
    eqs = equations(sys)
    # Build lookup from LHS to equation index
    lhs_to_idx = Dict{SymbolicT, Int}()
    for (i, eq) in enumerate(eqs)
        lhs_to_idx[unwrap(eq.lhs)] = i
    end

    # Collect all covered indices from all ArrayOp equations
    all_indices = Int[]
    for info in arrayop_infos
        scalar_eqs = flatten_equation(info.equation)
        for seq in scalar_eqs
            idx = get(lhs_to_idx, unwrap(seq.lhs), nothing)
            if idx !== nothing
                push!(all_indices, idx)
            end
        end
    end

    isempty(all_indices) && return UnitRange{Int}[]

    # Sort and merge into contiguous ranges
    sort!(unique!(all_indices))
    ranges = UnitRange{Int}[]
    range_start = all_indices[1]
    range_end = all_indices[1]
    for i in 2:length(all_indices)
        if all_indices[i] == range_end + 1
            range_end = all_indices[i]
        else
            push!(ranges, range_start:range_end)
            range_start = all_indices[i]
            range_end = all_indices[i]
        end
    end
    push!(ranges, range_start:range_end)
    return ranges
end

"""
    vectorize_iip_expr(res, sys, arrayop_infos, eqs)

Post-process the generated IIP function Expr to replace ArrayOp-covered scalar
assignments with `for` loops.

`res` is the `(oop_expr, iip_expr)` tuple from `build_function_wrapper`.
Returns a modified tuple.
"""
function vectorize_iip_expr(res, sys, arrayop_infos, eqs)
    oop_expr, iip_expr = res
    if !Meta.isexpr(iip_expr, :function)
        return res
    end

    # Find which equation indices are covered by ArrayOp
    covered_ranges = find_arrayop_equation_indices(sys, arrayop_infos)
    isempty(covered_ranges) && return res

    # All covered indices
    covered_set = Set{Int}()
    for r in covered_ranges
        union!(covered_set, r)
    end

    # Post-process the IIP expression by walking the Expr tree
    try
        new_iip = deepcopy(iip_expr)
        _replace_assignments_recursive!(new_iip, covered_ranges, covered_set)
        return (oop_expr, new_iip)
    catch e
        @warn "ArrayOp vectorization failed, falling back to scalar codegen" exception=(e, catch_backtrace())
        return res
    end
end

"""
Recursively walk an Expr, finding blocks that contain `out[i] = rhs` assignments.
Replace covered assignments with for loops.
"""
function _replace_assignments_recursive!(expr::Expr, covered_ranges, covered_set)
    if expr.head === :block
        # Check if this block directly contains out[i] = rhs assignments
        has_out_assignments = any(expr.args) do arg
            idx, _ = _extract_assignment_index(arg)
            idx !== nothing
        end

        if has_out_assignments
            _replace_in_block!(expr, covered_ranges, covered_set)
            return
        end
    end

    # Recurse into sub-expressions
    for i in eachindex(expr.args)
        if expr.args[i] isa Expr
            _replace_assignments_recursive!(expr.args[i], covered_ranges, covered_set)
        end
    end
end

"""
In a block that contains `out[i] = rhs` assignments, replace covered ones with for loops.
"""
function _replace_in_block!(block::Expr, covered_ranges, covered_set)
    # Collect all output assignments and their positions
    out_sym = nothing
    assignment_map = Dict{Int, Tuple{Int, Any}}() # out_idx => (pos_in_block, rhs_expr)

    for (pos, stmt) in enumerate(block.args)
        idx, sym = _extract_assignment_index(stmt)
        if idx !== nothing
            out_sym = sym
            assignment_map[idx] = (pos, stmt.args[2])  # rhs
        end
    end

    out_sym === nothing && return

    # Build new block args
    new_args = Any[]
    positions_to_skip = Set{Int}()

    # First, identify positions to skip (covered assignments) and where to insert loops
    loop_insert_positions = Dict{Int, Expr}()  # block position => loop expr
    for range in covered_ranges
        first_idx = first(range)
        if !haskey(assignment_map, first_idx)
            continue
        end

        # Mark all positions in this range for removal
        for idx in range
            if haskey(assignment_map, idx)
                pos, _ = assignment_map[idx]
                push!(positions_to_skip, pos)
            end
        end

        # Build for loop from template
        _, template_rhs = assignment_map[first_idx]
        loop_var = Symbol("##__arrayop_idx##")

        if length(range) == 1
            # Single element: no loop needed, just keep the assignment
            # (but still generate the parameterized form for correctness)
            for idx in range
                if haskey(assignment_map, idx)
                    delete!(positions_to_skip, assignment_map[idx][1])
                end
            end
            continue
        end

        parameterized_rhs = _parameterize_indices(template_rhs, first_idx, loop_var)

        # Build the for loop expression
        # for loop_var in first:last
        #     out[loop_var] = parameterized_rhs
        # end
        loop_range = first(range):last(range)
        loop_assignment = :($out_sym[$loop_var] = $parameterized_rhs)
        loop_body = Expr(:block, loop_assignment)
        loop_iter = :($loop_var = $loop_range)
        loop_expr = Expr(:for, loop_iter, loop_body)

        # Insert the loop at the position of the first assignment in this range
        first_pos, _ = assignment_map[first_idx]
        loop_insert_positions[first_pos] = loop_expr
    end

    # Rebuild the block
    for (pos, stmt) in enumerate(block.args)
        if pos in positions_to_skip
            # Check if we should insert a loop here
            if haskey(loop_insert_positions, pos)
                push!(new_args, loop_insert_positions[pos])
            end
        else
            push!(new_args, stmt)
        end
    end

    block.args = new_args
end

"""
Extract the output array index from an assignment expression like `out[5] = rhs`.
Returns (index::Int, out_symbol) or (nothing, nothing).
"""
function _extract_assignment_index(stmt)
    if !isa(stmt, Expr) || stmt.head !== :(=)
        return nothing, nothing
    end
    lhs = stmt.args[1]
    if !isa(lhs, Expr) || lhs.head !== :ref || length(lhs.args) != 2
        return nothing, nothing
    end
    idx = lhs.args[2]
    if idx isa Integer
        return Int(idx), lhs.args[1]
    end
    return nothing, nothing
end

"""
    _parameterize_indices(expr, base_idx, loop_var)

In the RHS expression, replace every occurrence of `arg[k]` (where k is a literal
integer) with `arg[loop_var + (k - base_idx)]`. This parameterizes the stencil so
that it can be used in a loop.
"""
function _parameterize_indices(expr, base_idx::Int, loop_var::Symbol)
    if !isa(expr, Expr)
        return expr
    end

    if expr.head === :ref && length(expr.args) == 2 && expr.args[2] isa Integer
        arr = expr.args[1]
        idx = Int(expr.args[2])
        offset = idx - base_idx
        if offset == 0
            return Expr(:ref, arr, loop_var)
        else
            return Expr(:ref, arr, Expr(:call, :+, loop_var, offset))
        end
    end

    # Recurse into sub-expressions
    new_args = Any[_parameterize_indices(a, base_idx, loop_var) for a in expr.args]
    return Expr(expr.head, new_args...)
end
