@doc """
    function mtkcompile(sys::System; kwargs...)

Compile the given system into a form that ModelingToolkit can generate code for. Also
performs a variety of symbolic-numeric enhancements. For ODEs, this includes processes
such as order reduction, index reduction, alias elimination and tearing. A subset of the
unknowns of the system may be eliminated as observables, eliminating the need for the
numerical solver to solve for these variables.

Does not rely on metadata to identify variables/parameters/brownians. Instead, queries
the system for which symbolic quantites belong to which category. Any variables not
present in the equations of the system will be removed in this process.

# Keyword Arguments

+ When `simplify=true`, the `simplify` function will be applied during the tearing process.
+ `allow_symbolic=false`, `allow_parameter=true`, and `conservative=false` limit the coefficient types during tearing. In particular, `conservative=true` limits tearing to only solve for trivial linear systems where the coefficient has the absolute value of ``1``.
+ `fully_determined=true` controls whether or not an error will be thrown if the number of equations don't match the number of inputs, outputs, and equations.
+ `inputs`, `outputs` and `disturbance_inputs` are passed as keyword arguments.` All inputs` get converted to parameters and are allowed to be unconnected, allowing models where `n_unknowns = n_equations - n_inputs`.
+ `sort_eqs=true` controls whether equations are sorted lexicographically before simplification or not.
""" mtkcompile

function MTKBase.__mtkcompile(
        sys::System;
        inputs::OrderedSet{SymbolicT} = OrderedSet{SymbolicT}(),
        outputs::OrderedSet{SymbolicT} = OrderedSet{SymbolicT}(),
        disturbance_inputs::OrderedSet{SymbolicT} = OrderedSet{SymbolicT}(),
        sort_eqs = true,
        kwargs...
    )
    sys, statemachines = extract_top_level_statemachines(sys)
    sys, source_info = expand_connections(sys, Val(true))
    state = TearingState(sys, source_info; sort_eqs)
    append!(state.statemachines, statemachines)

    @unpack structure, fullvars = state
    @unpack graph, var_to_diff, var_types = structure
    brown_vars = Int[]
    new_idxs = zeros(Int, length(var_types))
    idx = 0
    for (i, vt) in enumerate(var_types)
        if vt === BROWNIAN
            push!(brown_vars, i)
        else
            new_idxs[i] = (idx += 1)
        end
    end
    if isempty(brown_vars)
        block_eqs = state.block_eqs
        result = mtkcompile!(
            state; inputs, outputs, disturbance_inputs, kwargs...
        )
        if !isempty(block_eqs)
            result = _vectorize_system(result, block_eqs)
        end
        return result
    else
        Is = Int[]
        Js = Int[]
        vals = SymbolicT[]
        make_eqs_zero_equals!(state)
        new_eqs = copy(equations(state))
        dvar2eq = Dict{SymbolicT, Int}()
        eqs = equations(state)
        for (v, dv) in enumerate(var_to_diff)
            dv === nothing && continue
            deqs = 𝑑neighbors(graph, dv)
            if length(deqs) != 1
                error("$(eqs[deqs]) is not handled.")
            end
            dvar2eq[fullvars[dv]] = only(deqs)
        end
        for (j, bj) in enumerate(brown_vars), i in 𝑑neighbors(graph, bj)

            push!(Is, i)
            push!(Js, j)
            eq = new_eqs[i]
            brown = fullvars[bj]
            (coeff, residual, islinear) = Symbolics.linear_expansion(eq, brown)
            islinear || error("$brown isn't linear in $eq")
            new_eqs[i] = COMMON_ZERO ~ residual
            push!(vals, coeff)
        end
        g = Matrix(sparse(Is, Js, vals))
        sys = state.sys
        @set! sys.eqs = new_eqs
        @set! sys.unknowns = [
            v
                for (i, v) in enumerate(fullvars)
                if !iszero(new_idxs[i]) &&
                invview(var_to_diff)[i] === nothing
        ]
        ode_sys = mtkcompile(
            sys; inputs, outputs, disturbance_inputs, kwargs...
        )
        eqs = equations(ode_sys)
        sorted_g_rows = fill(COMMON_ZERO, length(eqs), size(g, 2))
        for (i, eq) in enumerate(eqs)
            dvar = eq.lhs
            # differential equations always precede algebraic equations
            _iszero(dvar) && break
            g_row = get(dvar2eq, dvar, 0)
            iszero(g_row) && error("$dvar isn't handled.")
            g_row > size(g, 1) && continue
            @views copyto!(sorted_g_rows[i, :], g[g_row, :])
        end
        # Fix for https://github.com/SciML/ModelingToolkit.jl/issues/2490
        if sorted_g_rows isa AbstractMatrix && size(sorted_g_rows, 2) == 1
            # If there's only one brownian variable referenced across all the equations,
            # we get a Nx1 matrix of noise equations, which is a special case known as scalar noise
            noise_eqs = reshape(sorted_g_rows[:, 1], (:, 1))
            is_scalar_noise = true
        elseif __num_isdiag_noise(sorted_g_rows)
            # If each column of the noise matrix has either 0 or 1 non-zero entry, then this is "diagonal noise".
            # In this case, the solver just takes a vector column of equations and it interprets that to
            # mean that each noise process is independent
            noise_eqs = __get_num_diag_noise(sorted_g_rows)
            is_scalar_noise = false
        else
            noise_eqs = sorted_g_rows
            is_scalar_noise = false
        end

        noise_eqs = substitute_observed(ode_sys, noise_eqs)
        ssys = System(
            Vector{Equation}(full_equations(ode_sys)),
            get_iv(ode_sys), unknowns(ode_sys),
            [parameters(ode_sys); collect(bound_parameters(ode_sys))]; noise_eqs,
            name = nameof(ode_sys), observed = observed(ode_sys), bindings = bindings(sys),
            initial_conditions = initial_conditions(sys),
            assertions = assertions(sys),
            guesses = guesses(sys), initialization_eqs = initialization_equations(sys),
            continuous_events = continuous_events(sys),
            discrete_events = discrete_events(sys),
            gui_metadata = get_gui_metadata(sys),
            tstops = symbolic_tstops(sys)
        )
        return ssys
    end
end

function MTKBase.simplify_sde_system(sys::System; kwargs...)
    return __mtkcompile(sys; kwargs...)
end

"""
    $(TYPEDSIGNATURES)

Given a system that has been simplified via `mtkcompile`, return a `Dict` mapping
variables of the system to equations that are used to solve for them. This includes
observed variables.

# Keyword Arguments

- `rename_dummy_derivatives`: Whether to rename dummy derivative variable keys into their
  `Differential` forms. For example, this would turn the key `yˍt(t)` into
  `Differential(t)(y(t))`.
"""
function map_variables_to_equations(sys::AbstractSystem; rename_dummy_derivatives = true)
    if !has_tearing_state(sys)
        throw(ArgumentError("$(typeof(sys)) is not supported."))
    end
    ts = get_tearing_state(sys)
    if ts === nothing
        throw(ArgumentError("`map_variables_to_equations` requires a simplified system. Call `mtkcompile` on the system before calling this function."))
    end

    dummy_sub = Dict()
    if rename_dummy_derivatives && has_schedule(sys) && (sc = get_schedule(sys)) !== nothing
        dummy_sub = Dict(v => k for (k, v) in sc.dummy_sub if isequal(default_toterm(k), v))
    end

    mapping = Dict{Union{Num, BasicSymbolic}, Equation}()
    eqs = equations(sys)
    for eq in eqs
        isdifferential(eq.lhs) || continue
        var = arguments(eq.lhs)[1]
        var = get(dummy_sub, var, var)
        mapping[var] = eq
    end

    graph = ts.structure.graph
    algvars = BitSet(
        findall(
            Base.Fix1(StateSelection.isalgvar, ts.structure), 1:ndsts(graph)
        )
    )
    algeqs = BitSet(
        findall(1:nsrcs(graph)) do eq
            all(!Base.Fix1(StateSelection.isdervar, ts.structure), 𝑠neighbors(graph, eq))
        end
    )
    alge_var_eq_matching = complete(maximal_matching(graph, in(algeqs), in(algvars)))
    for (i, eq) in enumerate(alge_var_eq_matching)
        eq isa Unassigned && continue
        mapping[get(dummy_sub, ts.fullvars[i], ts.fullvars[i])] = eqs[eq]
    end
    for eq in observed(sys)
        mapping[get(dummy_sub, eq.lhs, eq.lhs)] = eq
    end

    return mapping
end

"""
Mark whether an extra pass `p` can support compiling discrete systems.
"""
discrete_compile_pass(p) = false

"""
    _vectorize_system(sys, block_eqs)

Prepare a block-teared system for vectorized codegen. Instead of expanding M representative
equations to N scalar equations (O(N)), this function:
1. Adds all N scalar unknowns (needed for IndexCache and u0 mapping)
2. Stores block_eqs as metadata on the system (used by generate_rhs for loop codegen)
3. Expands only observed block equations (needed for user access to observed variables)
4. Keeps the M representative equations as the system's equation list

This makes mtkcompile O(1) in grid size — the only O(N) work is adding unknowns to a list.
"""
function _vectorize_system(sys::System, block_eqs::Dict{Int, MTKTearing.ArrayBlockInfo})
    compiled_dvs = unknowns(sys)

    # Rebuild unknowns list with all array elements in natural order (1,2,...,N).
    # First add all block array elements contiguously, then remaining scalar unknowns.
    # This ensures variable_index returns a contiguous range for each block.
    new_dvs = SymbolicT[]
    dvs_set = Set{SymbolicT}()

    # Collect all base arrays from ODE blocks
    block_arrays = Set{SymbolicT}()
    for (key, block) in block_eqs
        key < 0 && continue
        MTKBase.isdiffeq(block.representative_eq) || continue
        rep_lhs = unwrap(block.representative_eq.lhs)
        rep_var = arguments(rep_lhs)[1]
        if iscall(rep_var) && operation(rep_var) === getindex
            push!(block_arrays, unwrap(arguments(rep_var)[1]))
        end
    end

    # Phase 1: Add all elements of block arrays in natural order
    for base_arr in block_arrays
        base_sh = SU.shape(base_arr)
        SU.is_array_shape(base_sh) || continue
        for idx in Iterators.product(base_sh...)
            var = unwrap(base_arr[idx...])
            if !(var in dvs_set)
                push!(new_dvs, var)
                push!(dvs_set, var)
            end
        end
    end

    # Phase 2: Add remaining compiled unknowns (scalar BCs, etc.) that aren't
    # already covered by the block arrays
    for dv in compiled_dvs
        dv_uw = unwrap(dv)
        if !(dv_uw in dvs_set)
            push!(new_dvs, dv_uw)
            push!(dvs_set, dv_uw)
        end
    end

    # Block algebraic equations stay in block_eqs metadata — NOT expanded to N scalar
    # observed equations. Instead, block-level observed access (sol[compiled.v]) is
    # handled by _resolve_block_observed_expr which generates on-demand from the
    # representative equation. This keeps the entire pipeline O(M).
    new_obs = copy(observed(sys))
    new_eqs = copy(equations(sys))

    @set! sys.unknowns = new_dvs
    @set! sys.observed = new_obs
    @set! sys.eqs = new_eqs

    # Remap block_eqs keys from TearingState equation indices to compiled
    # equation indices. The codegen functions (e.g., _inline_block_observed_into_rhss)
    # look up blocks by compiled equation index, so the keys must match.
    remapped_block_eqs = Dict{Int, MTKTearing.ArrayBlockInfo}()
    for (key, block) in block_eqs
        if key < 0
            # Negative keys (eliminated algebraic blocks) stay as-is
            remapped_block_eqs[key] = block
        elseif block.compiled_eq_idx !== nothing
            # Remap to compiled equation index
            remapped_block_eqs[block.compiled_eq_idx] = block
        end
    end

    # Store remapped block_eqs metadata for codegen
    sys = SU.setmetadata(sys, MTKBase.BlockEquationsKey, remapped_block_eqs)

    return MTKBase.invalidate_cache!(sys)
end

"""Retrieve block_eqs metadata from a system, or nothing if not present."""
function _get_block_eqs(sys::System)
    SU.getmetadata(sys, MTKBase.BlockEquationsKey, nothing)
end

"""
    _expand_arrayop_blocks(sys::System, block_eqs::Dict{Int, MTKTearing.ArrayBlockInfo})

Post-process a compiled system to expand representative scalar equations back into
all N scalar equations, and add the missing scalar unknowns.

The speedup comes from structural analysis (tearing) operating on O(M) representative
equations instead of O(N*M). This function expands back to N scalar equations for
codegen compatibility. Future work can teach codegen to handle ArrayOp directly.
"""
function _expand_arrayop_blocks(sys::System, block_eqs::Dict{Int, MTKTearing.ArrayBlockInfo})
    compiled_eqs = equations(sys)
    compiled_obs = observed(sys)
    compiled_dvs = unknowns(sys)

    new_eqs = Equation[]
    new_obs = Equation[]
    new_dvs = copy(compiled_dvs)
    dvs_set = Set{SymbolicT}(unwrap.(new_dvs))

    # Expand ODE equations
    for eq in compiled_eqs
        block = _find_matching_block(eq, block_eqs)
        if block !== nothing
            expanded = _expand_block_eq(eq, block)
            append!(new_eqs, expanded)
            _add_block_unknowns!(new_dvs, dvs_set, block)
        else
            push!(new_eqs, eq)
        end
    end

    # Expand observed equations from compiled system
    for eq in compiled_obs
        block = _find_matching_block_obs(eq, block_eqs)
        if block !== nothing
            expanded = _expand_block_eq(eq, block)
            append!(new_obs, expanded)
        else
            push!(new_obs, eq)
        end
    end

    # Expand pre-substituted algebraic blocks as observed equations.
    # These were eliminated during TearingState construction (stored with negative keys)
    # and need to be added as observed equations for the solver.
    for (key, block) in block_eqs
        key >= 0 && continue  # Only process negative keys (eliminated algebraics)
        rep = block.representative_eq
        expanded = _expand_block_eq(rep, block)
        append!(new_obs, expanded)
    end

    @set! sys.eqs = new_eqs
    @set! sys.observed = new_obs
    @set! sys.unknowns = new_dvs
    return MTKBase.invalidate_cache!(sys)
end

"""
Find the ArrayBlockInfo matching a compiled equation by comparing LHS variables.
"""
function _find_matching_block(eq::Equation, block_eqs::Dict{Int, MTKTearing.ArrayBlockInfo})
    eq_lhs = unwrap(eq.lhs)
    for (_, block) in block_eqs
        rep_lhs = unwrap(block.representative_eq.lhs)
        if MTKBase.isdiffeq(eq) && MTKBase.isdiffeq(block.representative_eq)
            if isequal(arguments(eq_lhs)[1], arguments(rep_lhs)[1])
                return block
            end
        end
    end
    return nothing
end

"""
Find the ArrayBlockInfo matching an observed equation by comparing LHS variables.
"""
function _find_matching_block_obs(eq::Equation, block_eqs::Dict{Int, MTKTearing.ArrayBlockInfo})
    eq_lhs = unwrap(eq.lhs)
    SU._iszero(eq_lhs) && return nothing
    for (_, block) in block_eqs
        rep = block.representative_eq
        rep_lhs = unwrap(rep.lhs)
        if !MTKBase.isdiffeq(rep) && !SU._iszero(rep_lhs) && isequal(eq_lhs, rep_lhs)
            return block
        end
    end
    return nothing
end

"""
Expand a representative scalar equation into all N scalar equations by re-substituting
the ArrayOp's index variables with each concrete index value from the iteration ranges.

This correctly handles:
- Multi-dimensional ArrayOps (2D+ grids): iterates over Cartesian product of ranges
- Multi-variable systems: each variable's indices are shifted according to its own
  appearance in the expression, not uniformly
- Non-uniform index spaces: uses the actual iteration ranges from the ArrayOp
"""
function _expand_block_eq(compiled_rep::Equation, block::MTKTearing.ArrayBlockInfo)
    # Get the ArrayOp to extract index variables and ranges
    orig_eq = block.original_eq
    ao = MTKTearing._find_arrayop(unwrap(orig_eq.lhs))
    if ao === nothing
        ao = MTKTearing._find_arrayop(unwrap(orig_eq.rhs))
    end
    ao === nothing && return [compiled_rep]

    # Extract output_idx symbols and their iteration ranges
    output_idx, ranges, sh = _get_arrayop_index_info(ao)
    isempty(output_idx) && return [compiled_rep]

    # Get the representative's index values (what was substituted to create it)
    rep_idx_vals = Int[]
    for (dim_i, ii) in enumerate(output_idx)
        if haskey(ranges, ii)
            push!(rep_idx_vals, first(ranges[ii]))
        else
            push!(rep_idx_vals, first(sh[dim_i]))
        end
    end

    # Build the list of all iteration ranges
    iter_ranges = [haskey(ranges, ii) ? ranges[ii] : sh[dim_i]
                   for (dim_i, ii) in enumerate(output_idx)]

    # For each point in the Cartesian product of ranges, compute the per-dimension
    # shift from the representative's index values and apply it
    result = Equation[]
    for idx_tuple in Iterators.product(iter_ranges...)
        shifts = [idx_tuple[d] - rep_idx_vals[d] for d in eachindex(output_idx)]
        if all(iszero, shifts)
            push!(result, compiled_rep)
        else
            # Apply per-dimension shifts to the compiled representative
            new_lhs = _shift_array_indices_multidim(unwrap(compiled_rep.lhs), shifts)
            new_rhs = _shift_array_indices_multidim(unwrap(compiled_rep.rhs), shifts)
            push!(result, new_lhs ~ new_rhs)
        end
    end
    return result
end

"""
Extract output_idx symbols, ranges dict, and shape from an ArrayOp.
Returns (output_idx_symbols, ranges_dict, shape).
"""
function _get_arrayop_index_info(ao)
    SU.isarrayop(ao) || return SymbolicT[], Dict{SymbolicT, StepRange{Int,Int}}(), UnitRange{Int}[]
    # Use MTKTearing's _find_arrayop infrastructure to extract via @match
    # which is available in the MTKTearing module
    MTKTearing._get_arrayop_index_info(ao)
end

"""
Shift array indices in a multi-dimensional expression. `shifts` is a vector of
per-dimension shifts (e.g., [+1, +2] for a 2D system). All getindex calls in the
expression have their indices shifted: the k-th index argument is shifted by shifts[k].

This correctly handles multi-variable systems because the shift is applied to the
index positions (dimensions), not to specific variables. In a system where u[i,j]
and v[i,j] share the same index space, both get shifted identically. In a system
where flux[i] and u[i] are 1D, the single shift applies to both.
"""
function _shift_array_indices_multidim(expr::SymbolicT, shifts::Vector{Int})
    all(iszero, shifts) && return expr
    if iscall(expr)
        f = operation(expr)
        args = arguments(expr)
        if f === getindex && length(args) >= 2
            # Shift each index dimension
            n_idx = length(args) - 1  # number of index arguments
            new_args = Any[args[1]]  # base array stays the same
            for k in 1:n_idx
                if k <= length(shifts) && shifts[k] != 0
                    push!(new_args, args[k+1] + shifts[k])
                else
                    push!(new_args, args[k+1])
                end
            end
            return new_args[1][new_args[2:end]...]
        elseif f isa Differential
            new_inner = _shift_array_indices_multidim(args[1], shifts)
            return f(new_inner)
        else
            new_args = [_shift_array_indices_multidim(a, shifts) for a in args]
            return SU.maketerm(SymbolicT, f, new_args, SU.metadata(expr))
        end
    end
    return expr
end

"""
Add scalar unknowns for a block ODE variable. Extracts the base array from
the representative and adds all its indexed elements.
"""
function _add_block_unknowns!(dvs::Vector, dvs_set::Set{SymbolicT}, block::MTKTearing.ArrayBlockInfo)
    rep_eq = block.representative_eq
    rep_lhs = unwrap(rep_eq.lhs)

    if MTKBase.isdiffeq(rep_eq)
        inner = arguments(rep_lhs)[1]  # u(t)[1]
        if iscall(inner) && operation(inner) === getindex
            base_arr = arguments(inner)[1]  # u(t)
            base_sh = SU.shape(base_arr)
            if SU.is_array_shape(base_sh)
                for idx in Iterators.product(base_sh...)
                    var = unwrap(base_arr[idx...])
                    if !(var in dvs_set)
                        push!(dvs, var)
                        push!(dvs_set, var)
                    end
                end
            end
        end
    end
end
