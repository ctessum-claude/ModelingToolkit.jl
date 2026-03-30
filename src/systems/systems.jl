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
            result = _expand_arrayop_blocks(result, block_eqs)
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

    # Expand observed equations
    for eq in compiled_obs
        block = _find_matching_block_obs(eq, block_eqs)
        if block !== nothing
            expanded = _expand_block_eq(eq, block)
            append!(new_obs, expanded)
        else
            push!(new_obs, eq)
        end
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
Expand a representative scalar equation (from tearing) into all N scalar equations
by shifting array indices. The compiled representative has all tearing substitutions
applied; we replicate it for each element index.
"""
function _expand_block_eq(compiled_rep::Equation, block::MTKTearing.ArrayBlockInfo)
    n = block.scalar_count

    # Determine the base index used in the representative
    rep_lhs = unwrap(compiled_rep.lhs)
    rep_base_idx = _extract_first_index(rep_lhs)
    rep_base_idx === nothing && return [compiled_rep]

    # First element is the representative itself (shift=0)
    result = Equation[compiled_rep]
    # Remaining n-1 elements are shifted by +1, +2, ..., +(n-1)
    for shift in 1:(n - 1)
        new_lhs = _shift_all_array_indices(unwrap(compiled_rep.lhs), shift)
        new_rhs = _shift_all_array_indices(unwrap(compiled_rep.rhs), shift)
        push!(result, new_lhs ~ new_rhs)
    end
    return result
end

"""
Extract the integer index from the first getindex in an expression.
E.g., D(u(t)[3]) → 3, v(t)[1] → 1
"""
function _extract_first_index(expr::SymbolicT)
    if iscall(expr)
        f = operation(expr)
        args = arguments(expr)
        if f === getindex && length(args) >= 2
            idx = args[2]
            return SU.isconst(idx) ? Int(SU.unwrap_const(idx)) : nothing
        end
        for a in args
            result = _extract_first_index(a)
            result !== nothing && return result
        end
    end
    return nothing
end

"""
Shift all array getindex references in an expression by `shift`.
E.g., _shift_all_array_indices(D(u[1]) + v[1]^2, 2) → D(u[3]) + v[3]^2
"""
function _shift_all_array_indices(expr::SymbolicT, shift::Int)
    shift == 0 && return expr
    if iscall(expr)
        f = operation(expr)
        args = arguments(expr)
        if f === getindex && length(args) >= 2
            new_idx = args[2] + shift
            return args[1][new_idx]
        elseif f isa Differential
            new_inner = _shift_all_array_indices(args[1], shift)
            return f(new_inner)
        else
            new_args = [_shift_all_array_indices(a, shift) for a in args]
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
