using Test
using ModelingToolkit
using ModelingToolkit: t, D
using SymbolicUtils
using SymbolicUtils: BasicSymbolic, Const, scalarize, shape, isarrayop

# Helper to count effective scalar equations (accounting for ArrayOp)
function count_effective_equations(sys)
    n = 0
    for eq in equations(sys)
        lhs = unwrap(eq.lhs)
        rhs = unwrap(eq.rhs)
        sh = SymbolicUtils.shape(lhs)
        if SymbolicUtils.is_array_shape(sh)
            n += prod(length.(sh))
        else
            sh_r = SymbolicUtils.shape(rhs)
            if SymbolicUtils.is_array_shape(sh_r)
                n += prod(length.(sh_r))
            else
                n += 1
            end
        end
    end
    return n
end

# Test 1: Pure ODE with 1D stencil ArrayOp
@testset "Pure ODE stencil" begin
    N = 10
    @variables u(t)[1:N]

    # Interior: D(u[i]) ~ u[i-1] - 2u[i] + u[i+1]  (diffusion stencil)
    interior = @arrayop (i,) D(u(t))[i] ~ u(t)[i - 1] - 2u(t)[i] + u(t)[i + 1] i in 2:(N-1)
    # Boundaries
    bcs = [D(u(t)[1]) ~ u(t)[2] - u(t)[1],
           D(u(t)[N]) ~ u(t)[N-1] - u(t)[N]]

    @named sys = System([interior; bcs], t)
    compiled = mtkcompile(sys)

    # Should have N unknowns
    @test length(unknowns(compiled)) == N

    # The interior equation should remain as ArrayOp (not scalarized to N-2 equations)
    array_eqs = filter(equations(compiled)) do eq
        lhs = unwrap(eq.lhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs))
    end
    @test length(array_eqs) >= 1  # At least one ArrayOp equation preserved

    # Total effective equation count should equal N
    @test count_effective_equations(compiled) == N
end

# Test 2: Algebraic equation with diagonal solve (v[i] ~ u[i]^2)
@testset "Algebraic diagonal solve" begin
    N = 5
    @variables u(t)[1:N] v(t)[1:N]

    # ODE for u using v
    ode = @arrayop (i,) D(u(t))[i] ~ v(t)[i] i in 1:N
    # Algebraic: v[i] ~ u[i]^2 (diagonal incidence for v)
    alg = @arrayop (i,) v(t)[i] ~ u(t)[i]^2 i in 1:N

    @named sys = System([ode, alg], t)
    compiled = mtkcompile(sys)

    # v should be eliminated (moved to observed), u should be the unknown
    @test length(unknowns(compiled)) == N

    # v should appear in observed equations
    obs_vars = Set(unwrap(eq.lhs) for eq in observed(compiled))
    # Check that v's scalar elements are observed (or v as array)
    has_v_observed = any(obs_vars) do v_obs
        str = string(v_obs)
        occursin("v", str)
    end
    @test has_v_observed
end

# Test 3: Mixed ODE + algebraic
@testset "Mixed ODE + algebraic" begin
    N = 8
    @variables u(t)[1:N] v(t)[1:N]

    # ODE: D(u[i]) ~ u[i-1] - 2u[i] + u[i+1] + v[i]
    ode = @arrayop (i,) D(u(t))[i] ~ u(t)[i - 1] - 2u(t)[i] + u(t)[i + 1] + v(t)[i] i in 2:(N-1)
    # Algebraic: v[i] ~ u[i]^2
    alg = @arrayop (i,) v(t)[i] ~ u(t)[i]^2 i in 1:N
    # Boundary ODEs
    bcs = [D(u(t)[1]) ~ u(t)[2] - u(t)[1] + v(t)[1],
           D(u(t)[N]) ~ u(t)[N-1] - u(t)[N] + v(t)[N]]

    @named sys = System([ode, alg, bcs...], t)
    compiled = mtkcompile(sys)

    # v eliminated, u remains
    @test length(unknowns(compiled)) == N

    # System should be solvable (create and solve an ODE problem)
    using OrdinaryDiffEqDefault
    u0 = zeros(N)
    u0[N÷2] = 1.0
    prob = ODEProblem(compiled, [unknowns(compiled) .=> u0;], (0.0, 0.1))
    sol = solve(prob)
    @test sol.retcode == ReturnCode.Success
end

# Test 4: Non-trivially ordered algebraic (0 ~ v[i] - u[i]^2)
@testset "Non-trivially ordered algebraic" begin
    N = 5
    @variables u(t)[1:N] v(t)[1:N]

    ode = @arrayop (i,) D(u(t))[i] ~ v(t)[i] i in 1:N
    # v NOT on lhs: 0 ~ v[i] - u[i]^2
    alg = @arrayop (i,) 0 ~ v(t)[i] - u(t)[i]^2 i in 1:N

    @named sys = System([ode, alg], t)
    compiled = mtkcompile(sys)

    # v should still be solvable and eliminated
    @test length(unknowns(compiled)) == N
end

# Test 5: WENO-like wide stencil
@testset "WENO-like wide stencil" begin
    N = 20
    @variables u(t)[1:N] flux(t)[1:N]

    # WENO5 uses a 6-point stencil: u[i-2:i+3]
    # flux has diagonal incidence, u has wide banded incidence
    flux_eq = @arrayop (i,) flux(t)[i] ~ u(t)[i - 2] - 5u(t)[i - 1] + 10u(t)[i] - 10u(t)[i + 1] + 5u(t)[i + 2] - u(t)[i + 3] i in 3:(N-3)

    # ODE: D(u[i]) ~ -(flux[i] - flux[i-1])
    ode = @arrayop (i,) D(u(t))[i] ~ -(flux(t)[i] - flux(t)[i - 1]) i in 4:(N-3)

    # Simple boundary ODEs
    bcs = [D(u(t)[i]) ~ 0.0 for i in [1, 2, 3, N-2, N-1, N]]

    @named sys = System([flux_eq, ode, bcs...], t)
    compiled = mtkcompile(sys)

    # flux should be eliminated (observed), u remains
    @test length(unknowns(compiled)) == N

    # flux should appear in observed
    obs_vars = Set(string(unwrap(eq.lhs)) for eq in observed(compiled))
    has_flux_observed = any(s -> occursin("flux", s), obs_vars)
    @test has_flux_observed
end

# Test 6: Verify block tearing produces same numerical results as scalar
@testset "Numerical equivalence" begin
    N = 6
    @variables u(t)[1:N]

    # Build both scalar and ArrayOp versions of the same system
    # Scalar version
    scalar_eqs = [D(u(t)[1]) ~ u(t)[2] - u(t)[1]]
    for i in 2:(N-1)
        push!(scalar_eqs, D(u(t)[i]) ~ u(t)[i-1] - 2u(t)[i] + u(t)[i+1])
    end
    push!(scalar_eqs, D(u(t)[N]) ~ u(t)[N-1] - u(t)[N])

    @named scalar_sys = System(scalar_eqs, t)
    scalar_compiled = mtkcompile(scalar_sys)

    # ArrayOp version
    interior = @arrayop (i,) D(u(t))[i] ~ u(t)[i - 1] - 2u(t)[i] + u(t)[i + 1] i in 2:(N-1)
    bcs = [D(u(t)[1]) ~ u(t)[2] - u(t)[1],
           D(u(t)[N]) ~ u(t)[N-1] - u(t)[N]]
    @named array_sys = System([interior; bcs], t)
    array_compiled = mtkcompile(array_sys)

    # Both should produce same solution
    using OrdinaryDiffEqDefault
    u0_vals = zeros(N)
    u0_vals[N÷2] = 1.0

    prob_scalar = ODEProblem(scalar_compiled,
        [unknowns(scalar_compiled) .=> u0_vals;], (0.0, 0.5))
    prob_array = ODEProblem(array_compiled,
        [unknowns(array_compiled) .=> u0_vals;], (0.0, 0.5))

    sol_scalar = solve(prob_scalar)
    sol_array = solve(prob_array)

    @test sol_scalar.retcode == ReturnCode.Success
    @test sol_array.retcode == ReturnCode.Success

    # Compare solutions at final time
    for i in 1:N
        @test sol_scalar[u(t)[i]][end] ≈ sol_array[u(t)[i]][end] rtol=1e-8
    end
end
