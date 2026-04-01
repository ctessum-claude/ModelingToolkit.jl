using Test
using ModelingToolkit
using ModelingToolkit: t, D
using SymbolicUtils
using DynamicQuantities
using OrdinaryDiffEqDefault
using SparseArrays

@testset "ArrayOp Block Tearing" begin

    @testset "Pure ODE" begin
        @variables (u(t))[1:5]
        lhs = @arrayop (i,) D(u[i]) i in 1:5
        rhs = @arrayop (i,) -u[i] i in 1:5
        dvs = [u[i] for i in 1:5]
        @named sys = System([lhs ~ rhs], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        @test length(unknowns(compiled)) == 5

        prob = ODEProblem(compiled, [compiled.u[i] => Float64(i) for i in 1:5], (0.0, 1.0))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success
        for i in 1:5
            @test sol[compiled.u[i]][end] ≈ Float64(i) * exp(-1.0) rtol=1e-6
        end
    end

    @testset "Mixed ODE + algebraic" begin
        @variables (u(t))[1:5] (v(t))[1:5]
        lhso = @arrayop (i,) D(u[i]) i in 1:5
        rhso = @arrayop (i,) v[i] i in 1:5
        lhsa = @arrayop (i,) v[i] i in 1:5
        rhsa = @arrayop (i,) -u[i] i in 1:5
        dvs = [[u[i] for i in 1:5]; [v[i] for i in 1:5]]
        @named sys = System([lhso ~ rhso, lhsa ~ rhsa], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        @test length(unknowns(compiled)) == 5
        @test length(observed(compiled)) >= 5  # v should be observed

        prob = ODEProblem(compiled, [compiled.u[i] => Float64(i) for i in 1:5], (0.0, 1.0))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success
        for i in 1:5
            @test sol[compiled.u[i]][end] ≈ Float64(i) * exp(-1.0) rtol=1e-6
        end
    end

    @testset "1D stencil (diffusion)" begin
        N = 10
        @variables (u(t))[1:N]
        lint = @arrayop (i,) D(u[i]) i in 2:(N-1)
        rint = @arrayop (i,) u[i-1] - 2u[i] + u[i+1] i in 2:(N-1)
        bcs = [D(u[1]) ~ u[2] - u[1], D(u[N]) ~ u[N-1] - u[N]]
        dvs = [u[i] for i in 1:N]
        @named sys = System([lint ~ rint; bcs], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        @test length(unknowns(compiled)) == N

        prob = ODEProblem(compiled, [compiled.u[i] => (i == 5 ? 1.0 : 0.0) for i in 1:N], (0.0, 0.5))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success

        # Compare with scalar version
        scalar_eqs = [D(u[1]) ~ u[2] - u[1]]
        for i in 2:(N-1)
            push!(scalar_eqs, D(u[i]) ~ u[i-1] - 2u[i] + u[i+1])
        end
        push!(scalar_eqs, D(u[N]) ~ u[N-1] - u[N])
        @named ss = System(scalar_eqs, t; checks=false)
        sc = mtkcompile(ss)
        sp = ODEProblem(sc, [sc.u[i] => (i == 5 ? 1.0 : 0.0) for i in 1:N], (0.0, 0.5))
        ssol = solve(sp)
        for i in 1:N
            @test sol[compiled.u[i]][end] ≈ ssol[sc.u[i]][end] rtol=1e-6
        end
    end

    @testset "WENO-like wide stencil" begin
        N = 20
        @variables (u(t))[1:N] (flux(t))[1:N]
        lhs_flux = @arrayop (i,) flux[i] i in 3:(N-3)
        rhs_flux = @arrayop (i,) u[i-2] - 5u[i-1] + 10u[i] - 10u[i+1] + 5u[i+2] - u[i+3] i in 3:(N-3)
        lhs_ode = @arrayop (i,) D(u[i]) i in 4:(N-3)
        rhs_ode = @arrayop (i,) -(flux[i] - flux[i-1]) i in 4:(N-3)
        bcs = [D(u[i]) ~ 0.0 for i in [1,2,3,N-2,N-1,N]]
        dvs = [[u[i] for i in 1:N]; [flux[i] for i in 1:N]]
        @named sys = System([lhs_flux ~ rhs_flux, lhs_ode ~ rhs_ode, bcs...], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        @test length(unknowns(compiled)) == N
        # flux should be observed
        has_flux_obs = any(observed(compiled)) do eq
            occursin("flux", string(eq.lhs))
        end
        @test has_flux_obs

        prob = ODEProblem(compiled, [compiled.u[i] => 0.0 for i in 1:N], (0.0, 0.01);
            build_initializeprob=false)
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "Jacobian sparsity (banded, not dense)" begin
        N = 10
        @variables (u(t))[1:N]
        lint = @arrayop (i,) D(u[i]) i in 2:(N-1)
        rint = @arrayop (i,) u[i-1] - 2u[i] + u[i+1] i in 2:(N-1)
        bcs = [D(u[1]) ~ u[2] - u[1], D(u[N]) ~ u[N-1] - u[N]]
        dvs = [u[i] for i in 1:N]
        @named sys = System([lint ~ rint; bcs], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        sp = ModelingToolkitBase.jacobian_sparsity(compiled)
        @test size(sp) == (N, N)
        @test nnz(sp) < N * N  # Not dense
        @test nnz(sp) <= 3 * N  # At most tridiagonal
    end

    # MOL integration tests are in MethodOfLines.jl's own test suite.
    # Run them manually via:
    #   cd MethodOfLines.jl && julia --project -e 'ENV["GROUP"]="ArrayDisc"; using Pkg; Pkg.test()'

end
