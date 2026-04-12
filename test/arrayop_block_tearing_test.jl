using Test
using ModelingToolkit
using ModelingToolkit: t, D
using SymbolicUtils
using SymbolicIndexingInterface
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
        # v is eliminated via pre-substitution (O(M) — not stored as N scalar observed)

        prob = ODEProblem(compiled, [compiled.u[i] => Float64(i) for i in 1:5], (0.0, 1.0))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success
        for i in 1:5
            @test sol[compiled.u[i]][end] ≈ Float64(i) * exp(-1.0) rtol=1e-6
        end

        # Verify block-level observed access: sol[compiled.v] returns the whole block
        # as Vector{Vector{Float64}} (one entry per timestep, each an array of 5 values)
        v_vals = sol[compiled.v]
        for i in 1:5
            @test v_vals[end][i] ≈ -Float64(i) * exp(-1.0) rtol=1e-6
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
        # flux is eliminated via pre-substitution (O(M) — not in observed)

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

    @testset "Rearranged algebraic eq (expr ~ 0 form)" begin
        # Regression test: when the algebraic equation is in rearranged form
        # (e.g., -1 - 0.5sin(u[i]) + v[i] ~ 0 instead of v[i] ~ 1 + 0.5sin(u[i])),
        # the block tearing must still correctly substitute and solve.
        # This is the form MethodOfLines produces for PDE algebraic equations.
        N = 5
        @variables (u(t))[1:N] (v(t))[1:N]

        # ODE: D(u[i]) ~ v[i] (uses the algebraic variable)
        lhso = @arrayop (i,) D(u[i]) i in 1:N
        rhso = @arrayop (i,) v[i] i in 1:N

        # Algebraic in REARRANGED form: (-1 - 0.5sin(u[i]) + v[i]) ~ (v[i] - v[i])
        # The RHS simplifies to 0. This tests the case where the LHS is NOT a clean
        # getindex but a sum expression containing the algebraic variable.
        lhsa = @arrayop (i,) -1.0 - 0.5*sin(u[i]) + v[i] i in 1:N
        rhsa = @arrayop (i,) v[i] - v[i] i in 1:N

        dvs = [[u[i] for i in 1:N]; [v[i] for i in 1:N]]
        @named sys = System([lhso ~ rhso, lhsa ~ rhsa], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        # v should be eliminated — only u unknowns remain
        @test length(unknowns(compiled)) == N

        # No v references in compiled equations (v is fully substituted)
        for eq in equations(compiled)
            eq_str = string(eq)
            @test !occursin("v(t)", eq_str)
        end

        # The ODE should solve successfully with v inlined
        prob = ODEProblem(compiled, [compiled.u[i] => Float64(i) for i in 1:N], (0.0, 1.0))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "Block-observed indexed access (sol[compiled.v[i]])" begin
        @variables (u(t))[1:5] (v(t))[1:5]
        lhso = @arrayop (i,) D(u[i]) i in 1:5
        rhso = @arrayop (i,) v[i] i in 1:5
        lhsa = @arrayop (i,) v[i] i in 1:5
        rhsa = @arrayop (i,) -u[i] i in 1:5
        dvs = [[u[i] for i in 1:5]; [v[i] for i in 1:5]]
        @named sys = System([lhso ~ rhso, lhsa ~ rhsa], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        prob = ODEProblem(compiled, [compiled.u[i] => Float64(i) for i in 1:5], (0.0, 1.0))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success

        # Test indexed access: sol[compiled.v[i]] for each element
        for i in 1:5
            @test sol[compiled.v[i]][end] ≈ -Float64(i) * exp(-1.0) rtol=1e-6
        end
    end

    @testset "2D ArrayOp (nested ForLoop codegen)" begin
        M, N_dim = 4, 3
        @variables (u(t))[1:M, 1:N_dim]
        lhs = @arrayop (i, j) D(u[i, j]) i in 1:M, j in 1:N_dim
        rhs = @arrayop (i, j) -u[i, j] i in 1:M, j in 1:N_dim
        dvs = [u[i, j] for i in 1:M for j in 1:N_dim]
        @named sys = System([lhs ~ rhs], t, dvs, []; checks=false)
        compiled = mtkcompile(sys)

        @test length(unknowns(compiled)) == M * N_dim

        u0 = [compiled.u[i, j] => Float64(i + j) for i in 1:M for j in 1:N_dim]
        prob = ODEProblem(compiled, u0, (0.0, 1.0))
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success

        # Verify analytical solution: u[i,j](t) = (i+j) * exp(-t)
        for i in 1:M, j in 1:N_dim
            @test sol[compiled.u[i, j]][end] ≈ Float64(i + j) * exp(-1.0) rtol=1e-6
        end
    end

    # MOL integration tests are in MethodOfLines.jl's own test suite.
    # Run them manually via:
    #   cd MethodOfLines.jl && julia --project -e 'ENV["GROUP"]="ArrayDisc"; using Pkg; Pkg.test()'

end
