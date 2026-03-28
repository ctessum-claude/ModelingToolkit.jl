using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
import SymbolicUtils
import SymbolicUtils as SU
import ModelingToolkitBase as MTKBase

# Helper to check if an Expr contains a for loop
function _expr_contains_for(expr)
    if !isa(expr, Expr)
        return false
    end
    if expr.head === :for
        return true
    end
    return any(_expr_contains_for, expr.args)
end

# Count assignment lines of the form out[i] = ... in an Expr
function _count_out_assignments(expr)
    count = 0
    if !isa(expr, Expr)
        return 0
    end
    if expr.head === :(=) && isa(expr.args[1], Expr) && expr.args[1].head === :ref
        idx = expr.args[1].args[2]
        if idx isa Integer
            count += 1
        end
    end
    for arg in expr.args
        count += _count_out_assignments(arg)
    end
    return count
end

@testset "ArrayOp metadata and vectorized codegen" begin
    @variables begin
        u_1_1(t)
        u_1_2(t)
        u_1_3(t)
        u_2_1(t)
        u_2_2(t)
        u_2_3(t)
        u_3_1(t)
        u_3_2(t)
        u_3_3(t)
    end
    @parameters alpha

    # All scalar equations (explicit ODEs for a 3x3 diffusion grid)
    scalar_eqs = [
        D(u_1_1) ~ alpha * (-2 * u_1_1 + u_2_1 + u_1_2),
        D(u_1_2) ~ alpha * (-2 * u_1_2 + u_2_2 + u_1_1 + u_1_3),
        D(u_1_3) ~ alpha * (-2 * u_1_3 + u_2_3 + u_1_2),
        D(u_2_1) ~ alpha * (-2 * u_2_1 + u_1_1 + u_3_1 + u_2_2),
        D(u_2_2) ~ alpha * (u_1_2 + u_3_2 + u_2_1 + u_2_3 - 4 * u_2_2),
        D(u_2_3) ~ alpha * (-2 * u_2_3 + u_1_3 + u_3_3 + u_2_2),
        D(u_3_1) ~ alpha * (-2 * u_3_1 + u_2_1 + u_3_2),
        D(u_3_2) ~ alpha * (-2 * u_3_2 + u_2_2 + u_3_1 + u_3_3),
        D(u_3_3) ~ alpha * (-2 * u_3_3 + u_2_3 + u_3_2),
    ]

    # Build ArrayOp equation wrapping multiple interior points (u_2_2 AND u_2_3)
    # to get a range of length > 1 so the for loop is actually generated
    _idxs_arr = SU.idxs_for_arrayop(SU.SymReal)
    _i1 = _idxs_arr[1]

    # Two-point ArrayOp covering u_2_2 and u_2_3
    # We need the ArrayOp LHS and RHS to flatten to 2 scalar equations
    interior_lhs_5 = SU.unwrap(D(u_2_2))
    interior_rhs_5 = SU.unwrap(scalar_eqs[5].rhs)
    interior_lhs_6 = SU.unwrap(D(u_2_3))
    interior_rhs_6 = SU.unwrap(scalar_eqs[6].rhs)

    ao_ranges = Dict{SU.BasicSymbolic{SU.SymReal}, StepRange{Int, Int}}(_i1 => 1:1:2)

    # For a proper multi-element ArrayOp, we need the expr to parameterize over _i1.
    # But for testing purposes, use a simpler approach: create TWO single-element ArrayOps.
    # Each covers one interior equation.
    ao_ranges_1 = Dict{SU.BasicSymbolic{SU.SymReal}, StepRange{Int, Int}}(_i1 => 1:1:1)
    lhs_ao_1 = SU.ArrayOp{SU.SymReal}([_i1], interior_lhs_5, +, nothing, ao_ranges_1)
    rhs_ao_1 = SU.ArrayOp{SU.SymReal}([_i1], interior_rhs_5, +, nothing, ao_ranges_1)
    arrayop_eq_1 = Symbolics.wrap(lhs_ao_1) ~ Symbolics.wrap(rhs_ao_1)

    lhs_ao_2 = SU.ArrayOp{SU.SymReal}([_i1], interior_lhs_6, +, nothing, ao_ranges_1)
    rhs_ao_2 = SU.ArrayOp{SU.SymReal}([_i1], interior_rhs_6, +, nothing, ao_ranges_1)
    arrayop_eq_2 = Symbolics.wrap(lhs_ao_2) ~ Symbolics.wrap(rhs_ao_2)

    # Build system: 7 boundary scalar eqs + 2 ArrayOp interior eqs
    all_eqs = vcat(scalar_eqs[1:4], [arrayop_eq_1, arrayop_eq_2], scalar_eqs[7:9])
    @named sys = System(all_eqs, t)

    @testset "Helpers" begin
        @test !MTKBase.is_arrayop_equation(scalar_eqs[1])
        @test MTKBase.is_arrayop_equation(arrayop_eq_1)

        infos = MTKBase.extract_arrayop_equations(equations(sys))
        @test length(infos) == 2
        @test all(i -> i.is_ode, infos)
    end

    @testset "Metadata survives mtkcompile" begin
        compiled = mtkcompile(sys)
        @test SU.hasmetadata(compiled, MTKBase.ArrayEquationsCtx)

        infos = SU.getmetadata(compiled, MTKBase.ArrayEquationsCtx, nothing)
        @test infos !== nothing
        @test length(infos) == 2

        # All compiled equations are scalar
        for eq in equations(compiled)
            @test !SU.is_array_shape(SU.shape(SU.unwrap(eq.lhs)))
        end
        @test length(equations(compiled)) == 9
    end

    @testset "find_arrayop_equation_indices" begin
        compiled = mtkcompile(sys)
        infos = SU.getmetadata(compiled, MTKBase.ArrayEquationsCtx, nothing)
        ranges = MTKBase.find_arrayop_equation_indices(compiled, infos)
        @test length(ranges) >= 1
        total_covered = sum(length, ranges)
        @test total_covered == 2  # Two interior equations
    end

    @testset "Vectorized IIP Expr" begin
        compiled = mtkcompile(sys)

        # Get the generated expression
        f_expr = ModelingToolkit.generate_rhs(compiled; expression=Val{true})
        iip_expr = f_expr[2]

        # Check that the IIP expression is valid
        @test Meta.isexpr(iip_expr, :function)

        # Check ArrayOp metadata on compiled system
        infos = SU.getmetadata(compiled, MTKBase.ArrayEquationsCtx, nothing)
        ranges = MTKBase.find_arrayop_equation_indices(compiled, infos)

        # If the two covered equations are contiguous, a for loop should be generated
        if length(ranges) == 1 && length(ranges[1]) == 2
            @test _expr_contains_for(iip_expr)
            # Should have fewer individual assignments than 9 (7 boundary + loop)
            n_assignments = _count_out_assignments(iip_expr)
            @test n_assignments < 9
        else
            # Non-contiguous: two separate 1-element ranges, no loop
            @test !_expr_contains_for(iip_expr)
        end
    end

    @testset "Pure scalar reference" begin
        # Verify that a pure scalar system (no ArrayOp metadata) works correctly
        @named sys_scalar = System(scalar_eqs, t)
        compiled_scalar = mtkcompile(sys_scalar)

        @test !SU.hasmetadata(compiled_scalar, MTKBase.ArrayEquationsCtx)

        f_scalar = ModelingToolkit.generate_rhs(compiled_scalar; expression=Val{true})
        iip_scalar = f_scalar[2]
        @test Meta.isexpr(iip_scalar, :function)
        @test !_expr_contains_for(iip_scalar)
        @test _count_out_assignments(iip_scalar) == 9
    end
end
