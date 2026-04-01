# MWE: ArrayDiscretization fails with "Could not infer range of output index 1"
# for any 2D PDE discretized with MethodOfLines ArrayDiscretization.
#
# Tested with:
#   - ctessum-claude/ModelingToolkit.jl branch arrayop-block-tearing
#   - ctessum-claude/StateSelection.jl branch arrayop-block-tearing
#   - SciML/MethodOfLines.jl PR #531 (array-discretization-phase1)

using ModelingToolkit
using ModelingToolkit: t, D
using MethodOfLines, DomainSets, DynamicQuantities
using Symbolics

@parameters x [unit = u"m"]
@parameters y [unit = u"m"]
@parameters D_coeff = 0.01 [unit = u"m^2/s"]
@variables u(..) [unit = u"m"]
@constants u_ref = 1.0, [unit = u"m"]

Dxx = Differential(x) ∘ Differential(x)
Dyy = Differential(y) ∘ Differential(y)

eq = [D(u(t, x, y)) ~ D_coeff * (Dxx(u(t, x, y)) + Dyy(u(t, x, y)))]

domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0),
           y ∈ Interval(0.0, 1.0)]

bcs = [
    u(0.0, x, y) ~ u_ref * sin(3.14159 * x / 1.0),
    u(t, 0.0, y) ~ 0.0 * u_ref,
    u(t, 1.0, y) ~ 0.0 * u_ref,
    Differential(y)(u(t, x, 0.0)) ~ 0.0,
    Differential(y)(u(t, x, 1.0)) ~ 0.0,
]

@named pde = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)], [D_coeff, u_ref])
pde.initial_conditions[Symbolics.unwrap(D_coeff)] = 0.01
pde.initial_conditions[Symbolics.unwrap(u_ref)] = 1.0

dx = 0.1
disc = MOLFiniteDifference([x => dx, y => dx], t;
    discretization_strategy = MethodOfLines.ArrayDiscretization())

println("Discretizing 2D diffusion with ArrayDiscretization...")
flush(stdout)
try
    prob = MethodOfLines.discretize(pde, disc; checks = false, simplify = false)
    println("OK: $(length(prob.u0)) unknowns")
catch e
    println("FAIL: $(sprint(showerror, e))")
    println("\nStacktrace:")
    for (i, frame) in enumerate(catch_backtrace() |> stacktrace)
        i > 15 && break
        println("  [$i] $(frame.func) at $(frame.file):$(frame.line)")
    end
end
