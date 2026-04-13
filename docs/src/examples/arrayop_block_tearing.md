# ArrayOp Block Tearing for PDE-Discretized Systems

When a system has equations that repeat the same symbolic pattern across a large
grid — for example, a spatially-discretized PDE where every interior point satisfies
the same stencil — ModelingToolkit can compile and solve the system in time that
is **independent of the grid size** via *ArrayOp block tearing*.

This tutorial walks through a 1D heat equation to show:

 1. how to express a discretized PDE as `@arrayop` equations,
 2. what happens during `mtkcompile` (tearing operates on one representative per block,
    and code generation emits one `ForLoop` per block rather than `N` scalar assignments),
 3. how to access algebraic variables that are eliminated by block-level substitution,
 4. why the Jacobian sparsity remains banded rather than dense.

## The 1D heat equation

We discretize ``\partial_t u = \partial_{xx} u`` on ``x \in [0, 1]`` using a three-point
stencil with `N` grid points and Dirichlet boundary conditions at both ends.
The interior ODEs and the two boundary conditions can be written as three
`@arrayop` equations:

```@example arrayop_tearing
using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D

const N = 200
@variables (u(t))[1:N]

# Interior stencil: D(u[i]) = u[i-1] - 2u[i] + u[i+1] for i in 2:(N-1)
interior_lhs = @arrayop (i,) D(u[i]) i in 2:(N-1)
interior_rhs = @arrayop (i,) u[i-1] - 2u[i] + u[i+1] i in 2:(N-1)

# Dirichlet boundaries: u[1] and u[N] are held at zero via a large negative source
bc_left  = D(u[1]) ~ -u[1]
bc_right = D(u[N]) ~ -u[N]

dvs = [u[i] for i in 1:N]
@named sys = System([interior_lhs ~ interior_rhs, bc_left, bc_right], t, dvs, [];
                    checks = false)
compiled = mtkcompile(sys)
```

At this point `compiled` is a `System` with `N = 200` unknowns but only **3** live
equations: one block representative for the interior stencil plus the two scalar
boundary conditions. Block tearing stores the block metadata on the compiled
system and defers expansion to code-generation time.

```@example arrayop_tearing
length(unknowns(compiled)), length(equations(compiled))
```

## Solving and verifying

Building and solving the `ODEProblem` works exactly as for a scalar system —
there's no special API on the user side:

```@example arrayop_tearing
u0 = [compiled.u[i] => (i == N ÷ 2 ? 1.0 : 0.0) for i in 1:N]
prob = ODEProblem(compiled, u0, (0.0, 1.0))
sol  = solve(prob, Tsit5())
sol.retcode
```

```@example arrayop_tearing
using Plots
plot(0:(N - 1), sol[compiled.u][end]; xlabel = "grid index",
     ylabel = "u", label = "t = 1.0", lw = 2)
```

## Jacobian sparsity stays banded

The representative's stencil determines the Jacobian column offsets, and those
offsets are tiled across every element of the block. The result is a
tridiagonal pattern with `O(N)` nonzeros — not the dense `N×N` pattern you'd
get from naively computing the Jacobian symbolically:

```@example arrayop_tearing
using SparseArrays
sp = ModelingToolkitBase.jacobian_sparsity(compiled)
size(sp), nnz(sp)
```

```@example arrayop_tearing
# nnz should be O(N), not O(N^2)
nnz(sp) / N
```

## Block-observed variables

When an algebraic variable is defined by an `@arrayop` equation (e.g. a flux
computed from neighboring states), block tearing eliminates it by
pre-substitution. The variable does not appear in `unknowns(compiled)` but is
still accessible after solving — both as a whole array and by index — via the
representative equation shifted on demand.

Here's a toy example with `v[i] ~ -u[i]`:

```@example arrayop_tearing
@variables (x(t))[1:5] (y(t))[1:5]
ode_lhs = @arrayop (i,) D(x[i]) i in 1:5
ode_rhs = @arrayop (i,) y[i]    i in 1:5
alg_lhs = @arrayop (i,) y[i]    i in 1:5
alg_rhs = @arrayop (i,) -x[i]   i in 1:5

dvs = [[x[i] for i in 1:5]; [y[i] for i in 1:5]]
@named toy = System([ode_lhs ~ ode_rhs, alg_lhs ~ alg_rhs], t, dvs, [];
                    checks = false)
toy = mtkcompile(toy)

# `y` is eliminated — only `x` appears in unknowns
length(unknowns(toy))
```

```@example arrayop_tearing
prob = ODEProblem(toy, [toy.x[i] => Float64(i) for i in 1:5], (0.0, 1.0))
sol  = solve(prob, Tsit5())

# Indexed access to the block-observed variable works:
sol[toy.y[1]][end]
```

```@example arrayop_tearing
# And whole-block access returns the full time series:
typeof(sol[toy.y]), length(sol[toy.y])
```

## When to use `@arrayop`

Block tearing pays off when:

  - The same equation pattern repeats across a grid with ``N`` in the
    high hundreds or more.
  - The pattern is a *uniform stencil* (every element references neighbors at the
    same relative offsets). Non-uniform stencils or spatially-varying coefficients
    would produce a wrong banded pattern from the Jacobian sparsity tiler.
  - You want `mtkcompile` and `ODEProblem` construction times that are
    independent of ``N`` — for large grids this can turn minutes of
    Julia compilation into seconds.

Scalar systems and small array systems don't benefit much from the block path
and can use ordinary `@variables x(t)[1:N]` + scalar equations instead.

## See also

  - [`MethodOfLines.jl`](https://docs.sciml.ai/MethodOfLines/stable/) — uses
    `@arrayop` as its lowering target for the `ArrayDiscretization()` strategy,
    so user-facing PDE systems discretized via `MOLFiniteDifference` automatically
    flow through this path.
  - The `sparse_jacobians.md` example shows an alternative route for large
    systems where the source is a hand-written RHS function rather than a
    symbolic discretization.
