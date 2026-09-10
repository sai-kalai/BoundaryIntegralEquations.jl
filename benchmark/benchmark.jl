using JLD2
using StaticArrays

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

n = 200
Γ_dense = DiscreteClosedCurve(n, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n)
ys = range(ymin, ymax, length=n)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

result = run_all_simulations(
    x_dense;
    # n_vals=[400, 800,],
    # fd_acc_vals=[32,],
    # cutoff_vals=[
    #     0.0,
    #     # 0.01, 0.05, 0.1
    # ],
    # bc_types=[Neumann,],
    # approach_types=[Indirect],
    benchmark_kwargs=(; samples=100)
)

const F = "data/benchmark.jld2"

save_object(F, result)

@info "result = "
display(result)

@info "`result` saved to `$F`"

