# benchmark the performance of the solvers when a dense cartesian background
# grid is used for evaluation to study the scaling when many points are
# requested for evaluation

using JLD2
using StaticArrays
using BenchmarkTools

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

# define dense cartesian background grid
n = parse(Int, ARGS[1])
# n = 200
Γ_dense = DiscreteClosedCurve(n, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n)
ys = range(ymin, ymax, length=n)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

# run simulations to collect error results
convergence_result, suite = run_all_simulations(
    x_dense;
    n_vals=[50],
    fd_acc_vals=[32,],
    cutoff_vals=[
        0.0,
        # 0.01,
        # 0.05,
        # 0.1
    ],
    # bc_types=[Neumann,],
    # approach_types=[Indirect],
    return_benchmark_suite=true
)

# setup and run benchmark on the above solver configurations
benchmark_result = run(suite, verbose=true, samples=10)

# save error and timing data
convergence_file = joinpath(
    "data", "convergence_" * basename(@__FILE__) * ".jld2")
benchmark_file = joinpath(
    "data", "benchmark_" * basename(@__FILE__) * ".json")


save_object(convergence_file, convergence_result)
@info "convergence_result = "
display(convergence_result)
@info "`convergence_result` saved to `$convergence_file`"

BenchmarkTools.save(benchmark_file, benchmark_result)
@info "benchmark_result = "
display(benchmark_result)
@info "`benchmark_result` saved to `$benchmark_file`"

