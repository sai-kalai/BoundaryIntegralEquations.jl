# benchmark the performance of the solvers when a dense cartesian background
# grid is used for evaluation to study the scaling when many points are
# requested for evaluation

using JLD2
using StaticArrays
using BenchmarkTools

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

# define dense cartesian background grid
@show ARGS
mx = parse(Int, ARGS[1])
domain_type = ARGS[2]
@show mx
@show domain_type

if domain_type == "dense"
    Γ_dense = DiscreteClosedCurve(mx, starfish)
    xmin, xmax, ymin, ymax = extrema(Γ_dense)
    xs = range(xmin, xmax, length=mx)
    ys = range(ymin, ymax, length=mx)
    iter = Iterators.product(xs, ys)
    x = stack(((x, y),) -> SA[x, y], iter; dims=2)
elseif domain_type == "far"
    x = ball(0.01, mx)
end

# run simulations to collect error results
convergence_result, suite = run_all_simulations(
    x;
    # n_vals=[50, 60],
    # fd_acc_vals=[32,],
    # cutoff_vals=[
    #     # 0.0,
    #     # 0.01,
    #     # 0.05,
    #     # 0.1,
    #     # 1.0,
    # ],
    # bc_types=[Dirichlet,],
    # approach_types=[Indirect],
    return_benchmark_suite=true
)

m = size(x, 2)
id = "m$(m)_$domain_type"

# save error and timing data
convergence_file = joinpath(
    "data", "convergence_" * id * ".jld2")
save_object(convergence_file, convergence_result)
@info "convergence_result = "
display(convergence_result)
@info "`convergence_result` saved to `$convergence_file`"
# setup and run benchmark on the above solver configurations
benchmark_result = run(suite, verbose=true, samples=100)

benchmark_file = joinpath(
    "data", "benchmark_" * id * ".json")
BenchmarkTools.save(benchmark_file, benchmark_result)
@info "benchmark_result = "
display(benchmark_result)
@info "`benchmark_result` saved to `$benchmark_file`"

