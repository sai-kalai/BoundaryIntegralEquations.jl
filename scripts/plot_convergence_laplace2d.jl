# Plot the convergence behavior of the different methods
#
# First acquire the convergence data by running the script
#
#```bash
#$ julia --project=. test/convergence/laplace_2d.jl
#```
#
#or from the REPL
#
#```julia
#julia> include("test/convergence/laplace_2d.jl")
#```

using JLD2
using GLMakie

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("plot_utils.jl")

###########
# Read data
###########
# const FILE = "convergence_laplace_2d"
const FILE = "benchmark"
const DATAFILE = joinpath("data", FILE * ".jld2")
result = load_object(DATAFILE)
@info "loaded `result` from $DATAFILE"

###########
# Make figure
###########
fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel="n",
    ylabel="L∞-error",
    yscale=log10,
    xscale=log10,
    xticks=LinearTicks(5),
)
ylims!(ax, (1e-17, 1e+1))

###########
# Filter data
###########
filter!(result, (k) -> begin
    if !(k.solution_t <: BVPSolution)
        return false
    end
    # if !(order(k.correction) in [16, 32, Inf])
    #     return false
    # end
    # if !(cutoff(k.evalmethod) in [0.0, 0.01, 0.05, 0.1])
    #     return false
    # end
    # if !(k.approach_t <: Indirect)
    #     return false
    # end
    # if !(k.bdrycond_t <: Dirichlet)
    #     return false
    # end
    return true
end)

###########
# Add plots
###########
for (key, group) in result.solutions
    @show key
    sort!(group, by=swm -> numpoints(swm[1]))
    sols = solutions(group)
    ns = [numpoints(s) for s in sols]
    errs = errors(key, result, group)
    scatterlines!(
        ax,
        ns,
        errs,
        ;
        scatterlines_common_kwargs(key, result)...,
    )
end
# trendlines
conv_style = (; linestyle=:dashdotdot, linewidth=3)
α = 0.1
lines!(ax,
    result.n_vals, # use last iteration for getting ns
    exp.(-α .* result.n_vals),
    ;
    color=:grey,
    conv_style...
)

# colorbars and legend
c1, c2, c3 = scatterlines_common_colorbars!(fig, result)
legend = scatterlines_common_legend!(fig, result)
fig[0, 1] = legend
fig[1, 2][1, 1] = c1
fig[1, 2][1, 2] = c2
fig[1, 2][1, 3] = c3


###########
# Save and display plot
###########
const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", FILE * ".pdf")
else
    joinpath("figures", FILE * ".png")
end
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
