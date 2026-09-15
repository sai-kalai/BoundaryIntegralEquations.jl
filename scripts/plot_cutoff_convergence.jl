
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
const FILE = "benchmark-scp10"
const DATAFILE = joinpath("data", FILE * ".jld2")
result = load_object(DATAFILE)
@info "loaded `result` from $DATAFILE"

###########
# Make figure
###########
w = 1200
fig = Figure(
    size=(w, w*9÷16)
)
ax = Axis(
    fig[1, 1][1, 1],
    title="BVP Solution",
    xlabel="n",
    ylabel="L∞-error",
    yscale=log10,
    # xscale=log10,
    xticks=LinearTicks(5),
)
ax2 = Axis(
    fig[1, 1][1, 2],
    title="Cauchy Data",
    xlabel="n",
    ylabel="L∞-error",
    yscale=log10,
    xscale=log10,
    xticks=LinearTicks(5),
)
# ylims!(ax, (1e-16, 1e+2))
# ylims!(ax2, (1e-16, 1e+2))

result2 = deepcopy(result)
###########
# Filter data
###########
filter!(result, (k) -> begin
    if !(k.solution_t <: BVPSolution)
        return false
    end
    if !(order(k.correction) in [32])
        return false
    end
    # if !(cutoff(k.evalmethod) in [
    #     0.0,
    #     0.01, 0.05, 0.1
    # ])
    #     return false
    # end
    if !(k.approach_t <: Indirect)
        return false
    end
    if !(k.bdrycond_t <: Dirichlet)
        return false
    end
    return true
end)
filter!(result2, (k) -> begin
    if !(k.solution_t <: BDPSolution)
        return false
    end
    if !(order(k.correction) in [32,])
        return false
    end
    # if !(cutoff(k.evalmethod) in [
    #     0.0,
    #     # 0.01, 0.05, 0.1
    # ])
    #     return false
    # end
    if !(k.approach_t <: Indirect)
        return false
    end
    if !(k.bdrycond_t <: Dirichlet)
        return false
    end
    return true
end)

###########
# Add plots
###########


ks = result.solutions |> keys |> collect
@show ks
sorted_perm = ks |> sortperm
@show sorted_perm

deltas = [cutoff(k.evalmethod) for k in ks[sorted_perm]]

@show deltas
# collect error for all n values
all_errs = [errors(k, result, group) for (k, group) in result.solutions]
@show all_errs


# one line per each n, x values for deltas
for i in sorted_perm

    key = ks[i]

    group = result.solutions[key]

    @show key
    ns = [numpoints(s) for s in solutions(group)]

    @show ns
    @show i, ns[i]

    errs = [e[i] for e in all_errs]

    @show deltas
    @show errs

    kwargs = scatterlines_common_kwargs(key, result)

    scatterlines!(
        ax,
        deltas,
        errs,
        ;
        color=kwargs.colormap[i],
        marker=MARKER_LABELS[i],
        strokecolor=kwargs.colormap[i],
        label="n = $(ns[i])",
        kwargs...,
    )
end

l = axislegend(ax; position=:lb)

fig[1, 2][1, 1] = l

for (i, (key, group)) in enumerate(result2.solutions)
    kwargs = scatterlines_common_kwargs(key, result)
    @show key
    sort!(group, by=swm -> numpoints(swm[1]))
    sols = solutions(group)
    ns = [numpoints(s) for s in sols]
    errs = errors(key, result2, group)
    scatterlines!(
        ax2,
        ns,
        errs,
        ;
        color=kwargs.colormap[i],
        marker=MARKER_LABELS[i],
        strokecolor=kwargs.colormap[i],
        label="δ=$(cutoff(key.evalmethod)), $(key.correction)",
        kwargs...,
    )
end


# colorbars and legend
# c1, c2, c3 = scatterlines_common_colorbars!(fig, result)
# legend = scatterlines_common_legend!(fig, result)
# fig[0, 1] = legend
# fig[1, 2][1, 1] = c1
# fig[1, 2][1, 2] = c2
# fig[1, 2][1, 3] = c3


###########
# Save and display plot
###########
const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", "cutoff_convergence_" * FILE * ".pdf")
else
    joinpath("figures", "cutoff_convergence_" * FILE * ".png")
end
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
