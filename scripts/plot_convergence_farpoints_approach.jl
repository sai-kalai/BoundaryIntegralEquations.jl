# plot convergence for far, i.e. "nice" points, using 4 combinations
# [Direct, Indirect] x [Dirichlet, Neumann]

using JLD2
using GLMakie

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("plot_utils.jl")

###########
# Read data
###########
const FILE = "convergence_laplace_2d"
const DATAFILE = joinpath("data", FILE * ".jld2")
result = load_object(DATAFILE)
@info "loaded `result` from $DATAFILE"

###########
# Make figure
###########
w = 1200
fig = Figure(
    title="",
    size=(w, w*9÷16)
)
axs = [Axis(
        fig[1, 1][1, 1],
        title="BVP Solution",
        xlabel="n",
        ylabel="L∞-error",
        yscale=log10,
        xscale=log10,
        xticks=LinearTicks(5),
    ),
    Axis(
        fig[1, 1][1, 2],
        title="Cauchy Data",
        xlabel="n",
        ylabel="L∞-error",
        yscale=log10,
        xscale=log10,
        xticks=LinearTicks(5),
    )]
ylims!(ax, (1e-16, 1e+2))
ylims!(ax2, (1e-16, 1e+2))

###########
# Filter data
###########
filter!(result, (k) -> begin
    if !(k.solution_t <: BVPSolution || k.solution_t <: BDPSolution)
        return false
    end
    if !(order(k.correction) in [32,])
        return false
    end
    if !(cutoff(k.evalmethod) in [
        0.0,
        # 0.01, 0.05, 0.1
    ])
        return false
    end
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
visuals = String[]

for key in sort(collect(keys(result.solutions)), by=(k,) -> begin
    k
end)


    group = result.solutions[key]
    @show key

    sort!(group, by=swm -> numpoints(swm[1]))


    sols = [s for s in solutions(group)]

    ns = [numpoints(s) for s in sols]
    errs = errors(key, result)
    kwargs = scatterlines_common_kwargs(key, result)


    j = if first(sols) isa BVPSolution
        begin
            1
        end
    else
        2
    end

    s = string(key.bdrycond_t) * string(key.approach_t)

    if !(s in visuals)
        push!(visuals, s)
    end

    i = findfirst(==(s), visuals)

    @show i, j

    scatterlines!(
        axs[j],
        ns,
        errs,
        ;
        color=kwargs.colormap[i],
        marker=MARKER_LABELS[i],
        strokecolor=kwargs.colormap[i],
        label="$(string(nameof(key.approach_t))), $(string(nameof(key.bdrycond_t))) BC, $(key.correction)",
        kwargs...,
    )
end

# fig[0, 1][1, 1:2] = l

# trendlines
α = 0.1
conv_style = (;
    linestyle=:solid,
    alpha=0.4,
    color=:grey,
    linewidth=4,
    label="O(exp(-$(α)n))",
)

foreach(axs) do ax
    lines!(ax,
        result.n_vals, # use last iteration for getting ns
        exp.(-α .* result.n_vals),
        ;
        conv_style...
    )
end

l = axislegend(axs[2]; position=:lb)


###########
# Save and display plot
###########
const PLOTFILE =
    joinpath("figures", basename(@__FILE__) * FILE * begin
        if nameof(Makie.current_backend()) === :CairoMakie
            ".pdf"
        else
            ".png"
        end
    end
    )
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
