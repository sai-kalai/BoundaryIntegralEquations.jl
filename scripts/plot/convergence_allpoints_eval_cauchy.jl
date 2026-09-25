docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__)) NUMPTS
    ```
    ```julia
    julia> ARGS=[NUMPTS]; include("scripts/plot/$(basename(@__FILE__)))
    ```

Plot the convergence behavior of the different methods at NUMPTS densely
sampled points, including many points near the boundary

"""

@info docstring

using JLD2
using GLMakie

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("utils.jl")

###########
# Read data
###########

m = nothing
result = nothing
try
    global m = parse(Int, ARGS[1])
    global datafile = "data/convergence/m$(m)_dense.jld2"
    global result = load_object(datafile)
    @info "loaded `result` from $datafile"
catch e
    @info docstring
    rethrow(e)
end

###########
# Make figure
###########
w = 1200
fig = Figure(
    title="All points",
    size=(w, w*9÷16)
)


###########
# Filter data
###########
filter!(result, (k) -> begin
    if !(k.solution_t <: BVPSolution)
        return false
    end
    if !(order(k.correction) in [
        32,
        # 16,
        # 8,
    ])
        return false
    end
    if !(cutoff(k.evalmethod) in [
        1.0,
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

axs = [Axis(
    fig[1, 1][1, i],
    xlabel="$(_N)",
    ylabel="L∞-error",
    yscale=log10,
    xscale=log10,
    xticks=result.n_vals,
    palette=get_palette(),
)
       for (i, title) in enumerate(["BVP Solution",])
]

foreach(axs) do ax
    ylims!(ax, (1e-16, 1e-4))
end

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
    kwargs = scatterlines_common_kwargs(key, result)

    for _norm in [Inf, 2]
        global errs = errors(key, result; _norm=_norm)
        if _norm != Inf
            errs ./= length(errs)
        end
        scatterlines!(
            axs[1],
            ns,
            errs,
            ;
            marker=MARKER_LABELS[1],
            markersize=30,
            # color=kwargs.colormap[i],
            # strokecolor=kwargs.colormap[i],
            label="$(begin
                if _norm == 2
                    L"RMSE"
                elseif _norm == Inf
                    L"l_∞"
                else
                    _norm |> string
                end
            end
                )",
            kwargs...,
        )
    end
end


# fig[0, 1][1, 1:2] = l

# trendlines
α = 0.1
conv_style = (;
    linestyle=:solid,
    alpha=0.4,
    color=:grey,
    linewidth=4,
    label=L"O(\exp(-%$(α)%$(_N)))",
)
dense_n = extrema(result.n_vals) |> x -> range(x..., length=50) |> collect
foreach(axs) do ax
    lines!(ax,
        dense_n, # use last iteration for getting ns
        exp.(-α .* dense_n),
        ;
        conv_style...
    )
end

l = axislegend(axs[end]; position=:rt)

###########
# Save and display plot
###########
plotfile = save_and_display!(basename(@__FILE__) * "m$(m)", fig)

fig
