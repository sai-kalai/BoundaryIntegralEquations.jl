docstring = """
Plots the solution errors at far points, comparing Direct and Indirect approaches

# Usage

run from root directory of the project

## From shell
```bash
    \$ julia scripts/plot/$(basename(@__FILE__))
```
"""

@info docstring

using JLD2
using Makie

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("utils.jl")

###########
# Read data
###########
result = load_object("data/convergence/m20_test.jld2")

###########
# Make figure
###########
w = 1200
fig = Figure(
    title="",
    size=(w, w*9÷16)
)

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

axs = [Axis(
    fig[1, 1][1, i],
    title=title,
    xlabel="$(_N)",
    ylabel="L∞-error",
    yscale=log10,
    xscale=log10,
    xticks=result.n_vals,
    palette=get_palette(2),
)
       for (i, title) in enumerate(["BVPSolution", "Cauchy Data"])]

hideydecorations!(axs[2], grid=false)
foreach(axs) do ax
    ylims!(ax, (1e-16, 1e-3))
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

    sols = group |> solutions |> collect
    ns = numpoints.(sols)
    errs = errors(key, result)

    kwargs = scatterlines_common_kwargs(key, result)


    j = if key.solution_t <: BVPSolution
        begin
            1
        end
    elseif key.solution_t <: BDPSolution
        2
    else
        error("unexpected solution type: $(key.solution_t)")
    end

    s = string(key.bdrycond_t) * string(key.approach_t)

    if !(s in visuals)
        push!(visuals, s)
    end

    i = findfirst(==(s), visuals)


    scatterlines!(
        axs[j],
        ns,
        errs,
        ;
        cycle=Cycle([
                :marker,
                :color,
            ]; covary=false),
        label="$(nameof(key.bdrycond_t)) BC, $(nameof(key.approach_t))",
        kwargs...,
    )
end

# fig[0, 1][1, 1:2] = l

as = [
    0.1,
    # 0.05
]

foreach(axs) do ax
    foreach(as) do a
        convergence_trendline!(ax, extrema(result.n_vals), -5., a, :expo)
    end

    foreach(union(result.fd_acc_vals, result.kr_acc_vals) |> enumerate) do (i, p)
        convergence_trendline!(ax, extrema(result.n_vals), -4.5, p, :poly)
    end
end

l = axislegend(axs[1], labelsize=25,
    tellwidth=false,
    tellheight=true,
    orientation=:horizontal,
    halign=:center,
    nbanks=4,
    labeljustification=:center,
)

fig[2, 1][1, :] = l

###########
# Save and display plot
###########
save_and_display!(basename(@__FILE__) * join(result.fd_acc_vals, "_"), fig)
fig
