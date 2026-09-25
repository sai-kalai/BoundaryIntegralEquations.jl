docstring="""
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__))
    ```
    ```julia
    julia> include("scripts/plot/$(basename(@__FILE__)))
    ```

Plot convergence for far, i.e. "nice" points, using different correction orders
only Direct approach, Dirichlet BC
"""

using JLD2
using GLMakie

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("utils.jl")

###########
# Read data
###########

datafile = "data/convergence/m20_test.jld2"
result = load_object(datafile)
@info "loaded `result` from $datafile"
result

###########
# Make figure
###########
w = 1200
fig = Figure(
    # title="Far points",
    size=(w, w*9÷16)
)

###########
# Filter data
###########
filter!(result, (k) -> begin
    if !(k.solution_t <: BVPSolution || k.solution_t <: BDPSolution)
        return false
    end
    if !(order(k.correction) in [Inf, 32, 8,])
        return false
    end
    if !(cutoff(k.evalmethod) in [
        0.0,
        # 0.01, 0.05, 0.1
    ])
        return false
    end
    if !(k.approach_t <: Indirect)
        return false
    end
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
    palette=get_palette(3),
) for (i, title) in enumerate(["BVP Solution", "Cauchy Data"])]

hideydecorations!(axs[2], grid=false)

foreach(axs) do ax
    ylims!(ax, (1e-16, 1e-1))
end

###########
# Add plots
###########
visuals = String[]

for key in result.solutions |> keys |> collect |> sort

    # for key in sort(collect(keys(result.solutions)), by=(k,) -> begin
    #     k
    # end)

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

    s = string(key.correction) * string(key.approach_t)

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
        cycle=Cycle([
                :marker,
                :color,
            ]; covary=false),
        label="$(string(nameof(key.bdrycond_t))) BC, $(
            begin

if key.correction isa Sidi
                "Richardson"
            elseif key.correction isa Zeta

                @sprintf("FD(%d)", key.correction.order)
            elseif key.correction isa KapurRokhlin
                @sprintf("KR(%d)", key.correction.order)

            end

            end
            )",
        kwargs...,
    )
end

# trendlines
as = [
    # 0.1,
    0.05
]

dense_n = extrema(result.n_vals) |> x -> range(x..., length=50) |> collect

foreach(zip(axs, [-5., -1.5])) do (ax, yoffset)
    foreach(as) do a
        convergence_trendline!(ax, extrema(result.n_vals), yoffset - 1.5, a, :expo)
    end
    foreach(union(result.fd_acc_vals, result.kr_acc_vals) |> enumerate) do (i, p)
        convergence_trendline!(ax, extrema(result.n_vals), yoffset, p, :poly)
    end
end

l = axislegend(axs[1], labelsize=25;
    tellwidth=false,
    tellheight=true,
    orientation=:horizontal,
    halign=:center,
    nbanks=5,
    labeljustification=:center,
    # position=:rt
)
fig[2, 1][1, :] = l

resize_to_layout!(fig)


###########
# Save and display plot
###########
save_and_display!(basename(@__FILE__), fig)
fig
