docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__)) NUMPTS
    ```
    ```julia
    julia> ARGS=[NUMPTS]; include("scripts/plot/$(basename(@__FILE__)))
    ```

Plot solution error on a NUMPTS point grid inside the domain for different values
of discretization versus the distance to the boundary
"""
using GLMakie
using BenchmarkTools
using StaticArrays
using JLD2
using NearestNeighbors
using Printf

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools


include("utils.jl")

###########
# Read data
###########
m = nothing
result = nothing
try
    global m = parse(Int, ARGS[1])
    datafile = "data/convergence/m$(m)_dense.jld2"
    global res = load_object(datafile)
    @info "loaded `res` from $datafile"
catch e
    @info docstring
    rethrow(e)
end


display(res)

filter!(res, (k) -> begin
    if !(k.solution_t <: BVPSolution)
        return false
    end
    # if !(k.correction isa KapurRokhlin)
    #     return false
    # end
    if !(order(k.correction) in [32])
        return false
    end
    # if !(cutoff(k.evalmethod) in [0.0, 0.25])
    #     return false
    # end
    if !(k.bdrycond_t <: Dirichlet)
        return false
    end
    if !(k.approach_t <: Indirect)
        return false
    end
    return true
end)


k_ref = res.solutions |> keys |> collect |> maximum
k_pt = res.solutions |> keys |> collect |> minimum
@show k_ref, k_pt

sols_cau = Dict(numpoints(s) => s for s in solutions(res.solutions[k_ref]))
sols_pt = Dict(numpoints(s) => s for s in solutions(res.solutions[k_pt]))

n_vals = res.n_vals[2:end]
@show n_vals



w = 1200
fig = Figure(
    title="",
    size=(w, w*9÷16)
)

axs = [Axis(
    fig[i, 1],
    yscale=log10,
    ylabel=ylabel,
    # xscale=log10,
    xticks=LinearTicks(25),
    # yticks=10. .^ (1:-2:-15),
    yticks=LogTicks(LinearTicks(5)),
    xtickformat=values -> [@sprintf("%.3f", v) for v in values],
    # ytickformat=values -> [@sprintf("%.1E", v) for v in values],
    # title=title,
    xticklabelrotation=pi/6,
    # xautolimitmargin=(0.2, 0.2),
    # ylims=(1e-16, 1e+1),
    # xlims=(1e-10, 1e+0)
) for (i, (title, ylabel)) in enumerate([
# ("Potential Theory", L"|u - u_{pt}|"),
# ("Cauchy Integral", L"|u - u_{ci}|"),
    ("Difference", L"|u_{pt} - u_{ci}|")]
)]

axs[end].xlabel="δ"

display(sols_cau)
display(sols_pt)

tolerances = [1e-10, 1e-8]
optimal_dists_dict = Dict{Int,Vector{Float64}}()


for (j, n) in enumerate(n_vals)

    sol_cau = sols_cau[n]

    sol_pt = sols_pt[n]

    curve = bvp(sol_cau).boundary
    near, far, bad = classify(curve, res.x, Interior(), 0.0)
    ls = lengthscale(curve)
    global dists
    tree = KDTree(curve)
    _, dists = nn(tree, res.x)

    global vals
    vals = [
        abs.(err)
        for err in [
            # sol_pt.u - res.u_exact,
            # sol_cau.u - res.u_exact,
            sol_cau.u - sol_pt.u
        ]
    ]
    difference = vals[end]

    # sort points by distance, find smallest distance where error is below tolerance
    sp = sortperm(dists)
    optimal_ids = [findlast(>(t), difference[sp]) for t in tolerances]
    optimal_dists = dists[sp][optimal_ids] ./ ls

    @show n, tolerances, optimal_dists_dict

    optimal_dists_dict[n] = optimal_dists

    for (i, val) in enumerate(vals)

        sc = scatter!(
            axs[i],
            dists ./ ls,
            val,
            # label="$(_N) = $(n)",
            markersize=3,
            alpha=0.2,
            colormap=:tab10,
            color=j,
            colorrange=(1, 10),
        )

        vlines!(
            axs[i],
            optimal_dists,
            # [1e-15, 1e+1],
            colormap=:tab10,
            color=j,
            colorrange=(1, 10),
            label="$_N = $n",
            # label="dist = $(optimal_dist)"
        )

        # Hide interior axis labels/decorations
        if i < length(axs)
            hidexdecorations!(axs[i], grid=false)
        end

    end

    diffs = difference[sp][optimal_ids]

    @show n, optimal_dists, diffs

    # markers for optimal distance
    scatter!(axs[end], optimal_dists, diffs, color=:black,
        markersize=10, marker=:xcross,)

    # trendlines
    # slope of log error vs distance
    m = ((diffs .|> log10 |> diff |> first) / (optimal_dists |> diff |> first))
    lines!(axs[end],
        optimal_dists * 1.05,
        diffs * 10 ^ (0.5),
        colormap=:tab10,
        color=j,
        colorrange=(1, 10),
        linewidth=3,
        linestyle=:dashdot,
        label=L"\mathcal{O} (10^{%$(round(m/n, digits=2))\delta N})"
    )

    # arrows for marking the datapoints
    foreach(enumerate(zip(optimal_dists, diffs))) do (i, (dist, dff))
        annotation!(axs[end],
            -75 * (-1) ^ (i),
            50 * (-1) ^ i,
            dist, dff,
            text=@sprintf("(%.3f, %.1e)", dist, dff),
            path=Ann.Paths.Line(),
            style=Ann.Styles.LineArrow(),
            labelspace=:relative_pixel,
            fontsize=16,
            font=:bold,
        )
    end

end

# set axis limits
foreach(axs) do ax
    ylims!(ax, (1e-15, 1e-1))
    xlims!(ax, (
        0.,
        (optimal_dists_dict |> values |> maximum |> maximum) * 1.2)
    )
end

# add legend
l = axislegend(axs[end],
    merge=true, patchlabelgap=5, labelsize=25,
    tellwidth=false, orientation=:horizontal,
    halign=:center,
    nbanks=2,
)

fig[end+1, :] = l

# resize_to_layout!(fig)

# find delta such that CI and PT agree up to a tolerance

###########
# Save and display plot
###########
plotfile = save_and_display!(basename(@__FILE__) * "m$(m)", fig)
