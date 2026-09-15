
# Plot solution error on a dense grid inside the domain for different values
# of cutoff and discretization, for fixed quadrature order
# domain
using GLMakie
using StaticArrays
using JLD2
using NearestNeighbors
using Printf

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools


const FILE = "benchmark-scp10"
const DATAFILE = joinpath("data", FILE * ".jld2")



if false && isfile(DATAFILE)
    @info "loaded `res` from $(DATAFILE)"
    res = load_object(DATAFILE)
    n_grid = sqrt(length(res.x))
else
    # acquire data
    n_grid = 200
    @show "hi"
    Γ_dense = DiscreteClosedCurve(n_grid, starfish)
    xmin, xmax, ymin, ymax = extrema(Γ_dense)
    xs = range(xmin, xmax, length=n_grid)
    ys = range(ymin, ymax, length=n_grid)
    iter = Iterators.product(xs, ys)
    x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)
    res = run_all_simulations(
        x_dense,
        ;
        n_vals=[100, 200, 400,],
        cutoff_vals=[0., 0.05, 0.1, 0.5,],
        approach_types=[Indirect, Direct],
        bc_types=[Neumann, Dirichlet],
        fd_acc_vals=[32,],
    )
    save_object(DATAFILE, res)
    @info "saved `res` to $(DATAFILE)"
end


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
    if !(cutoff(k.evalmethod) in [0.0, 0.5])
        return false
    end
    if !(k.bdrycond_t <: Dirichlet)
        return false
    end
    if !(k.approach_t <: Indirect)
        return false
    end
    return true
end)

fig = Figure(
    size=(1920, 1080)
)

k_ref = res.solutions |> keys |> collect |> maximum
k_pt = res.solutions |> keys |> collect |> minimum
@show k_ref, k_pt

sols_cau = Dict(numpoints(s) => s for s in solutions(res.solutions[k_ref]))
sols_pt = Dict(numpoints(s) => s for s in solutions(res.solutions[k_pt]))

n_vals = [200, 400, 800,]

rows = length(n_vals)

axs = [Axis(
    fig[i, 1][1, 1],
    yscale=log10,
    xscale=log10,
    xticks=LogTicks(LinearTicks(10)),
    xtickformat=values -> [@sprintf("%.2f", v) for v in values],
    # ylims=(1e-16, 1e+1),
    # xlims=(1e-10, 1e+0)
) for i in 1:rows]

foreach(axs) do ax
    ylims!(ax, (1e-16, 1e-1))
    xlims!(ax, (1e-2, 0.2))
end

axs[1].title = "Potential Theory"
axs[2].title = "Cauchy Integral"
axs[3].title = "Difference"

for (j, n) in enumerate(n_vals[end:-1:1])

    sol_cau = sols_cau[n]

    sol_pt = sols_pt[n]

    curve = bvp(sol_cau).boundary

    near, far, bad = classify(curve, res.x, Interior(), 0.0)

    vals = [
        abs.(err)
        for err in [
            sol_pt.u - res.u_exact,
            sol_cau.u - res.u_exact,
            sol_cau.u - sol_pt.u
        ]
    ]

    for (i, val) in enumerate(vals)

        tree = KDTree(curve)
        _, dists = nn(tree, res.x)

        sc = scatter!(
            axs[i],
            dists ./ lengthscale(curve),
            val,
            label="n = $(n)",
            colormap=:tab10,
            color=j,
            alpha=0.1,
            colorrange=(1, 10),
        )

        axislegend(axs[i])

        # Hide interior axis labels/decorations
        if i < 3
            hidexdecorations!(axs[i], grid=false)
        end

        # labls = [
        #     "n = $(n)",
        #     # "correction = $(k.correction)",
        #     # "approach = $(k.approach_t)",
        #     # "bc = $(k.bdrycond_t)",
        # ]
        #
        # legend = Legend(
        #     fig[i, j][1, 3],
        #     [MarkerElement(marker=:circle, color=:transparent) for _ in 1:length(labls)],
        #     labls,
        #     position=:lt,       # :lt = left-top (or :rt, :lb, :rb)
        #     framecolor=:gray50,
        #     backgroundcolor=(:white, 0.85),
        #     patchsize=(0, 0),    # hide icon space so only text shows
        #     # tellwidth=true,
        #     # orientation=:vertical,
        #     # nbanks=4,
        # )

        # Colorbar(
        #     fig[i, j][1, 2],
        #     sc;
        #     # label="log10 error",
        #     tellheight=false,
        #     ticks=LinearTicks(10)
        # )
    end
end

const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", "dense" * FILE * ".pdf")
else
    joinpath("figures", "dense" * FILE * ".png")
end

save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
