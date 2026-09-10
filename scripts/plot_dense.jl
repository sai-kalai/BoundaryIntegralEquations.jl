# Plot solution error on a dense grid inside the domain for different values
# of cutoff and discretization, for fixed quadrature order
# domain

using GLMakie
using StaticArrays
using JLD2

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools



const FILE = "benchmark-scp7"
const DATAFILE = joinpath("data", FILE * ".jld2")


# acquire data
n = 200
Γ_dense = DiscreteClosedCurve(n, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n)
ys = range(ymin, ymax, length=n)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

res = if true && isfile(DATAFILE)
    @info "loaded `res` from $(DATAFILE)"
    load_object(DATAFILE)
else
    r = run_all_simulations(
        x_dense,
        ;
        n_vals=[100, 200, 400,],
        cutoff_vals=[0., 0.05, 0.1, 0.5,],
        approach_types=[Indirect,],
        bc_types=[Neumann,],
        fd_acc_vals=[32,],
    )
    save_object(DATAFILE, r)
    @info "saved `res` to $(DATAFILE)"
    r
end

filter!(res, (k) -> begin
    if !(k.solution_t <: BVPSolution)
        return false
    end
    if !(k.correction isa KapurRokhlin)
        return false
    end
    if !(order(k.correction) in [32,])
        return false
    end
    if !(cutoff(k.evalmethod) in [0.0, 0.05, 0.1])
        return false
    end
    # # if !(k.approach_t <: Indirect)
    # #     return false
    # # end
    return true
end)

display(res)

# valid_sols = [
#     (k, sol) for (k, group) in res.solutions if (k.solution_t <: BVPSolution && k.correction isa Zeta)
#     for sol in solutions(group)
# ]

valid_sols = sort(collect(res.solutions); by=begin
    ((k, group),) -> cutoff(k.evalmethod)
end
)

# n_sols = sum([length(solutions(group)) for (k, group) in valid_sols])
# @show n_sols
# n_cols = ceil(Int, sqrt(n_sols))
# n_rows = ceil(Int, n_sols / n_cols)

fig = Figure(
# size=(300 * n_cols, 300 * n_rows)
)


for (i, (k, group)) in enumerate(valid_sols)
    @show k

    sols = [s for s in solutions(group) if numpoints(s) == 400]

    for (j, sol) in enumerate(sols)
        @show i, j

        ax = Axis(
            fig[i, j][1, 1];
            aspect=DataAspect()
        )


        sol.u[isnan.(sol.u)] .= -1.

        curve = bvp(sol).boundary

        near, far, bad = classify(curve, x_dense, Interior(), 0.0)

        # sol.u .+= res.u_exact[1]

        val = log10.(abs.(sol.u - res.u_exact) .+ eps(eltype(sol.u)))
        # reshape to plot in contourf
        # val[bad] .= NaN
        @show extrema(val[union(near, far)])

        @show cutoff(k.evalmethod)

        # msk = mask(curve, x_dense, cutoff(k.evalmethod))

        val = reshape(val, (n, n))


        # if ! (cutoff(k.evalmethod) == 0.)
        #     val[.!msk].=NaN
        # end

        # @show sum(msk)

        # fig, ax = visualize(Γ_dense, false, false)

        # lo, hi = extrema(val[.! outside_mask])
        # step = (hi-lo) < 5 ? 0.5 : 1
        # levels = range(floor(lo), ceil(hi), step=step)


        co = contourf!(
            ax,
            curve,
            xs,
            ys,
            val,
            levels=10,
            extendlow=:auto,
            extendhigh=:auto,
        )

        # sc0 = scatter!(
        #     ax, [Fixtures.test_locations();;
        #         ball(0.1, 10);;
        #         ball(0.3, 30);;
        #         ball(0.6, 60);;
        #         stack((t) -> starfish(t, 0.9), 0:0.1:2pi)
        #     ], label="Test Locations", strokewidth=1, color=:red,
        #     marker=:star4, strokecolor=:black,
        # )

        # for c in [0., 0.01, 0.05, 0.1, 0.5]
        #         visualize!(ax, DiscreteClosedCurve(100, (t) -> starfish(t, 1-c)), false, false)
        #     end

        visualize!(ax, curve, false, false)

        # Hide interior axis labels/decorations
        # if col > 1
        #     hideydecorations!(ax, grid=false)
        # end
        # if row < n_rows
        #     hidexdecorations!(ax, grid=false)
        # end

        axislegend(
            ax,
            [MarkerElement(marker=:circle, color=:transparent) for _ in 1:2],
            [
                "n = $(numpoints(sol))",
                "key = $(k)"
            ],
            position=:lt,       # :lt = left-top (or :rt, :lb, :rb)
            framecolor=:gray50,
            backgroundcolor=(:white, 0.85),
            patchsize=(0, 0),    # hide icon space so only text shows
            # tellwidth=true,
            # orientation=:vertical,
            # nbanks=4,
        )
        # linkaxes!(ax, content(fig[1, 1])...)

        Colorbar(
            fig[i, j][1, 2],
            co;
            # label="log10 error",
            tellheight=false,
            ticks=LinearTicks(10)
        )
    end
end
# Colorbar(fig[:, n_cols+1], co, label="log10 error")



# problem setup plot
# cof = tricontourf!(ax, Γ, x_dense, u_dense, σ;
#     levels=range(extrema(u)..., 10))
# Colorbar(fig[1, 2], cof)
# val = u_dense

const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", "dense" * FILE * ".pdf")
else
    joinpath("figures", "dense" * FILE * ".png")
end

save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
