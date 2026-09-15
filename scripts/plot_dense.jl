# Plot solution error on a dense grid inside the domain for different values
# of cutoff and discretization, for fixed quadrature order
# domain

using GLMakie
using StaticArrays
using JLD2

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools


const FILE = "benchmark-scp10"
const DATAFILE = joinpath("data", FILE * ".jld2")


# acquire data
n_grid = 200
Γ_dense = DiscreteClosedCurve(n_grid, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n_grid)
ys = range(ymin, ymax, length=n_grid)
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


for (j, n) in enumerate([200, 400, 800,])

    sol_cau = sols_cau[n]

    sol_pt = sols_pt[n]

    curve = bvp(sol_cau).boundary

    near, far, bad = classify(curve, x_dense, Interior(), 0.0)

    vals = [
        log10.(abs.(err) .+ eps(eltype(err)))
        for err in [
            sol_pt.u - res.u_exact,
            sol_cau.u - res.u_exact,
            sol_cau.u - sol_pt.u
        ]
    ]

    for (i, val) in enumerate(vals)

        ax = Axis(
            fig[i, j][1, 1];
            aspect=DataAspect(),
            title=if j == 1
                if i == 1
                    "Potential Theory"
                elseif i == 2
                    "Cauchy Integral"
                elseif i == 3
                    "Difference"
                else
                    ""
                end
            else
                ""
            end
        )

        val = reshape(val, (n_grid, n_grid))

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

        visualize!(ax, curve, false, false)

        # Hide interior axis labels/decorations
        # if col > 1
        #     hideydecorations!(ax, grid=false)
        # end
        # if row < n_rows
        #     hidexdecorations!(ax, grid=false)
        # end

        labls = [
            "n = $(n)",
            # "correction = $(k.correction)",
            # "approach = $(k.approach_t)",
            # "bc = $(k.bdrycond_t)",
        ]

        legend = Legend(
            fig[i, j][1, 3],
            [MarkerElement(marker=:circle, color=:transparent) for _ in 1:length(labls)],
            labls,
            position=:lt,       # :lt = left-top (or :rt, :lb, :rb)
            framecolor=:gray50,
            backgroundcolor=(:white, 0.85),
            patchsize=(0, 0),    # hide icon space so only text shows
            # tellwidth=true,
            # orientation=:vertical,
            # nbanks=4,
        )

        Colorbar(
            fig[i, j][1, 2],
            co;
            # label="log10 error",
            tellheight=false,
            ticks=LinearTicks(10)
        )
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
