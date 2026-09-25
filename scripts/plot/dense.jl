docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__)) NUMPTS
    ```
    ```julia
    julia> ARGS=[NUMPTS]; include("scripts/plot/$(basename(@__FILE__)))
    ```

Plot solution error on a NUMPTS-point grid inside the domain, using several
values for N and different evaluation methods. Show plots in a grid with columns
for N and rows for boundary node count
"""
using GLMakie
using StaticArrays
using Printf
using JLD2

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

include("utils.jl")

res = nothing
m = nothing
try
    global m = parse(Int, ARGS[1])
    global datafile = "data/convergence/m$(m)_dense.jld2"
    global res = load_object(datafile)
    @info "loaded `result` from $datafile"
catch e
    @info "failed to acquire file from command line. Trying default"
    @info docstring
end


xs = res.x[1, :] |> unique |> sort
ys = res.x[2, :] |> unique |> sort

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
    if !(cutoff(k.evalmethod) in [0.0, 0.05, 1.0])
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
# size=(1920, 1080)
)

# find solutions for each row
k_ref = res.solutions |> keys |> collect |> maximum
k_pt = res.solutions |> keys |> collect |> minimum
k_dp = res.solutions |> keys |> collect |> ks -> filter(k -> k.evalmethod isa DistancePolicy, ks) |> minimum
@show k_ref, k_pt

sols_cau = Dict(numpoints(s) => s for s in solutions(res.solutions[k_ref]))
sols_pt = Dict(numpoints(s) => s for s in solutions(res.solutions[k_pt]))
sols_dp = Dict(numpoints(s) => s for s in solutions(res.solutions[k_dp]))

for (j, n) in enumerate([100, 200, 400])

    sol_cau = sols_cau[n]

    sol_pt = sols_pt[n]

    sol_dp = sols_dp[n]

    curve = bvp(sol_cau).boundary

    vals = [
        (log10.(abs.(err) .+ eps(eltype(err))), title)
        for (err, title) in [
            (sol_pt.u - res.u_exact, L"\log_{10}|u_{pt} - u|")
            (sol_cau.u - res.u_exact, L"\log_{10}|u_{ci} - u|")
            (sol_dp.u - res.u_exact, L"\log_{10}|u_{dp} - u|")
            (sol_cau.u - sol_pt.u, L"\log_{10}|u_{ci} - u_{pt}|")
        ]
    ]

    Label(fig[1, j, Top()], "$(_N) = $(n)")

    for (i, (val, title)) in enumerate(vals)
        ax = Axis(
            fig[i, j][1, 1];
            width=200, height=200,
            aspect=AxisAspect(1),
            # tellheight=true,
            # tellwidth=true,
        )

        val = reshape(val, (length(xs), length(ys)))

        colorlevels = 8
        co = contourf!(
            ax,
            curve,
            xs,
            ys,
            val,
            levels=colorlevels,
            colormap=:plasma,
        )

        visualize!(ax, curve, false, false)

        if j == 1
            Label(fig[i, 0], title, rotation=pi/2, tellheight=false,
                 fontsize=30)
        end

        # Hide interior axis labels/decorations
        if j > 1
            hideydecorations!(ax, grid=false)
        end

        if i < length(vals)
            hidexdecorations!(ax, grid=false)
        end


        low, high = val |> filter(!isnan) |> extrema
        edges = range(low, high, length=colorlevels + 1) |> collect
        @show edges[1], edges[end]
        centers = (edges[1:(colorlevels)] .+ edges[2:(colorlevels+1)]) .* 0.5
        ticks = (edges, [@sprintf("%.1f", e) for e in edges])
        # @show cb.ticks
        cb = Colorbar(
            fig[i, j][1, 2],
            co,
            ticks=ticks,
            tellheight=true,
        )
    end
end

resize_to_layout!(fig)
###########
# Save and display plot
###########
plotfile = save_and_display!(basename(@__FILE__) * "m$(m)", fig)
