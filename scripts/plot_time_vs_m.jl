using BenchmarkTools
using GLMakie
using JLD2
using Statistics
using Glob

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("plot_utils.jl")

###########
# Read data
###########

benchmark_results = [BenchmarkTools.load(p)[1] for p in glob("data/benchmark*mx*.json")]
convergence_results = [load_object(p) for p in glob("data/convergence*mx*.jld2")]

benchmark_results .|> display
convergence_results .|> display

###########
# Make figure
###########
# ncols = 5
# nrows = 2
w=1200
fig = Figure(
    # title="$(mx) x $(mx) evaluation grid",
    size=(w, w*9÷16)
)
nticks = 10
cm = Makie.to_colormap(:tab10)
ax_time = Axis(
    fig[1, 1],
    xlabel="M",
    ylabel="Time (ms)",
    xscale=log10,
    yscale=log10,
    xticks=LogTicks(LinearTicks(nticks)),
    yticks=LinearTicks(nticks) |> LogTicks,
    ytickformat=values -> [string(floor(Int, v/1e+6)) for v in values],
    # xtickformat=values -> [string(floor(Int, v)) for v in values],
    palette=(; color=cm,)
)

# ax_scaling = Axis(
#     fig[1, end+1],
#     xlabel="N",
#     ylabel="Relative Overhead vs. N₁",
#     xscale=log10,
#     yscale=log10,
#     xticks=LinearTicks(nticks),
#     yticks=LinearTicks(nticks),
#     ytickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
# )

# ax_slowdown = Axis(
#     fig[1, end+1],
#     xlabel="M",
#     ylabel="Slowdown",
#     xscale=log10,
#     # yscale=log10,
#     xticks=LinearTicks(nticks),
#     yticks=LinearTicks(nticks),
#     # xtickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
#     ytickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
#     palette=(; color=cm,)
# )

n = 200

ms = Vector{Int}()
mids = Dict{SolverParameters,Vector{Float64}}()
los = Dict{SolverParameters,Vector{Float64}}()
his = Dict{SolverParameters,Vector{Float64}}()

for (cr, br) in zip(convergence_results, benchmark_results)
    bdry = (cr.solutions |> values |> first |> solutions |> first |> bvp).boundary

    near_idx, far_idx, bad_idx = classify(bdry, cr.x, Interior())
    m = sum(length, [far_idx, near_idx])
    push!(ms, m)
    ###########
    # Filter data
    ###########
    filter!(
        cr,
        (k) -> begin
            if !(order(k.correction) in [32,])
                return false
            end
            if !(cutoff(k.evalmethod) in [0.0, 0.05, 0.25])
                return false
            end
            if !(k.solution_t <: BVPSolution)
                return false
            end
            if !(k.bdrycond_t <: Dirichlet)
                return false
            end
            if !(k.approach_t <: Indirect)

                return false
            end
            return true
        end
    )

    cr |> display

    # decide what metric to use from benchmarking (times, gctimes, etc...)
    metric = :times
    # decide what estimator to use for summarizing the times (median, mean, etc...)
    estimator = median

    # the reference run is the fastest solver, used to compute slowdown of more accurate
    # solvers
    # reference_run = minimum(keys(cr.solutions))
    # # fetch the reference trials associated to each n value
    # reference_times = [
    #     begin
    #         # index of the benchmark trial
    #         id = get_string(
    #             bvp(soln), cr.x, reference_run.correction, reference_run.approach_t(),
    #             reference_run.evalmethod
    #         )
    #         # the trial at this n value
    #         trial = br[id]
    #         # e.g. the mean time at each n value
    #         estimator(getproperty(trial, metric))
    #     end
    #     for soln in solutions(cr.solutions[reference_run])
    #     if numpoints(soln) == 200
    # ]
    # @show reference_run, reference_times

    for (i, (key, group)) in enumerate(cr.solutions)

        get!(mids, key, [])
        get!(los, key, [])
        get!(his, key, [])

        # find solution that corresponds to n
        idx = findfirst(s -> numpoints(s) == n, solutions(group))

        soln = collect(solutions(group))[idx]

        # get all the benchmarking trials related to each n value
        # WARN: code duplicated for extracting trials
        trial_times = begin
            # index of the benchmark trial
            s = get_string(bvp(soln), cr.x, key.correction, key.approach_t(), key.evalmethod)
            # the trial at this n value
            trial = br[s]
            getproperty(trial, metric)
        end
        @show mids

        push!(mids[key], estimator(trial_times))
        push!(los[key], quantile(trial_times, 0.25))
        push!(his[key], quantile(trial_times, 0.75))

        @show key
        # kwargs = scatterlines_common_kwargs(key, cr)
    end
end

@show ms
@show mids

sp = sortperm(ms)
ms = ms[sp]

for key in mids |> keys |> collect |> sort

    for (ax, ref_val) in zip(
        [
            ax_time,
            # ax_scaling,
            # ax_slowdown
        ],
        [
            1.,
            # coarsest_times,
            # reference_times
        ]
    )

        scatterlines!(ax, ms, mids[key][sp] ./ ref_val;
            cycle=[
                [:color, :strokecolor, :linecolor]=>:color,
            ],
            label="$(cutoff(key.evalmethod))",
            # kwargs...
        )

        rangebars!(
            ax, ms,
            los[key][sp] ./ ref_val,
            his[key][sp] ./ ref_val,
            whiskerwidth=20,
            # colormap=kwargs.colormap,
            # colorrange=kwargs.colorrange,
            # color=fill(i, length(ns)),
            # alpha=0.3,
        )
    end

end


lines!(
    ax_time, ms, first(values(mids))[1] .* (ms ./ ms[1]) .^ 1,
    label="O(M)",
    color=:grey,
    linewidth=3,
    alpha=0.5,
    linestyle=:dot
)

l = axislegend(ax_time, position=:rb)

# c1, c2, c3 = scatterlines_common_colorbars!(fig, res)
# legend = scatterlines_common_legend!(fig, res)
# # legend.halign=:left
# fig[1, end+1][1, 1] = c1
# fig[1, end][1, 2] = c2
# fig[1, end][1, 3] = c3
# fig[0, 1:end] = legend

plotfile =
    joinpath("figures", basename(@__FILE__) * begin
        if nameof(Makie.current_backend()) === :CairoMakie
            ".pdf"
        else
            ".png"
        end
    end
    )
save(plotfile, fig)
@info "saved `fig` to $(plotfile)"
fig

