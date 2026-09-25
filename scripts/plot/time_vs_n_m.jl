docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__)) DATAID...
    ```
    ```julia
    julia> ARGS=[DATAID...]; include("scripts/plot/$(basename(@__FILE__)))
    ```

Plots the solution times against the number of boundary nodes, using files
    identified by DATAID, which is formatted as
    m<NUMPTS>_<DOMAINTYPE>

    - NUMPTS is the number of evaluation points for evaluation of the BVP solution
    - DOMAINTYPE is the way the points are distributed, one of `far`, `dense`, `test`

The plot contains the times for several evaluation cases.
"""

@info docstring

using Printf
using BenchmarkTools
using GLMakie
using JLD2
using Statistics

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("utils.jl")

###########
# Read data
###########
convergence_results = nothing
benchmark_results = nothing
try
    convergence_files = ["data/convergence/$(dataid).jld2" for dataid in ARGS]
    benchmark_files = ["data/benchmark/$(dataid).json" for dataid in ARGS]
    global convergence_results = load_object.(convergence_files)
    global benchmark_results = BenchmarkTools.load.(benchmark_files) .|> first # [1] because json stores a 1-element vector of results
    @info "loaded `convergence_results` from $convergence_files"
    @info "loaded `benchmark_results` from $benchmark_files"
catch e
    @info docstring
    rethrow(e)
end

convergence_results |> display
benchmark_results |> display

###########
# Make figure
###########
w=1200
fig = Figure(
    size=(1200, 600)
)
nticks = 10
cm = Makie.to_colormap(:tab10)

axs = [
    Axis(
        fig[1, i],
        ;
        xlabel=_N,
        xscale=log10,
        # yticks=LogTicks(LinearTicks(nticks)),
        # palette=(; color=cm,),
        kwargs...
    ) for (i, kwargs) in enumerate([
        (
        ylabel="Time (ms)",
        ytickformat=values -> [@sprintf("%.1e", v/1e+6) for v in values],
        yscale=log10,
    ),
    # (
    #     yticks=[1, 2, 3, 4, 5],
    #     ylabel="Speedup",
    #     ytickformat=values -> [@sprintf("%.1f ×", v) for v in values],
    # )
    ])
]

for (res, benchmark_result) in zip(convergence_results, benchmark_results)

    ###########
    # Filter data
    ###########
    filter!(
        res,
        (k) -> begin
            if !(order(k.correction) in [32,])
                return false
            end
            # if !(cutoff(k.evalmethod) in [0.0, 0.1, 1.] || !(cutoff(k.evalmethod) in res.cutoff_vals))
            #     return false
            # end
            if !(cutoff(k.evalmethod) in [
                # 0.0,
                # # 0.05,
                1.
            ])
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

    res |> display

    # decide what BenchmarkTools.Trial attribute to use (times, gctimes, etc...)
    metric = :times
    # decide what estimator to use for summarizing the times (median, mean, etc...)
    estimator = median

    # the reference run is using the Cauchy Integral for evaluation everywhere,
    # used to compute speedup of using a better choice of cutoff
    reference_run = res.solutions |> keys |> maximum
    # fetch the reference trials associated to each n value
    global reference_times = [estimator(getproperty(t, metric)) for t in trials(reference_run, res, benchmark_result)]

    @show reference_run, reference_times

    global heuristic_times = Dict{Int,Float64}()

    # iterate over solutions and add elements to the plot
    for (i, key) in enumerate(res.solutions |> keys |> collect |> sort)

        group = res.solutions[key]

        filter!(pair -> 50 < numpoints(pair[1]) < 850, group)

        ns = numpoints.(solutions(group))

        # get all the benchmarking trials related to each n value
        times = [getproperty(t, metric) for t in trials(key, res, benchmark_result)]


        mids = estimator.(times)
        los = quantile.(times, 0.25)
        his = quantile.(times, 0.75)

        # WARN: this is very brittle
        if !(cutoff(key.evalmethod) in res.cutoff_vals)
            heuristic_times[ns[1]] = mids[1]
        end

        @show key
        kwargs = scatterlines_common_kwargs(key, res)

        for (ax, transformation) in zip(
            axs,
            [
                identity,
                (v) -> reference_times ./ v
            ]
        )
            scatterlines!(ax,
                ns,
                transformation(mids);
                marker=MARKER_LABELS[i],
                markersize=i,
                cycle=[
                    [:color, :strokecolor, :linecolor]=>:color,
                ],
                # label=L"\delta = %$(key.evalmethod |> cutoff), M = %$(size(res.x, 2))",
                label=L"\delta = %$(key.evalmethod |> cutoff), M = %$(size(res.x, 2))",
                kwargs...)

            rangebars!(
                ax, ns,
                transformation(los),# ./ ref_val,
                transformation(his),# ./ ref_val,
                whiskerwidth=20,
                # colormap=kwargs.colormap,
                # colorrange=kwargs.colorrange,
                # color=fill(i, length(ns)),
                # alpha=0.3,
            )
        end
    end
end

foreach(axs) do ax
    ax.xticks=convergence_results[1].n_vals[2:2:(end)]

end


if !isempty(heuristic_times)
    for (ax, ref_val) in zip(
        axs,
        [
            1.,
            reference_times
        ]
    )
        scatterlines!(ax,
            heuristic_times |> keys |> collect |> sort, (heuristic_times |> values |> collect |> sort) ./ ref_val;
            marker=MARKER_LABELS[end],
            cycle=[
                [:color, :strokecolor, :linecolor]=>:color,
            ],
            label=L"\delta = \frac{25}{%$_N}",
            # kwargs...
        )
    end
end

# draw straight lines for polynomial trends
convergence_trendline!(axs[1], (100, 400), +6.9, -1, :poly)
convergence_trendline!(axs[1], (100, 400), +5.7, -2, :poly)
# convergence_trendline!(axs[1], (100, 400), +5.7, -1, :poly)
convergence_trendline!(axs[1], (400, 800), +6.75, -3, :poly)
# convergence_trendline!(axs[1], (400, 800), +6.4, -3, :nlogn)
# for (p, xmin, xmax, yoffset, ls) in [
#     (1, 2, 3, -0.25, :dot),
#     # (2, 1, 3, 0.05, :dot),
#     (3, 1, 4, 0.15, :dashdot), (3, 3, 4, -0.15, :dashdot),
# ]
#     ns = convergence_results[1].n_vals[xmin:xmax]
#     ts = reference_times[xmin:xmax]
#
#     lines!(
#         axs[1],
#         ns,
#         ts[1] .* 10^(yoffset) .* (ns ./ ns[1]) .^ p,
#         label=L"\mathcal{O} (%$_N^%$p)",
#         color=:black,
#         linewidth=2,
#         # alpha=0.5,
#         linestyle=ls
#     )
# end

l = axislegend(axs[1], halign=:center,
    orientation=:horizontal,
    tekllheight=true,
    nbanks=2,
)
fig[2, 1] = l

resize_to_layout!(fig)

# Label(fig[0, :],
#     "Background Grid $(floor(Int, mx)) x $(floor(Int, mx))",
#     fontsize=20,
#     font=:bold
# )

###########
# Save and display plot
###########
plotfile = save_and_display!(basename(@__FILE__) * join(ARGS, "-"), fig)

