# Plot the solution time
# To acquire the data, first run `benchmark/benchmark.jl`

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools
using Statistics
using GLMakie
using JLD2

include("plot_utils.jl")


const FILE = "benchmark-scp"
const DATAFILE = joinpath("data", FILE * ".jld2")


res = load_object(DATAFILE)
# ncols = 5
# nrows = 2
fig = Figure(
    # size=(ncols * 300, nrows * 300)
)
nticks = 5
ax_time = Axis(
    fig[1, end],
    xlabel="N",
    ylabel="Time (ms)",
    xscale=log10,
    yscale=log10,
    xticks=LinearTicks(nticks),
    yticks=LinearTicks(nticks),
    ytickformat=values -> [string(v/1e+6) for v in values],
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

ax_slowdown = Axis(
    fig[1, end+1],
    xlabel="N",
    ylabel="Slowdown",
    xscale=log10,
    yscale=log10,
    xticks=LinearTicks(nticks),
    yticks=LinearTicks(nticks),
    # xtickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
    ytickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
)

order_filter(x) = x in [32, 16]
cutoff_filter(x) = x in [0.0, 0.05, 0.1]
filter!(order_filter, res.kr_acc_vals)
filter!(order_filter, res.fd_acc_vals)
filter!(cutoff_filter, res.cutoff_vals)

filter!(((k, v),) -> begin
        # filter by
        if !(k.approach_t <: Indirect)
            return false
        end
        if !(k.solution_t <: BVPSolution)
            return false
        end
        if !(k.correction isa Zeta)
            return false
        end

        if !(cutoff_filter(cutoff(k.evalmethod)))
            return false
        end

        # if k.bdrycond_t <: Neumann
        #     return false
        # end

        if k.correction isa Union{Zeta,KapurRokhlin}
            _r = order_filter(k.correction.order)
            return _r
        end
        return true

    end, res.solutions
)

reference_run = nothing
reference_times = nothing


metric = gctimes
estimator = mean

# filter groups and select reference run
for (key, group) in res.solutions
    filter!((x) -> begin
            (s, m) = x
            # if !(numpoints(s) > 300)
            #     return false
            # end
            return true
        end, group)

    if isnothing(reference_run)||key < reference_run
        global reference_run = key
        global reference_times = estimator.(metric(group))
    end
end

for (key, group) in res.solutions

    ns = numpoints.(solutions(group))


    mids = estimator.(metric(group))
    coarsest_times = mids[1]
    los = quantile.(metric(group), 0.4)
    his = quantile.(metric(group), 0.6)

    kwargs = scatterlines_common_kwargs(key, res)

    @show key, mids[end]

    for (ax, ref_val) in zip(
        [
            ax_time,
            # ax_scaling,
            ax_slowdown
        ],
        [
            1.,
            # coarsest_times,
            reference_times
        ]
    )

        scatterlines!(ax, ns, mids ./ ref_val; kwargs...)

        rangebars!(
            ax, ns,
            los ./ ref_val,
            his ./ ref_val,
            whiskerwidth=10,
            colormap=kwargs.markercolormap,
            colorrange=kwargs.markercolorrange,
            color=fill(kwargs.markercolor, length(ns)),
        )
    end



end

c1, c2, c3 = scatterlines_common_colorbars!(fig, res)
legend = scatterlines_common_legend!(fig, res)
# legend.halign=:left
fig[1, end+1] = c1
fig[1, end+1] = c2
fig[1, end+1] = c3
fig[0, 1:end] = legend

const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", FILE * ".pdf")
else
    joinpath("figures", FILE * ".png")
end
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"
fig

