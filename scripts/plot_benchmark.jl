# Plot the solution time
# To acquire the data, first run `benchmark/benchmark.jl`

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools
using Statistics
using GLMakie
using JLD2

include("plot_utils.jl")

const FILE = "benchmark-scp6"
const DATAFILE = joinpath("data", FILE * ".jld2")


###########
# Read data
###########
res = load_object(DATAFILE)
###########
# Make figure
###########
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

ax_scaling = Axis(
    fig[1, end+1],
    xlabel="N",
    ylabel="Relative Overhead vs. N₁",
    xscale=log10,
    yscale=log10,
    xticks=LinearTicks(nticks),
    yticks=LinearTicks(nticks),
    ytickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
)

ax_slowdown = Axis(
    fig[1, end+1],
    xlabel="N",
    ylabel="Slowdown",
    xscale=log10,
    # yscale=log10,
    xticks=LinearTicks(nticks),
    yticks=LinearTicks(nticks),
    # xtickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
    ytickformat=values -> [isinteger(v) ? "$(Int(v))x" : "$(v)x" for v in values],
)

###########
# Filter data
###########
filter!(
    res,
    (k) -> begin
        if !(order(k.correction) in [8, 32, 16])
            return false
        end
        if !(cutoff(k.evalmethod) in [0.0, 0.05, 0.1])
            return false
        end
        if (k.solution_t <: BIESolution)
            return false
        end
        # if !(k.bdrycond_t <: Neumann)
        #     return false
        # end
        return true
    end
)

reference_run = nothing
reference_times = nothing

metric = times
estimator = median

# filter groups and select reference run
for (key, group) in res.solutions
    if isnothing(reference_run)||key < reference_run
        global reference_run = key
        global reference_times = estimator.(metric(group))
    end
end

@show reference_run

for (key, group) in res.solutions

    ns = numpoints.(solutions(group))

    errs = errors(key, res, group)

    mids = estimator.(metric(group))
    coarsest_times = mids[1]
    los = quantile.(metric(group), 0.25)
    his = quantile.(metric(group), 0.75)

    @show key
    kwargs = scatterlines_common_kwargs(key, res)


    for (ax, ref_val) in zip(
        [
            ax_time,
            ax_scaling,
            ax_slowdown
        ],
        [
            1.,
            coarsest_times,
            reference_times
        ]
    )

        scatterlines!(ax, ns, mids ./ ref_val; kwargs...)

        rangebars!(
            ax, ns,
            los ./ ref_val,
            his ./ ref_val,
            whiskerwidth=20,
            colormap=kwargs.colormap,
            colorrange=kwargs.colorrange,
            color=fill(kwargs.color, length(ns)),
            alpha=0.3,
        )
    end
end

c1, c2, c3 = scatterlines_common_colorbars!(fig, res)
legend = scatterlines_common_legend!(fig, res)
# legend.halign=:left
fig[1, end+1][1, 1] = c1
fig[1, end][1, 2] = c2
fig[1, end][1, 3] = c3
fig[0, 1:end] = legend

const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", "runtime_" * FILE * ".pdf")
else
    joinpath("figures", "runtime_" * FILE * ".png")
end
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"
fig

