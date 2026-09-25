docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__)) DATAID...
    ```
    ```julia
    julia> ARGS=[DATAID...]; include("scripts/plot/$(basename(@__FILE__)))
    ```

Plots the solution times against the number of evaluation points using files
    identified by DATAID, which is formatted as m<NUMPTS>_<DOMAINTYPE>

    - NUMPTS is the number of evaluation points for evaluation of the BVP solution
    - DOMAINTYPE is the way the points are distributed, one of `far`, `dense`, `test`
"""

using BenchmarkTools
using Printf
using GLMakie
using JLD2
using Statistics
using Glob

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

@info docstring

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

# convergence_results |> display
# benchmark_results |> display

###########
# Make figure
###########
w=1600
fig = Figure(
    # title="$(mx) x $(mx) evaluation grid",
    size=(w, w*9÷16)
)
nticks = 10

axs = [
    Axis(
        fig[1, i],
        ;
        xlabel="M",
        xscale=log10,
        xtickformat=values -> [@sprintf("%.1e", v) for v in values],
        # xticks=LinearTicks(4),
        palette=get_palette(2),
        kwargs...
    ) for (i, kwargs) in enumerate([
        (
            ylabel="Time (ms)",
            ytickformat=values -> [@sprintf("%.1e", v/1e+6) for v in values],
            yscale=log10,
        ),
        (
            # yticks=LinearTicks(nticks),
            ylabel="Speedup",
            yticks=[1, 2, 3, 4, 5],
            ytickformat=values -> [@sprintf("%.1f x", v) for v in values],
        )
    ])
]

n_vals = [100, 800]

ms = Vector{Int}()
mids = Dict{Tuple{Int,SolverParameters},Vector{Float64}}()
los = Dict{Tuple{Int,SolverParameters},Vector{Float64}}()
his = Dict{Tuple{Int,SolverParameters},Vector{Float64}}()

# n value -> time of slowest solver
reference_times = Dict{Int,Vector{Float64}}()

for (cr, br) in zip(convergence_results, benchmark_results)
    bdry = (cr.solutions |> values |> first |> solutions |> first |> bvp).boundary

    near_idx, far_idx, bad_idx = classify(bdry, cr.x, Interior())
    m = sum(length, [far_idx, near_idx, bad_idx])
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
            if !(cutoff(k.evalmethod) in [0.01, 1.0])
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


    global reference_run = cr.solutions |> keys |> maximum

    rt = [getproperty(t, metric) for t in trials(reference_run, cr, br)]

    for n in n_vals
        for (i, (key, group)) in enumerate(cr.solutions)
            get!(mids, (n, key), [])
            get!(los, (n, key), [])
            get!(his, (n, key), [])
            get!(reference_times, n, [])
            # find solution that corresponds to n
            idx = findfirst(s -> numpoints(s) == n, solutions(group))
            @show key
            @show idx
            if isnothing(idx)
                continue
            end

            trial_times = [getproperty(t, metric) for t in trials(key, cr, br)][idx]

            push!(mids[(n, key)], estimator(trial_times))
            push!(los[(n, key)], quantile(trial_times, 0.25))
            push!(his[(n, key)], quantile(trial_times, 0.75))

            if key == reference_run
                push!(reference_times[n], estimator(rt[idx]))
            end

            @show n, idx, reference_times

        end
    end
end

foreach(axs) do ax
    ax.xticks = ms
    @show ax.xticks
end


mks = Dict()

sp = sortperm(ms)
ms = ms[sp]

for (i, (n, key)) in mids |> keys |> collect |> v -> sort(v, by=((n, key),) -> key) |> enumerate

    for (ax, transformation) in zip(
        axs,
        [
            identity,
            (v) -> reference_times[n][sp] ./ v
        ]
    )
        if mids[(n, key)] |> length < ms |> length
            continue
        end


        scatterlines!(ax,
            ms,
            transformation(mids[(n, key)][sp]);
            # cycle=[
            #     [:color, :strokecolor, :linecolor]=>:color,
            # ],
            linewidth=3,
            # marker=MARKER_LABELS[mod1(i, 6)],
            colormap=:tab10,
            color=cutoff(key.evalmethod),
            colorrange=extrema(convergence_results[1].cutoff_vals),
            label=L"\delta%$(cutoff(key.evalmethod)), %$_N = %$n",
            markersize=20,
            strokewidth=2,
            cycle=Cycle([
                    :marker
                ]; covary=false),
        )

        rangebars!(
            ax, ms,
            transformation(los[(n, key)][sp]),
            transformation(his[(n, key)][sp]),
            whiskerwidth=20,
            linewidth=3,
            colormap=:tab10,
            colorrange=extrema(convergence_results[1].cutoff_vals),
            color=fill(cutoff(key.evalmethod), length(ms)),
        )
    end
end


lines!(
    axs[1],
    ms,
    10 ^ (-1) * first(values(mids))[1] .* (ms ./ ms[1]) .^ 1,
    label="O(M)",
    color=:grey,
    linewidth=3,
    alpha=0.5,
    linestyle=:dot
)

l = axislegend(axs[1], position=:rb)
# l = axislegend(axs[2], position=:rb)
resize_to_layout!(fig)

###########
# Save and display plot
###########
plotfile = save_and_display!(basename(@__FILE__) * join(ARGS, "-"), fig)

