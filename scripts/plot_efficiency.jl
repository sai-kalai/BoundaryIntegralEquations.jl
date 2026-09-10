# Plot the solution efficiency = 1 / (error * time)
# First acquire the time and error data by running the script
#
#```bash
#$ julia --project=. benchmark/benchmark.jl
#```
#
#or from the REPL
#
#```julia
#julia> include("benchmark/benchmark.jl")
#```

using JLD2
using GLMakie
using LaTeXStrings
using Statistics

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("plot_utils.jl")

###########
# Read data
###########
const FILE = "benchmark-scp8"
const DATAFILE = joinpath("data", FILE * ".jld2")
result = load_object(DATAFILE)
@info "loaded `result` from $DATAFILE"

###########
# Make figure
###########
fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel="n",
    ylabel=L"\frac{1}{\text{error} \times \text{runtime}}",
    xscale=log10,
    yscale=log10,
)

###########
# Filter data
###########
filter!(result, (k) -> begin
    if !(k.solution_t <: BVPSolution)
        return false
    end
    # if !(order(k.correction) in [16, 32, Inf])
    #     return false
    # end
    if !(k.approach_t <: Indirect)
        return false
    end
    return true
end)

###########
# Add plots
###########
for (key, group) in result.solutions

    @show key
    sols = solutions(group)
    ns = numpoints.(sols)

    errs = errors(key, result, group)

    # compute efficiencies
    mids = 1 ./ (errs .* median.(times(group)))
    los = 1 ./ (errs .* quantile.(times(group), 0.25))
    his = 1 ./ (errs .* quantile.(times(group), 0.75))

    kwargs = scatterlines_common_kwargs(key, result)

    scatterlines!(ax, ns, mids; kwargs...)

    rangebars!(
        ax, ns, los, his,
        whiskerwidth=20,
        colormap=kwargs.colormap,
        colorrange=kwargs.colorrange,
        color=fill(kwargs.color, length(ns)),
    )
end

c1, c2, c3 = scatterlines_common_colorbars!(fig, result)
legend = scatterlines_common_legend!(fig, result)
fig[1, 2][1, 1] = c1
fig[1, 2][1, 2] = c2
fig[1, 2][1, 3] = c3
fig[0, 1:end] = legend

###########
# Save and display plot
###########
const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", "efficiency_" * FILE * ".pdf")
else
    joinpath("figures", "efficiency_" * FILE * ".png")
end
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"
fig


