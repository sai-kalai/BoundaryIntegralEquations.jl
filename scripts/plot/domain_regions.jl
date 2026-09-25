docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__)) NUMPTS
    ```
    ```julia
    julia> ARGS=[NUMPTS]; include("scripts/plot/$(basename(@__FILE__)))
    ```

Produce the data and plot the different regions of the domain """
include("utils.jl")

@info docstring

n = 200
Γ = DiscreteClosedCurve(n, starfish)
xmin, xmax, ymin, ymax = extrema(Γ)
xs = range(xmin, xmax, length=n) |> collect
ys = range(ymin, ymax, length=n) |> collect
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

sides = [Interior(), Exterior()]
d=0.1
fig = Figure(size=(840, 460))
axs = [Axis(
    fig[1, i], aspect=DataAspect(),
    title="$(side |> typeof |> string) Ω",
    tellwidth=false,
) for (i, side) in enumerate(sides)]

u = zeros(length(xs), length(ys))

labels = [
    L"\delta = %$(d)",
    "Cauchy Integral",
    "Potential Theory",
    "Ignored",
]
colors = [:blue, :green, :red]

for (ax, side) in zip(axs, sides)
    nr, fr, bd = classify(Γ, x_dense, side, d)

    u[nr] .= 1
    u[fr] .= 2
    u[bd] .= 3

    heatmap!(ax, xs, ys, u, colormap=colors, alpha=0.4)
    # label="Cauchy Integral")
    # heatmap!(ax, xs, ys, color=:green, alpha=0.4,
    #     label="Potential Theory")
    # heatmap!(ax, xs, ys, color=:red, alpha=0.4,
    #     label="Ignored",
    # )
    visualize!(ax, Γ, false, false)
end

leg_elements = [
    MarkerElement(color=:transparent, marker=:rect)
    [MarkerElement(color=c, marker=:rect) for c in colors];
]

Legend(fig[0, :], leg_elements, labels, orientation=:horizontal,
    tellwidth=false, tellheight=true, halign=:center
)

resize_to_layout!(fig)

###########
# Save and display plot
###########
plotfile = save_and_display!(basename(@__FILE__), fig)


