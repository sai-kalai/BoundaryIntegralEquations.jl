

include("utils.jl")

n = 40
c = DiscreteClosedCurve(n, starfish)

fig = Figure()
ax = Axis(fig[1, 1]; aspect=DataAspect())

curve, norm, tang = visualize!(ax, c, true, true; line_kwargs=(; linewidth=3), arr_kwargs=(;))

l1 = axislegend(
    ax,
    # [red_marker, blue_marker],
    # ["Unit Normal", "Complex Weight"],
    orientation=:vertical,
    # halign=:center,
    # valign=:center
)

red = MarkerElement(color=:red, marker=:utriangle, markersize=15)
blue = MarkerElement(color=:blue, marker=:utriangle, markersize=15)

l2 = Legend(
    fig,
    [red, blue],
    ["Unit Normal", "Complex Weight"],
    orientation=:vertical,
    halign=:center,
    valign=:center
)

# fig[1, 2] = l1

fig[1, 2] = l2


Colorbar(fig[1, 0], limits=(0, 2pi), colormap=:viridis, ticks=0:2pi, label="Parameter")



save_and_display!(@__FILE__, fig)
