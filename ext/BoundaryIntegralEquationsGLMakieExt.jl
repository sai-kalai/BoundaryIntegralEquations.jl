

module BoundaryIntegralEquationsGLMakieExt

using BoundaryIntegralEquations
using GLMakie

function BoundaryIntegralEquations.visualize(m::DiscreteClosedCurve)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())

    curve = lines!(ax, m.x[:, 1], m.x[:, 2])

    veloc = arrows2d!(ax, m.x[:, 1], m.x[:, 2], m.n[:, 1], m.n[:, 2],
        color="red",
        lengthscale=0.1,
    )

    accel = arrows2d!(ax, m.x[:, 1], m.x[:, 2], m.t[:, 1], m.t[:, 2],
        color="blue",
        lengthscale=0.1,
    )

    Legend(fig[1, 1][1, 2],
        [curve, veloc, accel],
        ["curve", "normal", "tangent"];
        # tellwidth=false,
        # halign=:left,
        # valign=:bottom,
    )

    return fig, ax

end

end
