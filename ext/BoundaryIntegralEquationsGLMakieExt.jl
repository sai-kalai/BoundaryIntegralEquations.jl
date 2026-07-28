

module BoundaryIntegralEquationsGLMakieExt

using BoundaryIntegralEquations
using GLMakie

function BoundaryIntegralEquations.visualize(m::DiscreteClosedCurve)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())

    curve = lines!(ax, m.x[1, :], m.x[2, :]; color=1:size(m, 2))

    norm = arrows2d!(ax, m.x[1, :], m.x[2, :], m.n[1, :], m.n[2, :],
        color="red",
        lengthscale=0.1,
    )
    tang = arrows2d!(ax, m.x[1, :], m.x[2, :], real.(m.cw), imag.(m.cw),
        color="red",
        lengthscale=0.1,
    )

    Legend(fig[1, 1][1, 2],
        [curve, norm],
        ["curve", "normal"];
        # tellwidth=false,
        # halign=:left,
        # valign=:bottom,
    )

    return fig, ax

end

end
