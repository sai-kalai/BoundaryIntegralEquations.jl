
module BoundaryIntegralEquationsMakieExt

using BoundaryIntegralEquations
using DelaunayTriangulation
using Makie


# non inplace version to allocate fig and ax
function BoundaryIntegralEquations.visualize(
    args...
)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())

    ax = visualize!(ax, args...)

    return fig, ax
end

function BoundaryIntegralEquations.visualize!(
    ax::Axis,
    b::DiscreteClosedCurve,
    # TODO: these should be kwargs
    arrows::Bool=true,
    parameter::Bool=true,
)

    curve = lines!(
        ax,
        polygon(b),
        label="Γ",
        ;
        color=parameter ? (1:size(b, 2)) : :black
    )

    if arrows
        norm = arrows2d!(ax, b.x, b.n[1, :], b.n[2, :],
            color="red",
            lengthscale=0.1,
        )
        tang = arrows2d!(ax, b.x, real.(b.cw), imag.(b.cw),
            color="red",
            lengthscale=0.1,
        )
    end

    # Legend(fig[1, 1][1, 2],
    #     [curve, norm],
    #     ["curve", "normal"];
    #     # tellwidth=false,
    #     # halign=:left,
    #     # valign=:bottom,
    # )
    return ax
end


@doc raw"""
    Makie.tricontourf!(ax::Axis, b::DiscreteClosedCurve, positions::AbstractMatrix, values::AbstractVector, boundary_values::AbstractVector, ;, kwargs...)

triangulated contour filled plot. Calculates triangulation including boundary
points
"""
function Makie.tricontourf!(
    ax::Axis,
    b::DiscreteClosedCurve,
    positions::AbstractMatrix,
    values::AbstractVector,
    boundary_values::AbstractVector,
    ;
    kwargs...
)

    # extract from col-major ordering into vector of tuples
    points = [Tuple(p) for p in eachcol(positions)]
    # make polygon from vertices of boundary
    poly = polygon(b)

    boundary_nodes, points = convert_boundary_points_to_indices(
        [poly,],
        ;
        existing_points=points
    )

    tri = triangulate(points; boundary_nodes=boundary_nodes)

    z = [
        values
        boundary_values
    ]


    co = Makie.tricontourf!(ax, tri, z; kwargs...)

    # x, y = eachrow(positions)
    # Makie.scatter!(ax, x, y; color=:red, marker=:cross)

    return co
end

@doc raw"""
    Makie.contourf!(ax::Axis, b::DiscreteClosedCurve, xs::AbstractVector, ys::AbstractVector, values::AbstractVector)

Contour plot some scalar function in a domain enclosed by `b`

# Arguments
- `ax::Axis`: axis to modify
- `b::DiscreteClosedCurve`: boundary
- `xs::AbstractVector`: x values, of size (N, )
- `ys::AbstractVector`: y values, of size (M, )
- `values::AbstractVector`: function values, of size (N, M)
"""
function Makie.contourf!(
    ax::Axis,
    b::DiscreteClosedCurve,
    xs::AbstractVector,
    ys::AbstractVector,
    values::AbstractMatrix,
    ;
    kwargs...
)

    co = Makie.contourf!(ax, xs, ys, values; kwargs...)
    return co
end

end
