module Operators

using Distances
using LinearAlgebra

include("manifolds.jl")
using .Manifolds

export LaplaceSLP


abstract type Operator end


struct LaplaceSLP{T<:Real, M<: AbstractMatrix{T}, V<:AbstractVector{T}} <: Operator
    target::M # target is m x 2
    source::Manifold{T, M, V} # source.x is n x 2
    matrix::M # matrix is n x m
end


# construct Laplace SLP from a source manifold and a list of target points
function LaplaceSLP(
    target::AbstractMatrix{T}, # target points to compute operator
    source::Manifold{T, M, V}, # source manifold e.g. domain boundary
) where {T<: Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}


    matrix = laplace_slp(target, source.x, source.w)

    return LaplaceSLP{T, M, V}(target, source, matrix)

end

"""
    laplace_slp_tangent(x::AbstractMatrix{T}, y::AbstractMatrix{T}, w::AbstractVector{T})

compute normal derivative of the laplace single layer potential operator

# Arguments
- `nx::AbstractMatrix{T}`: outward unit normal vectors
- `k::AbstractVector{T}`: curvatures
- `w::AbstractVector{T}`: weights
"""
function laplace_slp_tangent(
    nx::AbstractMatrix{T},
    k::AbstractVector{T},
    w::AbstractVector{T}
)::AbstractMatrix{T} where {T<:Real}
    # NOTE: this could be reused
    d = pairwise(Euclidean(), x, y, dims=1)

    An = nx ./d

    # TODO: handle if diagonal
    An[diagind(An)] = -0.5 * k

    An = An * (1/2pi) * w
    return An

end


function laplace_slp(
    x::AbstractMatrix{T}, # target (points where value is needed)
    y::AbstractMatrix{T}, # source (integration variable)
    w::AbstractVector{T}
)::AbstractMatrix{T} where {T<:Real}

    # TODO: maybe this check is expensive. should client be allowed to call with two same argument?
    if x == y
        return laplace_slp(x, w)
    end

    return _laplace_slp(x, y, w)
end

function laplace_slp(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
)::AbstractMatrix{T} where {T<:Real}
    return laplace_slp(x, y, 1.)
end


function laplace_slp(
    x::AbstractMatrix{T},
    w::AbstractVector{T}
)::AbstractMatrix{T} where {T<:Real}
    A = _laplace_slp(x, x, w)
    A[diagind(A)] .= 0.
    return A
end

function laplace_slp(
    x::AbstractMatrix{T},
)::AbstractMatrix{T} where {T<:Real}
    A = laplace_slp(x, x, 1.)
    return A
end

function _laplace_slp(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractVector{T}
)::AbstractMatrix{T} where {T<:Real}

    @assert size(x, 2) == 2 "x must be nx2"
    @assert size(y, 2) == 2 "y must be mx2"

    # kernel is symmetric, so source/target distinction is not meaningful??
    # shape of kernel matters though, e.g. apply operator to function acting on source/target points? in this case, function acts on the boundary (source points)
    # but to compute solution at arbitrary points, we are talking about m "target" points
    # so kernel is nxm and is applied to m x ...
    d = pairwise(Euclidean(), x, y, dims=1)

    return -log.(abs.(d)) .* (1/2pi) * w
end

end
