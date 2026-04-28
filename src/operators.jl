module Operators


using Distances
using LinearAlgebra

include("manifolds.jl")
include("kernels.jl")
import .Kernels
using .Manifolds


export LaplaceSLP, compute_laplace_slp_matrix, compute_laplace_slp_matrix_and_normal_derivative, compute_laplace_slp_matrix_normal_derivative


abstract type Operator end


# TODO: explore static vectors for performance

struct LaplaceSLP <: Operator
    target::Any # target is m x 2 # these are the points of interest
    source::Manifold # source.x is n x 2 # this is the manifold
    matrix::Any # resulting operator is mxn matrix.

    #   operate on nx1 vectors of quantities located at the manifold to obtain
    #   mx1 vectors of quantities located at the points of interest
end

struct LaplaceSlpDn

end

struct LaplaceDLP <: Operator
    target::Any,
    source::Manifold,
    matrix::Any
end

struct LaplaceDlpDn end



# construct Laplace SLP from a source manifold and a list of target points
function LaplaceSLP(
    target::Any, # target points to compute operator
    source::Manifold, # source manifold e.g. domain boundary
)

    matrix = compute_laplace_slp_matrix(target, source.x, source.w)

    return LaplaceSLP(target, source, matrix)

end

function compute_laplace_slp_matrix_and_normal_derivative(
    x, # target points, i.e. where to evaluate the function obtained by applying the operator
    y, # source points, i.e. where the integration variable moves when computing the operator
    nx,# outwards unitary normal vectors at the source
)::Tuple{Matrix,Matrix}
    return _compute_laplace_slp_matrix_and_normal_derivative(x, y, nx)
end


function compute_laplace_slp_matrix_normal_derivative(
    x,
    y,
    nx,
)

    return _compute_laplace_slp_matrix_normal_derivative(x, y, nx)

end

# TODO: weights are a property of the manifolds where integration occurrs

function compute_laplace_slp_matrix(
    x::AbstractMatrix{T}, # target (points where value is needed, anywhere)
    y::AbstractMatrix{T}, # source (integration variable around manifold)
    w::Union{T,AbstractVector{T}}
)::AbstractMatrix{T} where {T<:Real}

    # TODO: maybe this check is expensive. should client be allowed to call with two same argument?
    @assert x != y, "for self interaction, use method that takes one argument"
    return _compute_laplace_slp_matrix(x, y, w)
end


# self interaction: zero-out diagonal
function compute_laplace_slp_matrix(
    x::AbstractMatrix{T},
    w::Union{T,AbstractVector{T}}
)::AbstractMatrix{T} where {T<:Real}
    # how to avoid computing diagonal?
    A = _compute_laplace_slp_matrix(x, x, w)
    A[diagind(A)] .= 0. # zero diagonal
    return A
end


# NOTE: naming repetition could be replaced by namespacing
function compute_laplace_dlp_matrix(x, y, ny)

    @assert x != y, "call method with one argument for self interaction"

    return _compute_laplace_dlp_matrix(x, y, ny)

end

function compute_laplace_dlp_matrix_normal_derivative(x, y, nx, ny)

    @assert x != y, "call method with one argument for self interaction"

    return _compute_laplace_dlp_matrix_normal_derivative(x, y, nx, ny)

end

function compute_laplace_dlp_matrix_and_normal_derivative(x, y, nx, ny)

    @assert x != y, "call method with one argument for self interaction"

    return _compute_laplace_dlp_matrix_and_normal_derivative(x, y, nx, ny)

end


function _compute_laplace_slp_matrix(
    x, # list of x points (targets)
    y, # list of y points (source, integration variable)
    w,#::AbstractVector{T}
)

    # kernel is symmetric, so source/target distinction is not meaningful??
    # shape of kernel matters though, e.g. apply operator to function acting on source/target points? in this case, function acts on the boundary (source points)
    # but to compute solution at arbitrary points, we are talking about m "target" points
    # so kernel is nxm and is applied to m x ...
    m, dim_x = size(x)
    n, dim_y = size(y)

    A = zeros(Float64, m, n)


    @inbounds for i in 1:m, j in 1:n
        A[i, j] = Kernels.laplace_slp(
            view(x, i, :),
            view(y, j, :)
        )
    end
    return A
end

function _compute_laplace_slp_matrix_normal_derivative(
    x, # points of interest
    y, # domain boundary manifold
    nx, # unitary normal vectors at the y points
)

    m, dim_x = size(x)
    n, dim_y = size(y)

    # TODO: assert shapes

    dA_dn = zeros(Float64, m, n)

    @inbounds for i in 1:m, j in 1:n
        dA_dn[i, j] = Kernels.laplace_slp_dn(
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :)
        )
    end
    return dA_dn
end

function _compute_laplace_slp_matrix_and_normal_derivative(
    x,
    y,
    nx,
)
    m, dim_x = size(x)
    n, dim_y = size(y)

    # TODO: assert shapes

    A = zeros(Float64, m, n)
    dA_dn = zeros(Float64, m, n)



    # TODO: @views macro broken somehow
    @inbounds for i in 1:m, j in 1:n
        A[i, j], dA_dn[i, j] = Kernels.laplace_slp_and_dn(
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :)
        )
    end

    return A, dA_dn

end


function _compute_laplace_dlp_matrix(x, y, ny)
    m, dim_x = size(x)
    n, dim_y = size(y)

    A = zeros(Float64, m, n)

    @inbounds for i in 1:m, j in 1:n
        A[i, j] = Kernels.laplace_dlp(
            view(x, i, :),
            view(y, j, :),
            view(ny, j, :)
        )
    end

    return A

end

function _compute_laplace_dlp_matrix_normal_derivative(x, y, nx, ny)

    m, dim_x = size(x)
    n, dim_y = size(y)


    dA_dn = zeros(Float64, m, n)

    @inbounds for i in 1:m, j in 1:n
        dA_dn[i, j] = Kernels.laplace_dlp_dn(
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :),
            view(ny, j, :),
        )
    end
    return dA_dn

end

function _compute_laplace_dlp_matrix_and_normal_derivative(x, y, nx, ny)

    m, dim_x = size(x)
    n, dim_y = size(y)


    # TODO: maybe undef is better for initialization
    A = zeros(Float64, m, n)
    dA_dn = zeros(Float64, m, n)

    # TODO: @views macro broken somehow
    @inbounds for i in 1:m, j in 1:n
        A[i, j], dA_dn[i, j] = Kernels.laplace_dlp_and_dn(
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :),
            view(ny, j, :),
        )
    end

    return A, dA_dn



end


end

