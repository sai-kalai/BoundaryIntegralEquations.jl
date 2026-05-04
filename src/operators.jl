module Operators


using Distances
using LinearAlgebra

include("manifolds.jl")
include("kernels.jl")
import .Kernels
using .Manifolds


export
    LaplaceSLP,
    compute_laplace_slp_matrix,
    compute_laplace_slp_matrix_and_normal_derivative,
    compute_laplace_slp_matrix_normal_derivative,
    compute_laplace_dlp_matrix_normal_derivative,
    compute_laplace_dlp_matrix



abstract type Operator end

#
# # TODO: explore static vectors for performance
#
# struct LaplaceSLP <: Operator
#     target::Any # target is m x 2 # these are the points of interest
#     source::Manifold # source.x is n x 2 # this is the manifold
#     matrix::Any # resulting operator is mxn matrix.
#
#     #   operate on nx1 vectors of quantities located at the manifold to obtain
#     #   mx1 vectors of quantities located at the points of interest
# end
#
# struct LaplaceSlpDn
#
# end
#
# struct LaplaceDLP <: Operator
#     target::Any,
#     source::Manifold,
#     matrix::Any
# end
#
# struct LaplaceDlpDn end



# construct Laplace SLP from a source manifold and a list of target points
function LaplaceSLP(
    target::Any, # target points to compute operator
    source::Manifold, # source manifold e.g. domain boundary
)

    matrix = compute_laplace_slp_matrix(target, source.x)

    return LaplaceSLP(target, source, matrix)

end


function compute_laplace_slp_matrix(
    x, # list of x points (targets)
    y, # list of y points (source, integration variable)
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

# self interaction
function compute_laplace_slp_matrix(
    x, # list of x points (targets), matrix
)

    m, dim_x = size(x)

    A = zeros(Float64, m, m)


    @inbounds for i in 1:m
        # TODO: this branching might be problematic for GPUs
        A[i, i] = 0.
        for j in i:i-1
            val = Kernels.laplace_slp(
                view(x, i, :),
                view(x, j, :)
            )
            A[i, j] = val
            A[j, i] = val
        end
    end
    return A
end


function compute_laplace_slp_matrix_normal_derivative(
    x::AbstractMatrix, # points of interest
    y::AbstractMatrix, # domain boundary manifold
    nx::AbstractMatrix, # unitary normal vectors at the y points
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

# self interaction
function compute_laplace_slp_matrix_normal_derivative(
    x::AbstractMatrix, # points of interest
    nx::AbstractMatrix, # unitary normal vectors at the y points
    curvatures::AbstractVector, # curvature at x
)

    m, dim_x = size(x)

    dA_dn = zeros(Float64, m, m)

    @inbounds for i in 1:m

        # diagonal limit
        # -1/2 * curvature * 1/2pi
        dA_dn[i, i] = -0.25 / pi * curvatures[i]

        for j in 1:i-1
            val = Kernels.laplace_slp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )
            dA_dn[i, j] = val
        end

        for j in i+1:m
            val = Kernels.laplace_slp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )
            dA_dn[i, j] = val
        end

    end


    return dA_dn
end

function compute_laplace_slp_matrix_and_normal_derivative(
    x::AbstractMatrix, # points of interest
    y::AbstractMatrix, # domain boundary manifold
    nx::AbstractMatrix, # unitary normal vectors at the y points
)
    m, dim_x = size(x)
    n, dim_y = size(y)

    # TODO: assert shapes

    A = zeros(Float64, m, n)
    dA_dn = zeros(Float64, m, n)

    # zero

    # TODO: @views macro broken somehow
    @inbounds for i in 1:m, j in 1:n
        A[i, j], dA_dn[i, j] = Kernels.laplace_slp_and_dn(
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :)
        )
    end

    # TODO profileview.jl
    # benchmarktools.jl

    return A, dA_dn

end

# self interaction
function compute_laplace_slp_matrix_and_normal_derivative(
    x::AbstractMatrix, # points of interest
    nx::AbstractMatrix, # unitary normal vectors at the y points
    curvatures::AbstractVector, # curvature at x
)
    m, dim_x = size(x)

    # TODO: assert shapes

    A = zeros(Float64, m, m)
    dA_dn = zeros(Float64, m, m)

    @inbounds for i in 1:m

        A[i, i] = 0.
        dA_dn[i, i] = -0.5 * curvatures[i]

        for j in 1:i-1

            slp, slp_dn = Kernels.laplace_slp_and_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )

            A[i, j] = slp
            A[j, i] = slp

            dA_dn[i, j] = slp_dn

        end
        # NOTE: normal derivative isn't symmetric
        for j in i+1:m
            dA_dn[i, j] = Kernels.laplace_slp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )
        end
    end

    return A, dA_dn

end


function compute_laplace_dlp_matrix(
    x::AbstractMatrix,
    y::AbstractMatrix,
    ny::AbstractMatrix, # unitary normals at source
)
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

# self interaction
# NOTE: not symmetric
function compute_laplace_dlp_matrix(
    x::AbstractMatrix,
    nx::AbstractMatrix, # unitary normals at source
    curvatures::AbstractVector
)

    m, dim_x = size(x)

    D = zeros(Float64, m, m)

    @inbounds for i in 1:m

        D[i, i] = -0.25 / pi * curvatures[i]

        for j in 1:i-1
            val = Kernels.laplace_dlp(
                view(x, i, :),
                view(x, j, :),
                view(nx, j, :) # index with j
            )
            D[i, j] = val
        end
        for j in 1:i-1
            val = Kernels.laplace_dlp(
                view(x, i, :),
                view(x, j, :),
                view(nx, j, :)
            )
            D[j, i] = val
        end
    end
    return D
end




"""
    compute_laplace_dlp_matrix_normal_derivative(x::AbstractMatrix, y::AbstractMatrix, nx::AbstractMatrix, ny::AbstractMatrix)

a.k.a laplace hypersingular operator

# Arguments
- `x::AbstractMatrix`: [TODO:description]
- `y::AbstractMatrix`: [TODO:description]
- `nx::AbstractMatrix`: [TODO:description]
- `ny::AbstractMatrix`: [TODO:description]
"""
function compute_laplace_dlp_matrix_normal_derivative(
    x::AbstractMatrix,
    y::AbstractMatrix,
    nx::AbstractMatrix,
    ny::AbstractMatrix,
)
    m, dim_x = size(x)
    n, dim_y = size(y)

    dD_dn = zeros(Float64, m, n)

    @inbounds for i in 1:m, j in 1:n
        dD_dn[i, j] = Kernels.laplace_dlp_dn(
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :),
            view(ny, j, :),
        )
    end

    return dD_dn

end

# self interaction using Sidi's / Richarson's method
function compute_laplace_dlp_matrix_normal_derivative(
    x::AbstractMatrix,
    nx::AbstractMatrix,
)

    m, dim_x = size(x)

    dD_dn = zeros(Float64, m, m)

    @inbounds for i in 1:m

        # or leave diagonal empty and let quadrature client handle diagonal
        # dD_dn[i, i] = -pi/4 # NOTE: weights and dirichlet need to be multiplied to diagonal for computing the quadrature

        for j in (mod(i, 2)+1):2:i-1

            # twice weights for staggered grid
            val = 2 * Kernels.laplace_dlp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :),
                view(nx, j, :),
            )
            # WARN: this one turns out to be symmetric for some reason...?
            dD_dn[i, j] = val
            dD_dn[j, i] = val

        end


    end

    return dD_dn




end

# self interaction using FD correction
function compute_laplace_dlp_matrix_normal_derivative(
    x::AbstractMatrix,
    nx::AbstractMatrix,
    order::Int, # accuracy order
)
    native_matrix = compute_laplace_dlp_matrix_normal_derivative(x, nx, x, nx)

end

# NOTE: below code might not be needed, it is wip in anyway
function compute_laplace_dlp_matrix_and_normal_derivative(
    x::AbstractMatrix,
    y,
    nx,
    ny,
)

    m, dim_x = size(x)
    n, dim_y = size(y)


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

# self interaction
function compute_laplace_dlp_matrix_and_normal_derivative(
    x::AbstractMatrix,
    nx::AbstractMatrix,
)

    m, dim_x = size(x)
    n, dim_y = size(y)


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

