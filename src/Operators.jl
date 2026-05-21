module Operators



include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")


using LinearAlgebra


import ..Kernels
using ..Manifolds
using ..Models

export
    IntegralOperator,
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    matrix,
    compute_laplace_slp_matrix,
    compute_laplace_slp_matrix_and_normal_derivative,
    compute_laplace_slp_matrix_normal_derivative,
    compute_laplace_dlp_matrix_normal_derivative,
    compute_laplace_dlp_matrix

abstract type IntegralOperator end # TODO: move to Models

function operator_factory(
    ops::Vector{IntegralOperator}
)
end

# NOTE: is this the julian way for a getter/public api?
function matrix(op::IntegralOperator)::AbstractMatrix
    return op.matrix
end

function Base.:*(op::IntegralOperator, v::AbstractArray)
    return matrix(op) * v
end

function Base.:+(op::IntegralOperator, v::AbstractArray)
    return matrix(op) + v
end

function Base.:*(v::AbstractArray, op::IntegralOperator)
    return op * v
end

function Base.:+(v::AbstractArray, op::IntegralOperator)
    return op + v
end


# a.k.a S
struct SingleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a D a.k.a. ∂S/∂ny
struct DoubleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a  D* a.k.a. ∂S/∂nx
struct AdjointDoubleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a  N a.k.a. ∂S²/∂nx∂ny
struct Hypersingular{
    P<:BoundaryValueProblem,
    C<:HypersingularCorrection,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    correction::C
    matrix::M
end


# construct Laplace SLP from a source manifold and a list of target points
function SingleLayer(
    problem::Laplace,
    target::AbstractMatrix, # target points to compute operator
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    matrix = compute_laplace_slp_matrix(target, boundary.x) .* boundary.w'
    return SingleLayer(problem, matrix)
end

# self interaction
function SingleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    order::Int, # order of kapur rokhlin singular correction
)
    matrix = compute_laplace_slp_matrix(boundary.x, boundary.w, order)
    return SingleLayer(problem, matrix)
end

function DoubleLayer(
    problem::Laplace,
    target::AbstractMatrix, # target points to compute operator
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    matrix = compute_laplace_dlp_matrix(target, boundary.x, boundary.n) .* boundary.w'
    return DoubleLayer(problem, matrix)
end

# self interaction
function DoubleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    matrix = compute_laplace_dlp_matrix(boundary.x, boundary.n, boundary.k) .* boundary.w'
    return DoubleLayer(problem, matrix)
end


# self interaction
function AdjointDoubleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    matrix = compute_laplace_slp_matrix_normal_derivative(
        boundary.x,
        boundary.n,
        vec(boundary.k) # TODO: decide what to do here
    ) .* boundary.w' # TODO: check if weights do go here
    return AdjointDoubleLayer(problem, matrix)
end

# self interaction using zeta correction
function Hypersingular(
    problem::Laplace,
    boundary::AbstractManifold, # TODO: abstract manifold is no good, need to ensure that passed object has n, k, ...
    correction::Zeta,
)
    matrix = compute_laplace_dlp_matrix_normal_derivative(
        boundary.x,
        boundary.n,
        vec(boundary.k),
        vec(boundary.w),
        correction.order,
    )
    return Hypersingular(problem, matrix)
end

# self interaction using Sidi correction
function Hypersingular(
    problem::Laplace,
    boundary::AbstractManifold,
    ::Sidi,
)
    matrix = compute_laplace_dlp_matrix_normal_derivative(boundary.x, boundary.n) .* boundary.w'
    return Hypersingular(problem, matrix)
end


function compute_kernel_matrix(
    ::BoundaryValueProblem,
    ::IntegralOperator,
)::AbstractMatrix
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


# self interaction@vec  using kapur rokhlin
function compute_laplace_slp_matrix(
    x::AbstractMatrix, # list of x points (targets), matrix
    weights::AbstractVector, # list of weights
    order::Int, # quadrature accuracy order
)

    m, dim_x = size(x)

    A = zeros(Float64, m, m)

    k = clamp((order - 1) ÷ 2, 0, (m - 1) ÷ 2)
    stencil = krcoeffs(k + 1)
    stencil = [stencil[end:-1:2]; stencil] # TODO: make this prettier


    # TODO: inbounds macro was deleted
    for i in m:-1:1 # loop bacwards

        #diagonal term
        A[i, i] = -log(weights[i]) / 2pi

        for j in 1:i-1
            ker = Kernels.laplace_slp(
                view(x, i, :),
                view(x, j, :)
            )

            A[i, j] += ker
            A[j, i] += ker

        end

        for dj in -k:k
            j = mod1(i + dj, m)
            A[i, j] += stencil[dj+k+1] / 2pi
        end

        # TODO: i don't like having 3 loops
        for j in 1:m
            A[i, j] *= weights[j]
        end

    end
    return A
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

# obsolete-ish
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
            # TODO: maybe better to construct r as svector here
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
        for j in i+1:m
            val = Kernels.laplace_dlp(
                view(x, i, :),
                view(x, j, :),
                view(nx, j, :)
            )
            D[i, j] = val
        end
    end
    return D
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
    curvatures::AbstractVector,
    weights::AbstractVector,
    order::Int, # accuracy order
)

    m, dim_x = size(x)

    dD_dn = zeros(Float64, m, m)

    # FD stencil for second derivative
    k = (order - 2) ÷ 2

    stencil = fdcoeffs(2, k)

    stencil = [stencil[end:-1:2]; stencil] # TODO: make this prettier

    # `x` in the paper, i.e. for each point in the manifold, each row in the matrix
    @inbounds for i in m:-1:1

        dD_dn[i, i] = -π / 6 / weights[i] + curvatures[i]^2 * weights[i] / 4π


        # first sum: compute for other points  in the manifold the dlp normal derivative
        for j in 1:i-1
            ker = Kernels.laplace_dlp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :),
                view(nx, j, :),
            )

            # this way asymmetric weighting can be applied
            dD_dn[i, j] = ker * weights[j]
            dD_dn[j, i] = ker * weights[i]

        end

        # apply banded correction
        for dj in -k:k

            j = mod1(i + dj, m)

            r_norm_sq = norm(x[i, :] - x[j, :])^2
            r_prime_0_x = weights[i] * dj

            # B(j) = (r(j) ^ 2 - |ρ'(i) * j| ^ 2) / |ρ'(i) * j| ^ 2
            B = (r_norm_sq - r_prime_0_x^2) / r_prime_0_x^2

            # TODO: remove branch
            if i == j
                B = 0.
            end

            # g(j) = n(i) ⋅ n(j) |ρ'(j)|/(2π |ρ'(i)|) * (1 - B + B^2)
            g = dot(nx[i, :], nx[j, :]) * weights[j] / (weights[i]^2) * (1 - B + B^2)


            dD_dn[i, j] += stencil[dj+k+1] * g / 4π

        end


    end


    return dD_dn
end


end

