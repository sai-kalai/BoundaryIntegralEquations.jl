module Solvers
using Revise

using LinearAlgebra

using ..Models

using ..Operators

using ..Manifolds

export solve


function solve(
    ::BoundaryValueProblem,
    ::Side,
    ::AbstractManifold,
    ::BoundaryCondition,
    ::HypersingularCorrection,
    ::Approach,
    targets::AbstractMatrix
)

end

function solve(
    problem::Laplace,
    side::Interior,
    boundary::AbstractManifold,
    bc::Dirichlet,
    correction::Sidi,
    approach::Direct,
    targets::AbstractMatrix
)

end

function solve(
    problem::Laplace,
    ::Interior,
    boundary::AbstractManifold,
    bc::Dirichlet,
    correction::Zeta,
    ::Direct,
    targets::AbstractMatrix
)
    println("hello dispatch")
    # operators with quadrature weights applied
    # TODO: save computation by getting both operators at the same time
    S = SingleLayer(problem, targets, boundary)
    D = DoubleLayer(problem, targets, boundary)

    D_star = compute_laplace_slp_matrix_normal_derivative(boundary.x, boundary.n, vec(boundary.k))
    D_star .*= boundary.w' # apply quadrature weights
    A = -0.5 * I + D_star


    # direct approach

    # hypersingular operator using zeta quadrature
    H = compute_laplace_dlp_matrix_normal_derivative(
        boundary.x,
        boundary.n,
        vec(boundary.k),
        vec(boundary.w),
        correction.order
    )

    τ = A \ (H * bc.σ)
    u = S * τ - D * bc.σ
    return u, τ

end

function solve(
    problem::Laplace,
    type::Interior,
    bc::Neumann,
    approach::Direct,
    targets::AbstractMatrix
)

end

function solve(
    problem::Laplace,
    type::Interior,
    bc::Dirichlet,
    approach::Indirect,
    targets::AbstractMatrix
)

end

function solve(
    problem::Laplace,
    type::Interior,
    bc::Neumann,
    approach::Indirect,
    targets::AbstractMatrix
)

end
end
