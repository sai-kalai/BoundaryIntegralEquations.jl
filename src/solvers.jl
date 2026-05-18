module Solvers
include("models.jl")
export solve


function solve(
    ::BoundaryValueProblem,
    ::Side,
    ::BoundaryCondition,
    ::HypersingularCorrection,
    ::Approach,
    targets::AbstractMatrix
)

end

function solve(
    problem::Laplace,
    type::Interior,
    bc::Dirichlet,
    correction::Sidi,
    approach::Direct,
    targets::AbstractMatrix
)

end

function solve(
    ::Laplace,
    ::Interior,
    bc::Dirichlet,
    correction::Zeta,
    ::Direct,
    targets::AbstractMatrix
)
    println("hello dispatch")
    # operators with quadrature weights applied
    S = compute_laplace_slp_matrix(targets, bc.boundary.x) .* bc.boundary.w'
    D = compute_laplace_dlp_matrix(targets, bc.boundary.x, bc.boundary.n) .* bc.boundary.w'

    D_star = compute_laplace_slp_matrix_normal_derivative(bc.boundary.x, bc.boundary.n, vec(bc.boundary.k))
    D_star .*= bc.boundary.w' # apply quadrature weights
    A = -0.5 * I + D_star


    # direct approach

    # hypersingular operator using zeta quadrature
    H = compute_laplace_dlp_matrix_normal_derivative(
        bc.boundary.x,
        bc.boundary.n,
        vec(bc.boundary.k),
        bc.boundary.w,
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
