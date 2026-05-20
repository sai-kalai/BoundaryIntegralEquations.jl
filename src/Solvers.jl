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
    correction::HypersingularCorrection,
    approach::Direct,
    targets::AbstractMatrix
)
    # operators with quadrature weights applied
    # TODO: how to save computation by getting both operators at the same time
    S = SingleLayer(problem, targets, boundary)
    D = DoubleLayer(problem, targets, boundary)

    D_star = AdjointDoubleLayer(problem, boundary)
    H = Hypersingular(problem, boundary, correction)

    return solve(
        problem,
        side,
        bc,
        approach,
        D_star,
        H,
        S,
        D
    )

end


# solve with given precomputed operators
function solve(
    ::Laplace,
    ::Interior,
    bc::Dirichlet,
    ::Direct,
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
    S_target::SingleLayer,
    D_target::DoubleLayer
)
    A = -0.5 * I + matrix(D_star) # TODO: figure out how to seamlessly fulfill the matrix api
    τ = A \ (H * bc.σ)
    u = S_target * τ - D_target * bc.σ
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
