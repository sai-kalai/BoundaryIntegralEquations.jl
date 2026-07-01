
# api to solve with given precomputed operators
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
    # TODO: work in place to avoid allocating a new matrix

    A = -0.5 * I + matrix(D_star) # TODO: figure out how to seamlessly fulfill the matrix api
    τ = A \ (H * bc.σ)
    u = S_target * τ - D_target * bc.σ
    return u, τ
end

function solve_bie(
    ::Laplace,
    ::Interior,
    bc::Dirichlet,
    D::DoubleLayer,
)

    pbm = LS.LinearProblem(-0.5 * I + matrix(D), bc.σ)
    sln = LS.solve(pbm)
    # return lu(-0.5 * I + matrix(D)) \ bc.σ # auxiliary variable
    return sln.u
end

# indirect approach
function solve(
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    ::Indirect,
    D::DoubleLayer,
    H::Hypersingular,
    D_target::DoubleLayer
)

    φ = solve_bie(problem, side, bc, D)


    u = D_target * φ
    τ = H * φ
    return u, τ
end

function solve(
    ::Laplace,
    ::Interior,
    bc::Neumann,
    ::Direct,
    S::SingleLayer,
    D::DoubleLayer,
    S_target::SingleLayer,
    D_target::DoubleLayer,
)
    σ = (0.5 * I + matrix(D)) \ (S * bc.τ)

    u = S_target * bc.τ - D_target * σ

    return u, σ
end

function solve(
    ::Laplace,
    ::Interior,
    bc::Neumann,
    ::Indirect,
    S::SingleLayer,
    D_star::AdjointDoubleLayer,
    S_target::SingleLayer,
)
    η = (0.5 * I + matrix(D_star)) \ bc.τ

    u = S_target * η

    σ = S * η

    return u, σ
end


# api to compute operators internally
function solve(
    ::BoundaryValueProblem,
    ::Side,
    ::BoundaryCondition,
    targets::AbstractMatrix, # by convention, k(x, y), so targets go first
    ::AbstractManifold,
    ::HypersingularCorrection,
    ::Approach,
)
    error("specialized function not found")
end

function solve(
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    targets::AbstractMatrix,
    boundary::AbstractManifold,
    correction::HypersingularCorrection,
    approach::Direct,
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

function solve(
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    targets::AbstractMatrix,
    boundary::AbstractManifold,
    correction::HypersingularCorrection,
    approach::Indirect,
)

    D = DoubleLayer(problem, boundary)
    H = Hypersingular(problem, boundary, correction)

    D_target = DoubleLayer(problem, targets, boundary)

    return solve(
        problem,
        side,
        bc,
        approach,
        D,
        H,
        D_target,
    )
end


function solve(
    problem::Laplace,
    side::Interior,
    targets::AbstractMatrix,
    boundary::AbstractManifold,
    bc::Neumann,
    approach::Direct,
)

    S = SingleLayer(problem, boundary,)
    D = DoubleLayer(problem, boundary,)
    S_target = SingleLayer(problem, targets, boundary,)
    D_target = DoubleLayer(problem, targets, boundary,)

    return solve(
        problem,
        side,
        bc,
        approach,
        S,
        D,
        S_target,
        D_target,
    )
end

function solve(
    problem::Laplace,
    side::Interior,
    targets::AbstractMatrix,
    boundary::AbstractManifold,
    bc::Neumann,
    approach::Indirect,
)

    S = SingleLayer(problem, boundary)
    D_star = AdjointDoubleLayer(problem, boundary)
    S_target = SingleLayer(problem, targets, boundary,)

    return solve(
        problem,
        side,
        bc,
        approach,
        S,
        D_star,
        S_target,
    )


end

