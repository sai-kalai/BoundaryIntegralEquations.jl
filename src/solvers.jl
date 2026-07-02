


#
# Laplace - Interior - Dirichlet - Direct
#

# given precomputed operators
function solve_and_evaluate(
    # solve this
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    # with ansatz
    approach::Direct,
    # using data
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
    S_target::SingleLayer,
    D_target::DoubleLayer
)
    # TODO: work in place to avoid allocating a new matrix

    A = -0.5 * I + matrix(D_star) # TODO: replace \ by LinearSolve
    τ = A \ (H * bc.σ)
    u = S_target * τ - D_target * bc.σ
    return u, τ
end

# compute operators internally
function solve_and_evaluate(
    # solve this
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    # with ansatz
    approach::Direct,
    correction::HypersingularCorrection,
    # using data
    source::AbstractManifold,
    target::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)

    m = size(target, 1)
    n = size(source, 1)


    D_star = AdjointDoubleLayer(problem, matrix_factory(n, n))
    H = Hypersingular(problem, correction, matrix_factory(n, n))
    populate_matrices!(source, D_star, H)

    S_target = SingleLayer(problem, nothing, matrix_factory(m, n))
    D_target = DoubleLayer(problem, matrix_factory(m, n))
    populate_matrices!(source, target, S_target, D_target)

    return solve_and_evaluate(
        problem,
        side,
        bc,
        approach,
        D_star,
        H,
        S_target,
        D_target
    )

end

#
# Laplace - Interior - Dirichlet - Indirect
#

# given operators
function solve_and_evaluate(
    # solve this
    ::Laplace,
    ::Interior,
    bc::Dirichlet,
    # using ansatz
    ::Indirect,
    # with data
    D::DoubleLayer,
    H::Hypersingular,
    D_target::DoubleLayer
)

    φ = (-0.5 * I + matrix(D)) \ bc.σ

    u = D_target * φ
    τ = H * φ
    return u, τ
end

# compute operators
function solve_and_evaluate(
    # solve this
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    # with ansatz
    approach::Indirect,
    correction::HypersingularCorrection,
    # using data
    source::AbstractManifold,
    target::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)

    n = size(source, 1)
    m = size(target, 1)

    D = DoubleLayer(problem, matrix_factory(n, n))
    H = Hypersingular(problem, correction, matrix_factory(n, n))

    D_target = DoubleLayer(problem, matrix_factory(m, n))

    return solve_and_evaluate(
        problem,
        side,
        bc,
        approach,
        D,
        H,
        D_target,
    )
end


#
# Laplace - Interior - Neumann - Direct
#

#given operators
function solve_and_evaluate(
    # solve this
    ::Laplace,
    ::Interior,
    bc::Neumann,
    # with ansatz
    ::Direct,
    # using data
    S::SingleLayer,
    D::DoubleLayer,
    S_target::SingleLayer,
    D_target::DoubleLayer,
)
    σ = (0.5 * I + matrix(D)) \ (S * bc.τ)

    u = S_target * bc.τ - D_target * σ

    return u, σ
end

#
# Laplace - Interior - Neumann - Indirect
#

# given operators
function solve_and_evaluate(
    # solve this
    ::Laplace,
    ::Interior,
    bc::Neumann,
    # with ansatz
    ::Indirect,
    # using data
    S::SingleLayer,
    D_star::AdjointDoubleLayer,
    S_target::SingleLayer,
)
    η = (0.5 * I + matrix(D_star)) \ bc.τ

    u = S_target * η

    σ = S * η

    return u, σ
end

#
# Laplace - Interior - Neumann - Direct
#

# compute operators
function solve_and_evaluate(
    # solve this
    problem::Laplace,
    side::Interior,
    bc::Neumann,
    # with ansatz
    approach::Direct,
    correction::SingularCorrection,
    # using data
    source::AbstractManifold,
    target::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)

    n = size(source, 1)
    m = size(target, 1)

    S = SingleLayer(problem, correction, matrix_factory(n, n))
    D = DoubleLayer(problem, matrix_factory(n, n))
    populate_matrices!(source, S, D)

    S_target = SingleLayer(problem, nothing, matrix_factory(m, n))
    D_target = DoubleLayer(problem, matrix_factory(m, n))
    populate_matrices!(source, target, S_target, D_target)

    return solve_and_evaluate(
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

#
# Laplace - Interior - Neumann - Indirect
#

function solve_and_evaluate(
    # solve this
    problem::Laplace,
    side::Interior,
    bc::Neumann,
    # with ansatz
    approach::Indirect,
    correction::SingularCorrection,
    # using data
    source::AbstractManifold,
    target::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)
    n = size(source, 1)
    m = size(target, 1)

    S = SingleLayer(problem, correction, matrix_factory(n, n))
    D_star = AdjointDoubleLayer(problem, matrix_factory(n, n))
    populate_matrices!(source, S, D_star)

    S_target = SingleLayer(problem, nothing, matrix_factory(m, n))
    populate_matrices!(source, target, S_target)

    return solve_and_evaluate(
        problem,
        side,
        bc,
        approach,
        S,
        D_star,
        S_target,
    )


end

