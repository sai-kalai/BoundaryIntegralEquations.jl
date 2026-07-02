


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

function solve_bie(
    ::Laplace,
    ::Interior,
    bc::Dirichlet,
    D::DoubleLayer,
)
    return (-0.5 * I + matrix(D)) \ bc.σ # auxiliary variable
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


    D_star = AdjointDoubleLayer(problem, source; matrix_factory)
    H = Hypersingular(problem, source, correction; matrix_factory)

    S_target = SingleLayer(problem, source, target; matrix_factory)
    D_target = DoubleLayer(problem, source, target; matrix_factory)

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
    problem::Laplace,
    side::Interior,
    bc::Dirichlet,
    # using ansatz
    ::Indirect,
    # with data
    D::DoubleLayer,
    H::Hypersingular,
    D_target::DoubleLayer
)

    φ = solve_bie(problem, side, bc, D)
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

    S = SingleLayer(problem, source, correction; matrix_factory=matrix_factory)
    D = DoubleLayer(problem, source; matrix_factory=matrix_factory)

    S_target = SingleLayer(problem, source, target; matrix_factory)
    D_target = DoubleLayer(problem, source, target; matrix_factory)

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

    S = SingleLayer(problem, source, correction; matrix_factory)
    D_star = AdjointDoubleLayer(problem, source;  matrix_factory)

    S_target = SingleLayer(problem, source, target; matrix_factory)

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

