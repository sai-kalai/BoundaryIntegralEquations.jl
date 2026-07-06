

struct BoundaryValueProblem{
    E<:DifferentialEquation,
    C<:BoundaryCondition,
    S<:DomainSide,
    B<:AbstractManifold,
}
    equation::E
    boundary::B
    bc::C
    side::S
end

abstract type BoundaryIntegralEquationSolution end

#
# Laplace - Interior - Dirichlet - Direct
#

#given operators
function solve_bie(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,DiscreteClosedCurve},
    ::Direct,
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
)
    # TODO: work in place to avoid allocating a new matrix
    A = -0.5 * I + matrix(D_star) # TODO: replace \ by LinearSolve
    τ = A \ (H * problem.bc.σ)
    return τ

end

# compute operators
function solve_bie(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,DiscreteClosedCurve},
    approach::Direct,
    correction::HypersingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
)

    n = size(problem.boundary, 1)

    D_star = AdjointDoubleLayer(problem, matrix_factory(n, n))
    H = Hypersingular(problem, correction, matrix_factory(n, n))
    populate_matrices!(problem.boundary, D_star, H)

    τ = solve_bie(problem, approach, D_star, H)

    return τ

end


# given operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,DiscreteClosedCurve},
    ::Direct,
    τ::Vector,
    S_target::SingleLayer,
    D_target::DoubleLayer,
)
    u = S_target * τ - D_target * problem.bc.σ
    return u
end

# compute operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,DiscreteClosedCurve},
    ::Direct,
    τ::Vector,
    x::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)


    m = size(x, 1)
    n = size(problem.boundary, 1)

    S_target = SingleLayer(problem, nothing, matrix_factory(m, n))
    D_target = DoubleLayer(problem, matrix_factory(m, n))
    populate_matrices!(problem.boundary, x, S_target, D_target)

    u = S_target * τ - D_target * problem.bc.σ
    return u
end

# given operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,DiscreteClosedCurve},
    approach::Direct,
    # using data
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
    S_target::SingleLayer,
    D_target::DoubleLayer
)

    τ = solve_bie(problem, approach, D_star, H)

    u = evaluate(problem, approach, τ, S_target, D_target)

    return u, τ
end


# compute operators internally
function solve_and_evaluate(
    # solve this
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,DiscreteClosedCurve},
    # with ansatz
    approach::Direct,
    correction::HypersingularCorrection,
    # using data
    target::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)

    τ = solve_bie(problem, approach, correction; matrix_factory=matrix_factory)

    u = evaluate(problem, approach, τ, target; matrix_factory=matrix_factory)

    return u, τ

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

