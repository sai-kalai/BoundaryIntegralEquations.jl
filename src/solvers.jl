

struct BoundaryValueProblem{
    E<:DifferentialEquation,
    C<:BoundaryCondition,
    S<:DomainSide,
    B<:AbstractManifold,
}
    equation::E
    bc::C
    side::S
    boundary::B
end

function solve_linear_system(A, b; algorithm=nothing)
    pbm = LinearSolve.LinearProblem(A, b)
    sln = LinearSolve.solve(pbm, algorithm)
    return sln.u
end

# two variants are given: one where the operators are precomputed, and one that
# computes the operators.

#
# Laplace - Interior - Dirichlet - Direct
#

#given operators
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    ::Direct,
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
)::Neumann
    # TODO: work in place to avoid allocating a new matrix
    A = -0.5 + D_star # TODO: replace \ by LinearSolve
    τ = solve_linear_system(A, (H * problem.bc))
    return Neumann(τ) # this is actually already the unknown Neumann data
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    correction::HypersingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
)::Neumann

    D_star = AdjointDoubleLayer(problem.equation, problem.boundary; matrix_factory=matrix_factory)
    H = Hypersingular(problem.equation, problem.boundary, correction; matrix_factory=matrix_factory)

    # NOTE: other api:
    # D_star = AdjointDoubleLayer(problem.equation, matrix_factory(n, n))
    # H = Hypersingular(problem.equation, correction, matrix_factory(n, n))
    # populate_matrices!(problem.boundary, D_star, H)

    return solve(problem, approach, D_star, H)
end

# given operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    ::Direct,
    τ::Neumann,
    S_target::SingleLayer,
    D_target::DoubleLayer,
)::Tuple{AbstractVector,Neumann}
    u = S_target * τ - D_target * problem.bc
    return u, τ
end

# compute operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    τ::Neumann,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Neumann}

    S_target = SingleLayer(problem.equation, problem.boundary, target; matrix_factory=matrix_factory)
    D_target = DoubleLayer(problem.equation, problem.boundary, target; matrix_factory=matrix_factory)
    return evaluate(problem, approach, τ, S_target, D_target)
end

# given operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
    S_target::SingleLayer,
    D_target::DoubleLayer
)::Tuple{AbstractVector,Neumann}
    density = solve(problem, approach, D_star, H)
    u, τ = evaluate(problem, approach, density, S_target, D_target)
    return u, τ
end



# compute operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    correction::HypersingularCorrection,
    target::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Neumann}
    density = solve(problem, approach, correction; matrix_factory=matrix_factory)
    u, τ = evaluate(problem, approach, density, target; matrix_factory=matrix_factory)
    return u, τ
end

#
# Laplace - Interior - Dirichlet - Indirect
#

# given operators
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    D::DoubleLayer
)::BoundaryDensity

    φ = solve_linear_system(-0.5 + D, problem.bc.σ)
    return BoundaryDensity(φ)
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    ;
    matrix_factory::Function=default_allocator,
)::BoundaryDensity
    D = DoubleLayer(problem.equation, problem.boundary; matrix_factory=matrix_factory)
    solve(problem, approach, D)
end

# given  operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    φ::BoundaryDensity,
    # TODO: constrain operators to coincide with PDE of problem
    H::Hypersingular,
    D_target::DoubleLayer,
)::Tuple{AbstractVector,Neumann}
    τ = H * φ
    u = D_target * φ
    return u, Neumann(τ)
end


# compute operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    correction::HypersingularCorrection,
    φ::BoundaryDensity,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Neumann}


    H = Hypersingular(problem.equation, problem.boundary, correction; matrix_factory=matrix_factory)

    D_target = DoubleLayer(problem.equation, problem.boundary, target; matrix_factory=matrix_factory)

    return evaluate(problem, approach, φ, H, D_target)
end


# given operators
function solve_and_evaluate(problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    D::DoubleLayer,
    H::Hypersingular,
    D_target::DoubleLayer
)::Tuple{AbstractVector,Neumann}

    φ = solve(problem, approach, D)

    u, τ = evaluate(problem, approach, φ, H, D_target)

    return u, τ
end

# compute operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    correction::HypersingularCorrection,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Neumann}

    φ = solve(problem, approach; matrix_factory=matrix_factory)

    u, τ = evaluate(problem, approach, correction, φ, target; matrix_factory=matrix_factory)

    return u, τ
end


#
# Laplace - Interior - Neumann - Direct
#

# given operators
function solve(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    S::SingleLayer,
    D::DoubleLayer,
)::Dirichlet

    σ = solve_linear_system((0.5 + D), (S * problem.bc))

    return Dirichlet(σ)
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    correction::SingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
)::Dirichlet

    S = SingleLayer(problem.equation, problem.boundary, correction; matrix_factory=matrix_factory)
    D = DoubleLayer(problem.equation, problem.boundary; matrix_factory=matrix_factory)

    return solve(problem, approach, S, D)
end

# given  operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    σ::Dirichlet,
    S_target::SingleLayer,
    D_target::DoubleLayer,
)::Tuple{AbstractVector,Dirichlet}
    u = S_target * problem.bc - D_target * σ
    return u, σ
end

# compute operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    σ::Dirichlet,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Dirichlet}

    S_target = SingleLayer(problem, problem.boundary, target; matrix_factory=matrix_factory)
    D_target = DoubleLayer(problem, problem.boundary, target; matrix_factory=matrix_factory)

    return evaluate(problem, approach, σ, S_target, D_target)
end

#given operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    S::SingleLayer,
    D::DoubleLayer,
    S_target::SingleLayer,
    D_target::DoubleLayer,
)::Tuple{AbstractVector,Dirichlet}

    density = solve(problem, approach, S, D)
    u, σ = evaluate(problem, approach, density, S_target, D_target)

    return u, σ
end

# compute operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    correction::SingularCorrection,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator
)::Tuple{AbstractVector,Dirichlet}


    density = solve(problem, approach, correction; matrix_factory=matrix_factory)
    u, σ = evaluate(problem, approach, density, target; matrix_factory=matrix_factory)

    return u, σ
end

#
# Laplace - Interior - Neumann - Indirect
#


# given operators
function solve(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    D_star,
)::BoundaryDensity

    ψ = solve_linear_system((0.5 + D_star), problem.bc.τ)

    return BoundaryDensity(ψ)
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    ;
    matrix_factory::Function=default_allocator,
)::BoundaryDensity

    D_star = AdjointDoubleLayer(problem.equation, problem.boundary; matrix_factory=matrix_factory)

    return solve(problem, approach, D_star)
end

# given  operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    ψ::BoundaryDensity,
    S::SingleLayer,
    S_target::SingleLayer,
)::Tuple{AbstractVector,Dirichlet}

    σ = Dirichlet(S * ψ)
    u = S_target * ψ

    return u, σ
end

# compute operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    correction::SingularCorrection,
    ψ::BoundaryDensity,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Dirichlet}


    S = SingleLayer(problem.equation, problem.boundary, correction, matrix_factory=matrix_factory)

    S_target = SingleLayer(problem.equation, problem.boundary, target; matrix_factory=matrix_factory)
    # NOTE: big difference in api, because correction type is always passed...
    # S_target = SingleLayer(problem.equation, nothing, matrix_factory(m, n))

    return evaluate(problem, approach, ψ, S, S_target)
end

# given operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    S::SingleLayer,
    D_star::AdjointDoubleLayer,
    S_target::SingleLayer,
)::Tuple{AbstractVector,Dirichlet}
    density = solve(problem, approach, D_star)
    u, σ = evaluate(problem, approach, density, S, S_target)
    return u, σ
end


# compute operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    correction::SingularCorrection,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Dirichlet}

    density = solve(problem, approach; matrix_factory=matrix_factory)
    u, σ = evaluate(problem, approach, correction, density, target; matrix_factory=matrix_factory)
    return u, σ
end

