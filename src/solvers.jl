

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


# two variants are given: one where the operators are precomputed, and one that
# computes the operators.

#
# Laplace - Interior - Dirichlet - Direct
#

#given operators
@doc raw"""
    solve(problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}, ::Direct, D_star::AdjointDoubleLayer, H::Hypersingular)

solve the boundary integral equation associated to the given boundary value problem using tthe direct approach by internally computing the required integral operators
 given precomputed integral operators

# Arguments
- `problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}`: Boundary value problem to solve
- `D_star::AdjointDoubleLayer`: Precomputed $D^*$ operator
- `H::Hypersingular`: Precomputed $H$ operator
"""
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    ::Direct,
    D_star::AdjointDoubleLayer,
    H::Hypersingular,
)::Neumann
    # TODO: work in place to avoid allocating a new matrix
    A = -0.5 + D_star # TODO: replace \ by LinearSolve
    τ = A \ (H * problem.bc)
    return Neumann(τ) # this is actually already the unknown Neumann data
end

# compute operators
@doc raw"""
    solve(problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}, approach::Direct, correction::HypersingularCorrection, ;, matrix_factory::Function=default_allocator)

solve the boundary integral equation associated to the given boundary value problem using tthe direct approach by internally computing the required integral operators

# Arguments
- `problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}`: Problem to solve
- `approach::Direct`: Solution approach
- `correction::HypersingularCorrection`: Type of correction to use to deal with the singularity when computing the Hypersingular operator.
"""
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    correction::HypersingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
)::Neumann

    n = size(problem.boundary, 1)

    D_star = AdjointDoubleLayer(problem.equation, matrix_factory(n, n))
    H = Hypersingular(problem.equation, correction, matrix_factory(n, n))
    populate_matrices!(problem.boundary, D_star, H)

    return solve(problem, approach, D_star, H)
end


# given operators
@doc raw"""
    evaluate(problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}, ::Direct, τ::Neumann, S_target::SingleLayer, D_target::DoubleLayer)

evaluate the solution of a boundary value problem at requested locations, given the solution to the associated boundary integral equation through the direct approach, and given precomputed operators

# Arguments
- `problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}`: boundary value problem to solve
- `τ::Neumann`: solution to the boundary integral equation through the direct approach
- `S_target::SingleLayer`: source-target interaction integral operator
- `D_target::DoubleLayer`: source-target interaction integral operator
"""
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
@doc raw"""
    evaluate(problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}, approach::Direct, τ::Neumann, target::AbstractMatrix, ;, matrix_factory::Function=default_allocator)

evaluate the solution of a boundary value problem at requested locations, given the solution to the associated boundary integral equation through the direct approach, computing the required operators

# Arguments
- `problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve}`: boundary value problem to solve
- `approach::Direct`: approach to solve the boundary integral equation
- `τ::Neumann`: solution to the associated boundary integral equation
- `target::AbstractMatrix`: locations to evaluate the solution to the BVP
"""
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    τ::Neumann,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
)::Tuple{AbstractVector,Neumann}
    m = size(target, 1)
    n = size(problem.boundary, 1)

    S_target = SingleLayer(problem.equation, nothing, matrix_factory(m, n))
    D_target = DoubleLayer(problem.equation, matrix_factory(m, n))
    populate_matrices!(problem.boundary, target, S_target, D_target)
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

    φ = (-0.5 + D) \ problem.bc.σ

    return BoundaryDensity(φ)
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    ;
    matrix_factory::Function=default_allocator,
)::BoundaryDensity

    n = size(problem.boundary, 1)
    D = DoubleLayer(problem.equation, matrix_factory(n, n))
    populate_matrices!(problem.boundary, D)

    solve(problem, approach, D)
end

# given  operators
function evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    φ::BoundaryDensity,
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

    m = size(target, 1)
    n = size(problem.boundary, 1)

    H = Hypersingular(problem, correction, matrix_factory(n, n))
    populate_matrices!(problem.boundary, H)

    D_target = DoubleLayer(problem, matrix_factory(m, n))
    populate_matrices!(problem.boundary, target, D_target)

    return evaluate(problem, approach, φ, H, D_target)
end


# given operators
function solve_and_evaluate(
    problem::BoundaryValueProblem{Laplace,Dirichlet,Interior,<:DiscreteClosedCurve},
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

    m = size(target, 1)
    n = size(problem.boundary, 1)

    D = DoubleLayer(problem.equation, matrix_factory(n, n))
    H = Hypersingular(problem.equation, correction, matrix_factory(n, n))
    populate_matrices!(problem.boundary, D, H)

    D_target = DoubleLayer(problem, matrix_factory(m, n))
    populate_matrices!(problem.boundary, target, D_target)

    return solve_and_evaluate(problem, approach, D, H, D_target)
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

    σ = Dirichlet((0.5 + D) \ (S * problem.bc))

    return σ
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Direct,
    correction::SingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
)::Dirichlet

    n = size(problem.boundary, 1)
    S = SingleLayer(problem.equation, correction, matrix_factory(n, n))
    D = DoubleLayer(problem.equation, matrix_factory(n, n))
    populate_matrices!(problem.boundary, S, D)

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

    m = size(target, 1)
    n = size(problem.boundary, 1)


    S_target = SingleLayer(problem, nothing, matrix_factory(m, n))
    D_target = DoubleLayer(problem, matrix_factory(m, n))
    populate_matrices!(problem.boundary, target, D_target)

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

    ψ = (0.5 + D_star) \ problem.bc.τ

    return BoundaryDensity(ψ)
end

# compute operators
function solve(
    problem::BoundaryValueProblem{Laplace,Neumann,Interior,<:DiscreteClosedCurve},
    approach::Indirect,
    ;
    matrix_factory::Function=default_allocator,
)::Dirichlet

    n = size(problem.boundary, 1)
    D_star = AdjointDoubleLayer(problem.equation, matrix_factory(n, n))
    populate_matrices!(problem.boundary, D_star)

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

    m = size(target, 1)
    n = size(problem.boundary, 1)

    S = SingleLayer(problem.equation, correction, matrix_factory(n, n))
    populate_matrices!(problem.boundary, S)

    S_target = SingleLayer(problem.equation, nothing, matrix_factory(m, n))
    populate_matrices!(problem.boundary, target, S_target)

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

    m = size(target, 1)
    n = size(problem.boundary, 1)

    S = SingleLayer(problem.equation, correction, matrix_factory(n, n))
    D_star = AdjointDoubleLayer(problem.equation, matrix_factory(n, n))
    populate_matrices!(problem.boundary, S)

    S_target = SingleLayer(problem.equation, nothing, matrix_factory(m, n))
    populate_matrices!(problem.boundary, target, S_target)

    u, σ = solve_and_evaluate(problem, approach, S, D_star, S_target)

    return u, σ
end

