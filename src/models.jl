include("manifolds.jl")
using .Manifolds


abstract type BoundaryValueProblem end
struct Laplace <: BoundaryValueProblem end
struct Helmholtz <: BoundaryValueProblem end
struct Stokes <: BoundaryValueProblem end

abstract type Side end
struct Interior <: Side end
struct Exterior <: Side end


abstract type Approach end
struct Direct <: Approach end
struct Indirect <: Approach end

abstract type HypersingularCorrection end
struct Sidi <: HypersingularCorrection end
struct Zeta{T<:Int} <: HypersingularCorrection
    order::T # NOTE: question here
end

abstract type BoundaryCondition end
struct Dirichlet <: BoundaryCondition
    boundary::Manifold
    σ::AbstractVector
end
struct Neumann <: BoundaryCondition
    boundary::Manifold
    τ::AbstractVector
end

