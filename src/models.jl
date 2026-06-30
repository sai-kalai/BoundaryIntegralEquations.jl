
abstract type BoundaryValueProblem end
struct Laplace <: BoundaryValueProblem end
struct Helmholtz <: BoundaryValueProblem end
struct Stokes <: BoundaryValueProblem end

abstract type HypersingularCorrection end
struct Sidi <: HypersingularCorrection end
struct Zeta <: HypersingularCorrection
    order::Int
end

abstract type SingularCorrection end
struct KapurRokhlin <: SingularCorrection
    order::Int
end

abstract type IntegralOperator end
# a.k.a S
struct SingleLayer{
    P<:BoundaryValueProblem,
    C<:Union{SingularCorrection,Nothing},
    M<:AbstractMatrix{<:Number}, # TODO: change order of members/constructor arguments to match order of generic parameters
} <: IntegralOperator
    problem::P
    correction::C
    matrix::M
end

# a.k.a D a.k.a. ∂S/∂ny
struct DoubleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a  D* a.k.a. ∂S/∂nx
struct AdjointDoubleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a  N a.k.a. ∂S²/∂nx∂ny
struct Hypersingular{
    P<:BoundaryValueProblem,
    C<:HypersingularCorrection,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    problem::P
    correction::C
    matrix::M
end

abstract type Side end
struct Interior <: Side end
struct Exterior <: Side end

abstract type Approach end
struct Direct <: Approach end
struct Indirect <: Approach end


abstract type BoundaryCondition end
struct Dirichlet <: BoundaryCondition
    σ::AbstractVector
end
struct Neumann <: BoundaryCondition
    τ::AbstractVector
end

