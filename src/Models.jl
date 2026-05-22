module Models

using ..Manifolds

export
    #
    BoundaryValueProblem,
    Laplace,
    Helmholtz,
    Stokes,
    #
    IntegralOperator,
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    #
    Side,
    Interior,
    Exterior,
    #
    BoundaryCondition,
    Dirichlet,
    Neumann,
    #
    Approach,
    Direct,
    Indirect,
    #
    HypersingularCorrection,
    Sidi,
    Zeta,
    #
    SingularCorrection,
    KapurRokhlin

abstract type BoundaryValueProblem end
struct Laplace <: BoundaryValueProblem end
struct Helmholtz <: BoundaryValueProblem end
struct Stokes <: BoundaryValueProblem end

abstract type HypersingularCorrection end
struct Sidi <: HypersingularCorrection end
struct Zeta{T<:Int} <: HypersingularCorrection
    order::T
end

abstract type SingularCorrection end
struct KapurRokhlin{T<:Int} <: SingularCorrection
    order::T
end

abstract type IntegralOperator{P<:BoundaryValueProblem} end
# a.k.a S
struct SingleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number},
    C<:Union{SingularCorrection,Nothing},
} <: IntegralOperator
    problem::P
    matrix::M
    correction::C
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
    M<:AbstractMatrix{<:Number},
    C<:HypersingularCorrection,
} <: IntegralOperator
    problem::P
    matrix::M
    correction::C
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

end
