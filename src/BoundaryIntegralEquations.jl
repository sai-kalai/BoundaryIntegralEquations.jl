module BoundaryIntegralEquations


#
# external packages
#
using LinearAlgebra
using StaticArrays
using FFTW
using PolygonOps
using NearestNeighbors
using LinearSolve
using RecursiveFactorization


# internal modules

#
# type definitions
#
@doc raw"""
    IntegralOperator

Represents an integral operator that maps densities in $\Gamma$ to their produced
potential in $\bar\Omega$.

TODO:
- comply with SciMLOperators instead of wrapping AbstractMatrix
- make parametric on equation, isself (only SL and DL can be not self)
"""
abstract type IntegralOperator end


@doc raw"""
    DifferentialEquation

Represents an elliptic or parabolic partial differenial equation that can
be solved using the Boundary Integral Method

"""
abstract type DifferentialEquation end
struct Laplace <: DifferentialEquation end
struct Helmholtz <: DifferentialEquation end
struct Stokes <: DifferentialEquation end


@doc raw"""
    AbstractSingularCorrection

Represents an approach used to evaluate singular integrals

"""
abstract type AbstractSingularCorrection end
# TODO: tie this concept of weakly singular and hypersingular explicitly to BC type
abstract type HypersingularCorrection <: AbstractSingularCorrection end
# rename: RichardsonExtrapolation; subtype: ZetaQuadrature
struct Sidi <: HypersingularCorrection end
# rename: FiniteDifferences
struct Zeta <: HypersingularCorrection
    order::Int
end
# rename: WeaklySingularCorrection
abstract type SingularCorrection <: AbstractSingularCorrection end
struct KapurRokhlin <: SingularCorrection
    order::Int
end

order(c::Zeta) = c.order
order(c::KapurRokhlin) = c.order
order(::Sidi) = Inf # unsure about this value


@doc raw"""
    DomainSide

Determines if the BVP is internal or external

"""
abstract type DomainSide end
struct Interior <: DomainSide end
struct Exterior <: DomainSide end
# add: both sides

# rename: BIEApproach
abstract type Approach end
struct Direct <: Approach end
struct Indirect <: Approach end

@doc raw"""
    AbstractBoundaryDensity

Represents a continuous density distribution on the boundary of a domain

"""
abstract type AbstractBoundaryDensity end
struct BoundaryDensity{T<:AbstractVector} <: AbstractBoundaryDensity
    φ::T # remove unicode from public API acc. to SciML best practices
end
abstract type BoundaryCondition <: AbstractBoundaryDensity end
struct Dirichlet{T<:AbstractVector} <: BoundaryCondition
    σ::T
end
struct Neumann{T<:AbstractVector} <: BoundaryCondition
    τ::T
end

# wip
@doc raw"""
    EvaluationMethod

Method used to evaluate the solution to a BVP inside the domain

"""
abstract type EvaluationMethod end
@doc raw"""
    CauchyIntegral

Use Global Quadrature via Cauchy Integral
\cite{barnettSpectrallyAccurateQuadratures2015}
"""
struct CauchyIntegral <: EvaluationMethod end
@doc raw"""
    PotentialTheory

Use layer potentials via Nystrom approximation

"""
struct PotentialTheory <: EvaluationMethod end
@doc raw"""
    Adaptive

Choose method depending on the distance from boundry to target point

TODO: this mixes policy and algorithm...

"""
struct DistancePolicy{T<:Real} <: EvaluationMethod
    cutoff::T
end

cutoff(::CauchyIntegral) = Inf
cutoff(::PotentialTheory) = 0.0
cutoff(d::DistancePolicy) = d.cutoff

@doc raw"""
    AbstractAlgorithm

Represents an algorithm used to solve a problem by an approach

"""

abstract type AbstractAlgorithm{A<:Approach} end


@doc raw"""

    DirectBIEAlgorithm
[TODO:description]

"""
abstract type BIEAlgorithm{A<:Approach} <: AbstractAlgorithm{A} end
function BIEAlgorithm{Direct}(
    correction::C,
    ls=RFLUFactorization(),
) where {C<:AbstractSingularCorrection}
    DirectBIEAlgorithm(correction, ls)
end
correction(a::BIEAlgorithm{Direct}) = a.correction
function BIEAlgorithm{Indirect}(ls=RFLUFactorization())
    IndirectBIEAlgorithm(ls)
end
struct DirectBIEAlgorithm{ # private
    C<:AbstractSingularCorrection,
    L<:LinearSolve.SciMLLinearSolveAlgorithm
} <: BIEAlgorithm{Direct}
    correction::C
    linearsolve::L
end
struct IndirectBIEAlgorithm{ # private
    L<:LinearSolve.SciMLLinearSolveAlgorithm
} <: BIEAlgorithm{Indirect}
    linearsolve::L
end

abstract type BVPAlgorithm{A<:Approach} <: AbstractAlgorithm{A} end
function BVPAlgorithm{Direct}()
    DirectBVPAlgorithm()
end
function BVPAlgorithm{Indirect}(m::EvaluationMethod)
    IndirectBVPAlgorithm(m)
end
struct DirectBVPAlgorithm <: BVPAlgorithm{Direct} # private
end
struct IndirectBVPAlgorithm{M<:EvaluationMethod} <: BVPAlgorithm{Indirect} # private
    evalmethod::M
end

# Boundary Data Problem Algorithms
abstract type BDPAlgorithm{A<:Approach} <: AbstractAlgorithm{A} end
function BDPAlgorithm{Direct}()
    DirectBDPAlgorithm()
end
function BDPAlgorithm{Indirect}(correction::C) where {C<:AbstractSingularCorrection}
    IndirectBDPAlgorithm(correction)
end
struct DirectBDPAlgorithm <: BDPAlgorithm{Direct} end
struct IndirectBDPAlgorithm{C<:AbstractSingularCorrection} <: BDPAlgorithm{Indirect}
    correction::C
end
# traits
approach(::AbstractAlgorithm{A}) where {A} = A
correction(::AbstractAlgorithm) = nothing
evalmethod(::AbstractAlgorithm) = Type{PotentialTheory}
evalmethod(a::BVPAlgorithm{Indirect}) = a.evalmethod
correction(a::BDPAlgorithm{Indirect}) = a.correction

# remaining: check that the correction specified for BIE and BDP match the respective
# boundary condition when calling solve(prob, alg)

# represents boundary curve
abstract type AbstractManifold end


@doc raw"""
    BoundaryValueProblem

Represents a boundary value problem

"""
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


@doc raw"""
    BIEProblem

Represents the boundary integral equation associated with a BVP

"""
struct BIEProblem{A<:Approach,P<:BoundaryValueProblem}
    # NOTE: this already depends on approach! approach is much more fundamental than i was considering it...
    # also consider caching operators here the same way that LinearProblem caches A and b
    # CommonSolve.init returns a "cache" object that contains the problem and algorithm
    bvp::P

    function BIEProblem{A}(p::P) where {A,P}
        return new{A,P}(p)
    end
end

abstract type AbstractSolution{A,T,P} end


struct BIESolution{
    A<:Approach,
    T<:Number, # floating point representation
    P<:BoundaryValueProblem, # BVP that this helps solve
    U<:AbstractVector{T},
    AL<:BIEAlgorithm{A},
    BP<:BIEProblem{A,P}, # BIE that this solves NOTE: superfluous
} <: AbstractSolution{A,T,P}
    # TODO: Link this with Linear Solution, wrapping or inheriting
    u::U # boundary density that solves the bie
    prob::BP
    alg::AL
end


function BIESolution(
    u,
    prob::BIEProblem{A,P},
    alg::BIEAlgorithm{A},
) where {A,P}

    return BIESolution{
        A,
        eltype(u),
        P,
        typeof(u),
        typeof(alg),
        typeof(prob),
    }(
        u,
        prob,
        alg
    )
end

struct BVPSolution{
    A<:Approach,
    T<:Number,
    P<:BoundaryValueProblem,
    U<:AbstractVector{T},
    AL<:BVPAlgorithm{A},
    BS<:BIESolution{A,T,P},
} <: AbstractSolution{A,T,P}
    u::U # potential generated from the boundary density in the domain
    bie_solution::BS # keep solution to the associated BIE
    prob::P
    alg::AL
    # TODO: missing: information about spatial discretization
end
function BVPSolution(
    u,
    bie_solution::BIESolution{A,T,P},
    prob::P,
    alg::BVPAlgorithm{A},
) where {A,T,P}
    return BVPSolution{
        A,
        eltype(u),
        P,
        typeof(u),
        typeof(alg),
        typeof(bie_solution),
    }(
        u,
        bie_solution,
        prob,
        alg,
    )

end


@doc raw"""
    BDProblem

Represents the problem of finding the missing cauchy data in a boundary value
problem, i.e. finding the Neumann data given a Dirichlet problem and vice-versa


"""
struct BDProblem{A<:Approach,P<:BoundaryValueProblem}
    bvp::P

    function BDProblem{A}(p::P) where {A,P}
        return new{A,typeof(p)}(p)
    end
end
@doc raw"""
    BDPSolution

Solution to a Boundary Data Problem, where the unknown Cauchy data of a Boundary
value problem is needed.

"""
struct BDPSolution{
    A<:Approach,
    T<:Number,
    P<:BoundaryValueProblem,
    U<:AbstractVector{T},
    AL<:BDPAlgorithm{A},
    DP<:BDProblem{A,P},
    BS<:BIESolution{A,T,P}, # solution of the bie
} <: AbstractSolution{A,T,P}
    u::U # potential generated from the boundary density in the domain
    bie_solution::BS # keep solution to the associated BIE
    prob::DP
    alg::AL
end
function BDPSolution(
    u,
    bie_solution::BIESolution{A,T,P},
    prob::BDProblem{A,P},
    alg::BDPAlgorithm{A},
) where {A,T,P}
    return BDPSolution{
        A,
        eltype(u),
        P,
        typeof(u), # WARN: actually, eltype(u) and T are tied... make this explicit
        typeof(alg),
        typeof(prob),
        typeof(bie_solution),
    }(
        u,
        bie_solution,
        prob,
        alg,
    )

end

const NumericalSolution{
    A,T,P
} = Union{
    BVPSolution{A,T,P},
    BIESolution{A,T,P},
    BDPSolution{A,T,P},
}

function Base.show(io::IO, ::Type{<:BVPSolution{A,T,P}}) where {A,T,P}
    print(io, "BVPSolution")
end
function Base.show(io::IO, ::Type{<:BIESolution{A,T,P}}) where {A,T,P}
    print(io, "BIESolution")
end
function Base.show(io::IO, ::Type{<:BDPSolution{A,T,P}}) where {A,T,P}
    print(io, "BDPSolution")
end

# solution/algorithm traits

numpoints(s::AbstractSolution) = size(bvp(s).boundary.x, 2)
bvp(s::BIESolution) = s.prob.bvp
bvp(s::BDPSolution) = s.prob.bvp
bvp(s::BVPSolution) = s.prob
boundary_condition(::AbstractSolution{A,T,<:BoundaryValueProblem{E,C}}) where {A,T,E,C} = C


#
# includes
#
include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")
include("densities.jl")
include("manifolds.jl")
include("operators.jl")
include("kernels.jl")
include("solvers.jl")
include("utils.jl")
include("close_evaluation.jl")

#
# exports
#
export DiscreteClosedCurve, make_dummy, make_offset, polygon, mask,
    lengthscale, classify

export DifferentialEquation, Laplace, Helmholtz, Stokes
export AbstractSingularCorrection, SingularCorrection, KapurRokhlin,
    HypersingularCorrection, Sidi, Zeta, order

export DomainSide, Interior, Exterior
export IntegralOperator, SingleLayer, DoubleLayer, AdjointDoubleLayer, Hypersingular
export Approach, Direct, Indirect
export BoundaryDensity, BoundaryCondition, Dirichlet, Neumann, data, apply
export cauchy_integral, holomorphism_boundary_limit

export BDProblem, BIEProblem

export AbstractSolution, BIESolution, BVPSolution, BDPSolution, NumericalSolution

export AbstractAlgorithm, BIEAlgorithm, BVPAlgorithm, BDPAlgorithm

export EvaluationMethod, PotentialTheory, CauchyIntegral, DistancePolicy, cutoff

export approach, boundary_condition

export kernel
export populate_matrices!
export BoundaryValueProblem, solve, evaluate, solve_and_evaluate

export starfish, ball

export numpoints, bvp
export visualize, visualize!

function visualize end
function visualize! end

# development tools module
include("DevTools/DevTools.jl")



# trick lsp into tracking files to get tab completion and jump to source (nvim hack)
@static if false
    include("../scripts/ellipse.jl")
    include("../scripts/plot_convergence_laplace2d.jl")
    include("../scripts/plot_utils.jl")
    include("../scripts/plot_benchmark.jl")
    include("../scripts/plot_dense.jl")
    include("../scripts/plot_problem_setup.jl")
    include("../scripts/plot_efficiency.jl")
    include("../scripts/profile.jl")
    include("../scripts/plot_domain_regions.jl")


    include("../benchmark/benchmark.jl")

    include("../test/convergence/laplace_2d.jl")
    include("../test/operators.jl")
    include("../test/close_evaluation.jl")
end

end # module BoundaryIntegralEquations
