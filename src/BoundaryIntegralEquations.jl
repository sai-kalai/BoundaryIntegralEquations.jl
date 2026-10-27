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

@doc raw"""
    DomainSide

Determines if the BVP is internal or external

"""
abstract type DomainSide end


abstract type AbstractDiscretization end
function populate_matrices!(::AbstractDiscretization, args...) end
struct Nystroem <: AbstractDiscretization end
# TODO: figure out how to move concrete struct to its own file. Problem is,
# constructors in operators.jl depend on it.

@doc raw"""
    AbstractBoundaryDensity

Represents a continuous density distribution on the boundary of a domain

"""
abstract type AbstractBoundaryDensity end
abstract type BoundaryCondition <: AbstractBoundaryDensity end

# rename: AbstractBIEApproach
abstract type Approach end

# wip
@doc raw"""
    EvaluationMethod

Method used to evaluate the solution to a BVP inside the domain

"""
abstract type EvaluationMethod end # todo rename abstract
@doc raw"""
    CauchyIntegral

Use Global Quadrature via Cauchy Integral
\cite{barnettSpectrallyAccurateQuadratures2015}
"""

@doc raw"""
    AbstractAlgorithm

Represents an algorithm used to solve a problem by an approach

"""
abstract type AbstractAlgorithm{A<:Approach} end

# TODO: rename abstract
abstract type AbstractBIEAlgorithm{A<:Approach} <: AbstractAlgorithm{A} end
abstract type AbstractBVPAlgorithm{A<:Approach} <: AbstractAlgorithm{A} end
abstract type AbstractBDPAlgorithm{A<:Approach} <: AbstractAlgorithm{A} end

abstract type AbstractSingularCorrection end
abstract type HypersingularCorrection <: AbstractSingularCorrection end
abstract type SingularCorrection <: AbstractSingularCorrection end

# represents boundary curve
# TODO: rename
abstract type AbstractManifold end

abstract type AbstractSolution{A,T,P} end


#
# includes
#
include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")
include("densities.jl")
include("manifolds.jl")
include("problems.jl")
include("algorithms.jl")
include("operators.jl")
include("kernels.jl")
include("nystroem.jl")
include("assemble.jl")
include("solutions.jl")
include("solve/bvp.jl")
include("solve/bie.jl")
include("solve/cdp.jl")
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

# NOTE: don't like exporting so many names, rather a factory function might be enough
export AbstractDiscretization, Nystroem
export AbstractAlgorithm, AbstractBIEAlgorithm, AbstractBVPAlgorithm,
    AbstractBDPAlgorithm, DirectBIEAlgorithm, IndirectBIEAlgorithm,
    DirectBVPAlgorithm, IndirectBVPAlgorithm,
    DirectBDPAlgorithm, IndirectBDPAlgorithm



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
