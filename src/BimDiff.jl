module BimDiff


#
# external packages
#
using CairoMakie
using LinearAlgebra
using StaticArrays
using LinearSolve
using FFTW

#
# type definitions
#
abstract type IntegralOperator end

# TODO: bvp should already be aware of not only the pde, but also type of bc, side of domain
# solve stage should allow choice of approach
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


#
# includes
#
include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")
include("manifolds.jl")
include("operators.jl")
include("kernels.jl")
include("solvers.jl")
include("close_evaluation.jl")


#
# exports
#
export DiscreteClosedCurve, visualize

export BoundaryValueProblem, Laplace, Helmholtz, Stokes
export HypersingularCorrection, Sidi, Zeta
export SingularCorrection, KapurRokhlin
export Side, Interior, Exterior
export IntegralOperator, SingleLayer, DoubleLayer, AdjointDoubleLayer, Hypersingular
export Approach, Direct, Indirect
export BoundaryCondition, Dirichlet, Neumann
export kernel
export solve, solve_bie
export cauchy_integral, compute_boundary_limit
export compute_laplace_slp_matrix, compute_laplace_dlp_adjoint_matrix

# trick lsp
@static if false
    # include("../scripts/*.jl")
    include("../scripts/main.jl")
    include("../scripts/precomputed_coeffs.jl")
    include("../scripts/test_lap2d_hyper_bie.jl")
    include("../scripts/ellipse.jl")
end

end # module BimDiff
