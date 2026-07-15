module BimDiff



#
# external packages
#
using LinearAlgebra
using StaticArrays
using FFTW
using LinearSolve


#
# type definitions
#
abstract type IntegralOperator end


# TODO: bvp should already be aware of not only the pde, but also type of bc, side of domain
# solve stage should allow choice of approach
abstract type DifferentialEquation end
struct Laplace <: DifferentialEquation end
struct Helmholtz <: DifferentialEquation end
struct Stokes <: DifferentialEquation end




abstract type HypersingularCorrection end
struct Sidi <: HypersingularCorrection end
struct Zeta <: HypersingularCorrection
    order::Int
end

abstract type SingularCorrection end
struct KapurRokhlin <: SingularCorrection
    order::Int
end


abstract type DomainSide end
struct Interior <: DomainSide end
struct Exterior <: DomainSide end

abstract type Approach end
struct Direct <: Approach end
struct Indirect <: Approach end



function spatial_gradient end
function visualize end

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
include("close_evaluation.jl")

include("utils.jl")

#
# exports
#
export DiscreteClosedCurve, visualize, make_dummy_curve

export DifferentialEquation, Laplace, Helmholtz, Stokes
export HypersingularCorrection, Sidi, Zeta
export SingularCorrection, KapurRokhlin
export DomainSide, Interior, Exterior
export IntegralOperator, SingleLayer, DoubleLayer, AdjointDoubleLayer, Hypersingular
export Approach, Direct, Indirect
export BoundaryCondition, Dirichlet, Neumann
export kernel
export cauchy_integral, compute_boundary_limit
export compute_laplace_slp_matrix, compute_laplace_dlp_adjoint_matrix
export BoundaryDensity, BoundaryCondition, Dirichlet, Neumann, data

export kernel
export BoundaryValueProblem, solve, evaluate, solve_and_evaluate

export starfish, ball

export spatial_gradient


function visualize() end


# trick lsp
@static if false
    include("../scripts/main.jl")
    include("../scripts/precomputed_coeffs.jl")

    include("../scripts/ellipse.jl")

    include("../test/quick_test.jl")
    include("../test/convergence/laplace_2d.jl")
    include("../test/test_operators.jl")


end

end # module BimDiff
