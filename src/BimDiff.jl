module BimDiff



using CairoMakie
using LinearAlgebra
using StaticArrays
using FFTW



# includes
include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")
include("manifolds.jl")
include("models.jl")
include("kernels.jl")
include("operators.jl")
include("solvers.jl")

# exports
export DiscreteClosedCurve, visualize

export BoundaryValueProblem, Laplace, Helmholtz, Stokes
export HypersingularCorrection, Sidi, Zeta
export SingularCorrection, KapurRokhlin
export IntegralOperator, SingleLayer, DoubleLayer, AdjointDoubleLayer, Hypersingular
export Side, Interior, Exterior
export Approach, Direct, Indirect
export BoundaryCondition, Dirichlet, Neumann

export kernel
export populate_matrices!
export solve



# trick lsp
@static if false
    # include("../scripts/*.jl")
    include("../scripts/main.jl")
    include("../scripts/precomputed_coeffs.jl")
    include("../scripts/test_lap2d_hyper_bie.jl")
end

end # module BimDiff
