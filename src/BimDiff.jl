module BimDiff
using Reexport

include("finite_differences.jl")

include("Manifolds.jl")
include("Models.jl")
include("Kernels.jl")
include("Operators.jl")
include("Solvers.jl")

@reexport using .Models

using .Manifolds
export DiscreteClosedCurve, visualize

export fdcoeffs

using .Kernels

using .Operators
export
    fdcoeffs,
    SingleLayer,
    DoubleLayer,
    compute_laplace_slp_matrix,
    compute_laplace_slp_matrix_normal_derivative,
    compute_laplace_slp_matrix_and_normal_derivative,
    compute_laplace_dlp_matrix,
    compute_laplace_dlp_matrix_normal_derivative

using .Solvers
export solve




export greet


greet() = println("Hello World!\n $compute_laplace_slp_matrix")

# trick lsp
@static if false
    # include("../scripts/*.jl")
    include("../scripts/main.jl")
    include("../scripts/precomputed_coeffs.jl")
    include("../scripts/test_lap2d_hyper_bie.jl")
end

end # module BimDiff
