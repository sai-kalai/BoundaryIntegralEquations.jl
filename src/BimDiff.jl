module BimDiff
using Reexport

include("finite_differences.jl")

include("Manifolds.jl")
include("Models.jl")
include("Kernels.jl")
include("Operators.jl")
include("Solvers.jl")
include("close_evaluation.jl")

@reexport using .Models

using .Manifolds
export
    DiscreteClosedCurve,
    visualize,
    periodic_spectral_diff

export fdcoeffs


using .Kernels

using .Operators
export
    fdcoeffs,
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    populate_matrices!,
    compute_laplace_slp_matrix,
    compute_laplace_dlp_adjoint_matrix,
    compute_laplace_slp_matrix_and_normal_derivative,
    compute_laplace_dlp_matrix,
    compute_laplace_hypersingular_matrix

using .Solvers
export
    solve,
    solve_bie


using .CloseEvaluation
export cauchy_integral, compute_boundary_limit


export greet


greet() = println("Hello World!\n $compute_laplace_slp_matrix")

# trick lsp
@static if false
    # include("../scripts/*.jl")
    include("../scripts/main.jl")
    include("../scripts/precomputed_coeffs.jl")
    include("../scripts/test_lap2d_hyper_bie.jl")
    include("../scripts/ellipse.jl")
end

end # module BimDiff
