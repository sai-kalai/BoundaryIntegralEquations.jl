module BimDiff

include("finite_differences.jl")

include("operators.jl")
using .Operators
export
    fdcoeffs,
    LaplaceSLP,
    compute_laplace_slp_matrix,
    compute_laplace_slp_matrix_normal_derivative,
    compute_laplace_slp_matrix_and_normal_derivative,
    compute_laplace_dlp_matrix,
    compute_laplace_dlp_matrix_normal_derivative





include("manifolds.jl")
using .Manifolds
export Manifold, visualize

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
