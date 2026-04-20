module BimDiff

include("finite_differences.jl")
include("operators.jl")
include("manifolds.jl")

using .Operators, .Manifolds
export fdcoeffs, precomputed_coeffs, LaplaceSLP, Manifold, visualize

export greet

greet() = println("Hello World!\n $laplace_slp")

end # module BimDiff
