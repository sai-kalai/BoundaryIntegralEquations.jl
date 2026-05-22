module Kernels

using ..Models


struct PairwiseData{T<:Number}
    r_norm_sq::T
    r_dot_nx::T
    r_dot_ny::T
    nx_dot_ny::T
end

function kernel(
    ::IntegralOperator{BoundaryValueProblem},
    PairwiseData{T},
)::T where {T<:Number}

end


# laplace single layer potential (SLP) kernel
# k_SLP(x, y) = -1/2pi log(√|x - y|^2)
@inline function kernel(
    ::SingleLayer{::Laplace},
    d::PairwiseData{T}
)::T where {T<:Number}
    return -1 / 4pi * log(d.r_norm_sq) # avoid sqrt: log(√a) = 1/2 log(a)
end

# normal derivative of the laplace SLP kernel a.k.a. adjoint double layer
# ∇_x k_SLP(x, y) · n_x = ∂k_SLP(x, y)/∂n_x = -1/2pi (x - y) ⋅ n_x / |x - y|^2
@inline function kernel(
    ::AdjointDoubleLayer{::Laplace},
    d::PairwiseData{T},
)::T where T{<:Number}
    return -1 / 2pi * d.r_dot_nx / d.r_norm_sq
end

# Laplace double layer potential (DLP) kernel
# k_DLP(x, y) = 1/2pi  (x - y) ⋅ n_y / |x - y|^2
@inline function kernel(
    ::DoubleLayer{::Laplace},
    d::PairwiseData{T}
)::T where {T<:Number}
    return 1 / 2pi * d.r_dot_ny / d.r_norm_sq

end

#  1/2pi (
# -2[(x - y) ⋅ n_x] [(x - y) ⋅ n_y] / |x - y|^4
# + n_x ⋅ n_y / |x - y|^2
#  )
@inline function kernel(
    ::Hypersingular{::Laplace},
    r_dot_nx,
    r_dot_ny,
    r_norm_sq,
    nx_dot_ny
)
    return 1 / 2pi * (
        -2 * r_dot_nx * r_dot_ny / (r_norm_sq * r_norm_sq)
        +
        nx_dot_ny / r_norm_sq
    )
end

# Loop unrolling for linear algebra operations in 2D
@inline function _a_dot_b(a1, a2, b1, b2)
    return a1 * b1 + a2 * b2
end
@inline function _a_dot_a(a1, a2)
    return a1 * a1 + a2 * a2
end

end



