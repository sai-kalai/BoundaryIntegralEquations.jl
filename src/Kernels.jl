module Kernels

using ..Models


# function kernels(
#     problem::BoundaryValueProblem,
#     operators::Vector{type(IntegralOperator)},
# )::Vector{<:Number}
#
#     # find out unique needed quantities
#     #
#     # pass to requested kernels
#     #
#     # return requested kernels
#
# end


# laplace single layer potential (SLP) kernel
# k_SLP(x, y) = -1/2pi log(√|x - y|^2)
function laplace_slp(x, y)
    # TODO: should i inline these too?

    r1, r2 = x[1] - y[1], x[2] - y[2]
    r_norm_sq = _a_dot_a(r1, r2)

    return _laplace_slp(r_norm_sq)
end


# normal derivative of the laplace SLP kernel
# ∇_x k_SLP(x, y) · n_x = ∂k_SLP(x, y)/∂n_x = -1/2pi (x - y) ⋅ n_x / |x - y|^2
function laplace_slp_dn(x, y, nx)

    # TODO: still allocating two scalars all over the place...
    r1, r2 = x[1] - y[1], x[2] - y[2]
    r_norm_sq = _a_dot_a(r1, r2) # |x - y|^2
    r_dot_nx = _a_dot_b(r1, r2, nx[1], nx[2]) # (x - y) ⋅ n_x


    return _laplace_slp_dn(r_dot_nx, r_norm_sq)
end


# sometimes both quantities are needed at the same time, reuse intermediate computation
function laplace_slp_and_dn(x, y, nx)
    r1, r2 = x[1] - y[1], x[2] - y[2]

    r_norm_sq = _a_dot_a(r1, r2) # |x - y|^2
    r_dot_nx = _a_dot_b(r1, r2, nx[1], nx[2]) # (x - y) ⋅ n_x

    return _laplace_slp(r_norm_sq), _laplace_slp_dn(r_dot_nx, r_norm_sq)

end


# Laplace double layer potential (DLP) kernel
# k_DLP(x, y) = 1/2pi  (x - y) ⋅ n_y / |x - y|^2
function laplace_dlp(
    x, # target points
    y, # source points i.e. manifold/curve
    ny # unitary normals at curve
)
    r1, r2 = x[1] - y[1], x[2] - y[2]

    r_norm_sq = _a_dot_a(r1, r2) # |x - y|^2
    r_dot_ny = _a_dot_b(r1, r2, ny[1], ny[2]) # (x - y) ⋅ n_y

    return _laplace_dlp(r_dot_ny, r_norm_sq)

end

#  1/2pi (
# -2[(x - y) ⋅ n_x] [(x - y) ⋅ n_y] / |x - y|^4
# + n_x ⋅ n_y / |x - y|^2
#  )
function laplace_dlp_dn(x, y, nx, ny)
    r1, r2 = x[1] - y[1], x[2] - y[2]

    r_norm_sq = _a_dot_a(r1, r2) # |x - y|^2
    r_dot_nx = _a_dot_b(r1, r2, nx[1], nx[2]) # (x - y) ⋅ n_x
    r_dot_ny = _a_dot_b(r1, r2, ny[1], ny[2]) # (x - y) ⋅ n_y
    nx_dot_ny = _a_dot_b(nx[1], nx[2], ny[1], ny[2]) # n_x ⋅ n_y

    return _laplace_dlp_dn(r_dot_nx, r_dot_ny, r_norm_sq, nx_dot_ny)
end

# TODO: benchmark implementation against dot() and norm()

# compute both kernels
function laplace_dlp_and_dn(x, y, nx, ny)
    r1, r2 = x[1] - y[1], x[2] - y[2]

    r_norm_sq = _a_dot_a(r1, r2) # |x - y|^2

    r_dot_nx = _a_dot_b(r1, r2, nx[1], nx[2]) # (x - y) ⋅ n_x
    r_dot_ny = _a_dot_b(r1, r2, ny[1], ny[2]) # (x - y) ⋅ n_y
    nx_dot_ny = _a_dot_b(nx[1], nx[2], ny[1], ny[2])  # n_x ⋅ n_y

    return _laplace_dlp(r_dot_nx, r_norm_sq), _laplace_dlp_dn(r_dot_nx, r_dot_ny, r_norm_sq, nx_dot_ny)

end




# actual kernel computations, operating on scalars
@inline function _laplace_slp(r_norm_sq)
    return -1 / 4pi * log(r_norm_sq) # avoid sqrt: log(√a) = 1/2 log(a)
end

@inline function _laplace_slp_dn(r_dot_nx, r_norm_sq)
    return -1 / 2pi * r_dot_nx / r_norm_sq
end

@inline function _laplace_dlp(r_dot_ny, r_norm_sq)
    return 1 / 2pi * r_dot_ny / r_norm_sq

end

@inline function _laplace_dlp_dn(r_dot_nx, r_dot_ny, r_norm_sq, nx_dot_ny)
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



