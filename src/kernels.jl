module Kernels

using LinearAlgebra


# TODO: how can this be made efficient over pairwise lists of vectors?

# laplace single layer potential (SLP) kernel
# k_SLP(x, y) = -1/2pi log(√|x - y|^2)
function laplace_slp(x, y)

    r_norm_sq = _r_norm_sq(x, y)

    return _laplace_slp(r_norm_sq)
end


# normal derivative of the laplace SLP kernel
# ∇_x k_SLP(x, y) · n_x = ∂k_SLP(x, y)/∂n_x = -1/2pi (x - y) ⋅ n_x / |x - y|^2
function laplace_slp_dn(x, y, nx)

    r_norm_sq = 0. # |x - y|^2
    r_dot_nx = 0. # (x - y) ⋅ n_x

    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]

    end

    return _laplace_slp_dn(r_dot_nx, r_norm_sq)
end


# sometimes both quantities are needed at the same time, reuse intermediate computation
function laplace_slp_and_dn(x, y, nx)

    r_norm_sq = 0. # |x - y|^2
    r_dot_nx = 0. # (x - y) ⋅ n_x

    # TODO: maybe just @inline _dot(a, b)? how to avoid allocations?
    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]
    end

    return _laplace_slp(r_norm_sq), _laplace_slp_dn(r_dot_nx, r_norm_sq)

end


# Laplace double layer potential (DLP) kernel
# k_DLP(x, y) = 1/2pi  (x - y) ⋅ n_y / |x - y|^2
function laplace_dlp(
    x, # target points
    y, # source points i.e. manifold/curve
    ny # unitary normals at curve
)

    r_norm_sq = 0. # |x - y|^2
    r_dot_ny = 0. # (x - y) ⋅ n_y

    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_ny += r_k * ny[k]
    end

    return _laplace_dlp(r_dot_ny, r_norm_sq)

end

#  1/2pi (
# -2[(x - y) ⋅ n_x] [(x - y) ⋅ n_y] / |x - y|^4
# + n_x ⋅ n_y / |x - y|^2
#  )
function laplace_dlp_dn(x, y, nx, ny)
    r_norm_sq = 0. # |x - y|^2

    r_dot_nx = 0. # (x - y) ⋅ n_x
    r_dot_ny = 0. # (x - y) ⋅ n_y
    nx_dot_ny = 0. # n_x ⋅ n_y

    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]
        r_dot_ny += r_k * ny[k]
        nx_dot_ny += nx[k] * ny[k]


    end


    return _laplace_dlp_dn(r_dot_nx, r_dot_ny, r_norm_sq, nx_dot_ny)
end

# TODO: benchmark implementation against dot() and norm()

# compute both kernels
function laplace_dlp_and_dn(x, y, nx, ny)

    r_norm_sq = 0. # |x - y|^2

    r_dot_nx = 0. # (x - y) ⋅ n_x
    r_dot_ny = 0. # (x - y) ⋅ n_y
    nx_dot_ny = 0. # n_x ⋅ n_y

    # WARN: these formulas are only valid for 2d anyway...
    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]
        r_dot_ny += r_k * ny[k]
        nx_dot_ny += nx[k] * ny[k]
    end

    return _laplace_dlp(r_dot_nx, r_norm_sq), _laplace_dlp_dn(r_dot_nx, r_dot_ny, r_norm_sq, nx_dot_ny)

end


@inline function _a_dot_b(a, b)
    return a[1] * b[1] + a[2] * b[2]
end

@inline function _r_norm_sq(x, y)
    # in any case allocating here, however cheap?
    r1 = x[1] - y[1]
    r2 = x[2] - y[2]

    return r1 * r1 + r2 * r2
end

@inline function _r_dot_b(x, y, b)
    r1 = x[1] - y[1]
    r2 = x[2] - y[2]
    return r1 * b[1] + r2 * b[2]
end


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

end



