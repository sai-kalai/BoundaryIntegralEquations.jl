module Kernels

using LinearAlgebra


# TODO: how can this be made efficient over pairwise lists of vectors?

# laplace single layer potential (SLP) kernel
# k_SLP(x, y) = -1/2pi log(√|x - y|^2)
function laplace_slp(x, y)

    r_norm_sq = 0. # |x - y|^2

    # loop over dimensions to compute dot products/magnitudes
    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k] # distance along dimension
        r_norm_sq += r_k * r_k
    end

    return -1 / 4pi * log(r_norm_sq) # avoid sqrt: log(√a) = 1/2 log(a)
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

    return -1 / 2pi * r_dot_nx / r_norm_sq
end


# sometimes both quantities are needed at the same time, reuse intermediate computation
function laplace_slp_and_dn(x, y, nx)

    r_norm_sq = 0. # |x - y|^2
    r_dot_nx = 0. # (x - y) ⋅ n_x

    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]
    end

    return -1 / 4pi * log(r_norm_sq), -1 / 2pi * r_dot_nx / r_norm_sq

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

    return 1 / 2pi * r_dot_ny / r_norm_sq

end

# 1/2pi [(x - y) ⋅ n_x] [(x - y) ⋅ n_y] / |x - y|^2
# NOTE: missing term is corrected through diagonal
function laplace_dlp_dn(x, y, nx, ny)
    r_norm_sq = 0. # |x - y|^2

    r_dot_nx = 0. # (x - y) ⋅ n_x
    r_dot_ny = 0. # (x - y) ⋅ n_y

    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]
        r_dot_ny += r_k * ny[k]


    end

    return 1 / 2pi * r_dot_nx * r_dot_ny / (r_norm_sq * r_norm_sq)
end

# compute both kernels
function laplace_dlp_and_dn(x, y, nx, ny)

    r_norm_sq = 0. # |x - y|^2

    r_dot_nx = 0. # (x - y) ⋅ n_x
    r_dot_ny = 0. # (x - y) ⋅ n_y

    @inbounds for k in eachindex(x)
        r_k = x[k] - y[k]

        r_norm_sq += r_k * r_k
        r_dot_nx += r_k * nx[k]
        r_dot_ny += r_k * ny[k]
    end

    dlp = 1 / 2pi * r_dot_ny / r_norm_sq
    dlp_dn = 1 / 2pi * r_dot_nx * r_dot_ny / (r_norm_sq * r_norm_sq)

    return dlp, dlp_dn

end

end



