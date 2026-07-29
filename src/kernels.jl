



# NOTE: consider passing a "quadrature point" containing access to all the geometric
# data, and cache, instead of separate scalars

# laplace single layer potential (SLP) kernel
# k_SLP(x, y) = -1/2pi log(√|x - y|^2)
@inline function kernel(::Type{<:SingleLayer{Laplace}}, r_norm_sq)
    return -1 / 4pi * log(r_norm_sq) # avoid sqrt: log(√a) = 1/2 log(a)
end

# normal derivative of the laplace SLP kernel a.k.a. adjoint double layer
# ∇_x k_SLP(x, y) · n_x = ∂k_SLP(x, y)/∂n_x = -1/2pi (x - y) ⋅ n_x / |x - y|^2
@inline function kernel(::Type{<:AdjointDoubleLayer{Laplace}}, r_norm_sq, r_dot_nx,)
    return -1 / 2pi * r_dot_nx / r_norm_sq
end

# Laplace double layer potential (DLP) kernel
# k_DLP(x, y) = 1/2pi  (x - y) ⋅ n_y / |x - y|^2
@inline function kernel(::Type{<:DoubleLayer{Laplace}}, r_norm_sq, r_dot_ny)

    return 1 / 2pi * r_dot_ny / r_norm_sq

end

#  1/2pi (
# -2[(x - y) ⋅ n_x] [(x - y) ⋅ n_y] / |x - y|^4
# + n_x ⋅ n_y / |x - y|^2
#  )
@inline function kernel(
    ::Type{<:Hypersingular{Laplace}},
    r_norm_sq, r_dot_nx, r_dot_ny, nx_dot_ny)

    return 1 / 2pi * (
        -2 * r_dot_nx * r_dot_ny / (r_norm_sq * r_norm_sq)
        +
        nx_dot_ny / r_norm_sq
    )
end



function kernel_gradient(
    ::Type{<:SingleLayer{Laplace}},
    x,
    y,
)
    r = x - y
    return r / dot(r, r) / (-2π)
end

function kernel_gradient(
    ::Type{<:DoubleLayer{Laplace}},
    x,
    y,
    ny,
)
    r = x - y
    r_norm_sq = dot(r, r)
    return (I * r_norm_sq - 2 * (r * r')) * ny / r_norm_sq^2 / (2π)
end


function solution_derivative(
    approach::Direct,
    x,
    y,
    ny,
    w,
    τ,
    σ,
)

    grad_u = zero(x)


    for i in axes(x, 2)
        for j in axes(y, 2)
            xi = make_svector2(x, i)
            yj = make_svector2(y, j)
            nyj = make_svector2(ny, j)

            t1 = kernel_gradient(
                SingleLayer{Laplace},
                xi,
                yj
            ) * τ[j]
            t2 = kernel_gradient(
                DoubleLayer{Laplace},
                xi,
                yj,
                nyj) * σ[j]


            grad_u[:, i] += (t1 - t2) * w[j]

        end
    end

    return grad_u

end


