const SV2 = SVector{2}

# forward from instances to types
@inline function kernel(op::IntegralOperator, args...)
    return kernel(typeof(op), args...)
end

@doc raw"""
    kernel(::SingleLayer{Laplace}, r_norm_sq)

Compute the kernel of the Laplace single-layer potential (SLP) integral operator

```math
k_{\text{Lap}}(x, y) = -\frac{1}{2\pi} \log|x - y|
```

# Arguments
- `r_norm_sq`: square magnitude of the displacement vector
"""
@inline function kernel(
    ::Type{<:SingleLayer{Laplace}},
    x::SV2, y::SV2
)
    r = x - y
    r_norm_sq = dot(r, r)
# NOTE: log call is quite expensive
    return -1 / 4pi * log(r_norm_sq) # avoid sqrt: log(√a) = 1/2 log(a)
end


@doc raw"""
    kernel(::AdjointDoubleLayer{Laplace}, r_norm_sq, r_dot_nx)

Compute the kernel of the adjoint of the Laplace double-layer potential (DLP) integral operator, which is
the normal derivative at x of the laplace SLP kernel.

```math
\frac{\partial}{\partial n_x}k_{\text{Lap}}(x, y) = \nabla_x k_{\text{Lap}}(x, y) \cdot n_x = -\frac{1}{2\pi} \frac{(x - y) \cdot n_x}{|x - y|^2}
```

# Arguments
- `r_norm_sq`: square magnitude of the displacement vector
- `r_dot_nx`: dot product between the displacement vector and the normal vector at x
"""
@inline function kernel(::Type{<:AdjointDoubleLayer{Laplace}},
    x::SV2, y::SV2, nx::SV2
)
    r = x - y
    r_norm_sq = dot(r, r)
    r_dot_nx = dot(r, nx)
    return -1 / 2pi * r_dot_nx / r_norm_sq
end

@doc raw"""
    kernel(::DoubleLayer{Laplace}, r_norm_sq, r_dot_ny)

Compute the kernel of the Laplace double-layer potential integral operator, which is
the normal derivative at y of the laplace SLP kernel.

```math
\frac{\partial}{\partial n_y}k_{\text{Lap}}(x, y) = \nabla_y k_{\text{Lap}}(x, y) \cdot n_y= \frac{1}{2\pi} \frac{(x - y) \cdot n_y}{|x - y|^2}
```

# Arguments
- `r_norm_sq`: square magnitude of the displacement vector
- `r_dot_ny`: dot product between the displacement vector and the normal vector at y
"""
@inline function kernel(
    ::Type{<:DoubleLayer{Laplace}},
    x::SV2,
    y::SV2,
    ny::SV2
)
    r = x - y

    r_norm_sq = dot(r, r)
    r_dot_ny = dot(r, ny)

    return 1 / 2pi * r_dot_ny / r_norm_sq
end

@doc raw"""
    kernel(::Hypersingular{Laplace}, r_norm_sq, r_dot_nx, r_dot_ny, nx_dot_ny)


Compute the kernel of the Laplace hypersingular integral operator, which is
the mixed second-order normal derivative of the laplace SLP kernel.

```math
\frac{\partial^2}{\partial n_x \partial n_y}k_{\text{Lap}}(x, y) = \frac{1}{2\pi} \left(\frac{-2((x - y) \cdot n_x) ((x - y) \cdot n_y)}{|x - y|^4} + \frac{n_x \cdot n_y}{|x - y|^2}\right)
```

# Arguments
- `r_norm_sq`: square magnitude of the displacement vector
- `r_dot_nx`: dot product between the displacement vector and the normal vector at x
- `r_dot_ny`: dot product between the displacement vector and the normal vector at y
- `nx_dot_ny`: dot product between the normal vectory at x and the normal vector at y

"""
@inline function kernel(
    ::Type{<:Hypersingular{Laplace}},
    x::SV2,
    y::SV2,
    nx::SV2,
    ny::SV2,
)

    r = x - y

    r_norm_sq = dot(r, r)
    r_dot_nx = dot(r, nx)
    r_dot_ny = dot(r, ny)
    nx_dot_ny = dot(nx, ny)

    return 1 / 2pi * (
        -2 * r_dot_nx * r_dot_ny / (r_norm_sq * r_norm_sq)
        +
        nx_dot_ny / r_norm_sq
    )
end

