

@doc raw"""
    kernel(::SingleLayer{Laplace}, r_norm_sq)

Compute the kernel of the Laplace single-layer potential (SLP) integral operator

```math
k_{\text{Lap}}(x, y) = -\frac{1}{2\pi} \log|x - y|
```

# Arguments
- `r_norm_sq`: square magnitude of the displacement vector
"""
@inline function kernel(::SingleLayer{Laplace}, r_norm_sq)
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
@inline function kernel(::AdjointDoubleLayer{Laplace}, r_norm_sq, r_dot_nx,)
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
@inline function kernel(::DoubleLayer{Laplace}, r_norm_sq, r_dot_ny)
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
@inline function kernel(::Hypersingular{Laplace}, r_norm_sq, r_dot_nx, r_dot_ny, nx_dot_ny)
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




