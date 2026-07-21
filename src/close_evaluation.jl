

"""
    cauchy_integral(target, source::DiscreteClosedCurve, boundary_data)

computes the inner value of a holomorphic function given its boundary data

# Arguments
- `source::DiscreteClosedCurve`: source manifold
- `target`: target points
- `boundary_data`: limit of the function at the curve
"""
function cauchy_integral(
    source::DiscreteClosedCurve,
    target::AbstractMatrix,
    boundary_data::AbstractVector,
)

    m = size(target, 1)
    n = size(source.x, 1)

    v = similar(target, ComplexF64, m)

    # x = reinterpret(ComplexF64, target')
    # y = reinterpret(ComplexF64, source.x')

    x = ComplexF64.(target[:, 1], target[:, 2])
    y = ComplexF64.(source.x[:, 1], source.x[:, 2])


    for k in 1:m
        num = zero(eltype(source.cw))
        den = zero(eltype(source.cw))

        exact_match = false

        for j in 1:n

            if y[j] == x[k]
                v[k] = boundary_data[j]
                exact_match = true
                break
            end

            tmp = source.cw[j] / (y[j] - x[k])
            num += boundary_data[j] * tmp
            den += tmp
        end

        if ! exact_match
            v[k] = num / den
        end
    end
    return v
end


@doc raw"""
    holomorphism_boundary_limit(::Interior, op::SingleLayer{Laplace}, bc::BoundaryCondition)

computes the interior limit $v^-(x), x \in \Gamma$ of the holomorphic function
$v(x), x \in \mathbb C \setminus \Gamma$, from the boundary density $\varphi$
which is the solution to the boundary integral equation by the indirect approach.

The The real part of $v$ is the the double-layer potential of the boundary density,
i.e., it is the solution to the Dirichlet laplace problem.

This means, if $\varphi$ is the solution to

```math
(D - \frac{1}{2})[\varphi] = \sigma

```

then the solution to the interior Laplace problem with Dirichlet data $\sigma$ is

```math
u(x) = Re(v) = D[\varphi](x)
```

and since $v$ is holomorphic, it can be computed by a Cauchy integral, given the
boundary limit of $v$.
```math
v(x) = \frac{1}{2\pi i} \int_{\Gamma}{\frac{v^-(y)}{y - x} dy}
```

This function  computes the boundary limit $v^-(x)$ from the boundary density
$\varphi$ as:

```math
v^-(x) = - \frac{1}{2} \varphi(x) - \frac{1}{2\pi i} \text{p.v.} \int_{\Gamma}{\frac{\varphi(y)}{y-x}dy}

```




# Arguments
- `op::SingleLayer{Laplace}`: Operator that is applied to the density
- `density::BoundaryDensity`: Density that is the solution to the boundary integral equation
- `source::DiscreteClosedCurve`: Boundary $Γ$ of the domain
"""
function holomorphism_boundary_limit(
    ::Interior,
    op::DoubleLayer{Laplace},
    density::BoundaryDensity,
    source::DiscreteClosedCurve,
)

    n, dim_y = size(source.x)

    φ = data(density)
    τ_prime = periodic_spectral_diff(φ)

    v_lim = similar(source.x, ComplexF64, n)

    # y = reinterpret(ComplexF64, source.x')
    y = ComplexF64.(source.x[:, 1], source.x[:, 2])

    for k in 1:n

        res = zero(ComplexF64)

        for j in Iterators.flatten((1:(k-1), (k+1):n))
            res += (φ[j] - φ[k]) / (y[j] - y[k]) * source.cw[j]
        end
        v_lim[k] = -φ[k] - τ_prime[k]/(im * n) + res * im / 2pi

    end
    return v_lim
end

"""
    compute_boundary_limit(::Interior, op::SingleLayer{Laplace}, bc::BoundaryCondition, source::DiscreteClosedCurve)

[TODO:description]

# Arguments
- `op::SingleLayer{Laplace}`: [TODO:description]
- `bc::BoundaryCondition`: [TODO:description]
- `source::DiscreteClosedCurve`: [TODO:description]
"""
function holomorphism_boundary_limit(
    ::Exterior,
    op::DoubleLayer{Laplace},
    bc::BoundaryCondition,
    source::DiscreteClosedCurve,
)

    error("not implemented")

end

function holomorphism_boundary_limit(
    ::Interior,
    op::SingleLayer{Laplace},
    bc::BoundaryCondition,
    source::DiscreteClosedCurve,
)

    error("not implemented")

end

function holomorphism_boundary_limit(
    ::Exterior,
    op::SingleLayer{Laplace},
    bc::BoundaryCondition,
    source::DiscreteClosedCurve,
)

    error("not implemented")

end

