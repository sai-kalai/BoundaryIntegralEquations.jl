module CloseEvaluation
using GLMakie

using ..Models
using ..Manifolds
using ..Operators

export cauchy_integral, compute_boundary_limit

"""
    cauchy_integral(target, source::DiscreteClosedCurve, boundary_data)

computes the inner value of a holomorphic function given its boundary data

# Arguments
- `source::DiscreteClosedCurve`: source manifold
- `target`: target points
- `boundary_data`: limit of the function at the curve
"""
function cauchy_integral(
    target,
    source::DiscreteClosedCurve,
    boundary_data
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


"""
    compute_boundary_limit(::Interior, op::SingleLayer{Laplace}, bc::BoundaryCondition)

computes the interior limit of a holomorphic function

# Arguments
- `op::SingleLayer{Laplace}`: [TODO:description]
- `bc::BoundaryCondition`: [TODO:description]
"""
function compute_boundary_limit(
    ::Interior,
    op::DoubleLayer{Laplace},
    density,
    source::DiscreteClosedCurve,
)

    n, dim_y = size(source.x)

    τ = density
    τ_prime = periodic_spectral_diff(τ)

    v_lim = similar(source.x, ComplexF64, n)

    # y = reinterpret(ComplexF64, source.x')
    y = ComplexF64.(source.x[:, 1], source.x[:, 2])

    for k in 1:n

        res = zero(ComplexF64)

        for j in Iterators.flatten((1:(k-1), (k+1):n))
            res += (τ[j] - τ[k]) / (y[j] - y[k]) * source.cw[j]
        end
        v_lim[k] = -τ[k] - τ_prime[k]/(im * n) + res * im / 2pi

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
function compute_boundary_limit(
    ::Exterior,
    op::DoubleLayer{Laplace},
    bc::BoundaryCondition,
    source::DiscreteClosedCurve,
)

    error("not implemented")

end

function compute_boundary_limit(
    ::Interior,
    op::SingleLayer{Laplace},
    bc::BoundaryCondition,
    source::DiscreteClosedCurve,
)

    error("not implemented")

end

function compute_boundary_limit(
    ::Exterior,
    op::SingleLayer{Laplace},
    bc::BoundaryCondition,
    source::DiscreteClosedCurve,
)

    error("not implemented")

end

end
