module Manifolds

export AbstractManifold, DiscreteClosedCurve, visualize

using FFTW, LinearAlgebra, GLMakie

abstract type AbstractManifold end

# IDEA:
# specialize the concept of manifold. e.g. geometric manifold has tangents, etc.
# curve, closed curve, 2d, 3d, surface, closed surface, ...

# TODO: maybe set upper bounds as <: AbstractMatrix{<:Number}} for all
struct DiscreteClosedCurve{TX,TV,TA,TS,TT,TN,TK,TW} <: AbstractManifold
    x::TX # locations of points in the manifold
    v::TV # velocities
    a::TA # accelerations
    s::TS # speeds
    t::TT # unit tangential vectors
    n::TN # unit normal vectors
    k::TK # curvatures # TODO: think 2d vs 3d
    w::TW # weights
end


function DiscreteClosedCurve(x, v, a)

    # TODO: assert shapes


    s = sqrt.(sum(abs2, v; dims=2))
    t = v ./ s

    # normal is rotated tangential
    n = similar(t)
    n[:, 1], n[:, 2] = t[:, 2], -t[:, 1]

    k = -sum(a .* n, dims=2) ./ s .^ 2

    N = size(x, 1)

    w = (2π / N) .* s # WARN: discretization in parameter space h is hardcoded here

    return DiscreteClosedCurve(x, v, a, s, t, n, k, w)

end

# construct from number of points and parametrization
# using standard containers
function DiscreteClosedCurve(n_points::Int, ρ::Function)

    # range [0, 2pi) to evaluate parametrization
    θ = range(0, 2π; length=n_points + 1)[1:end-1]
    x = Matrix(stack(ρ, θ)') # TODO: don't transpose, work with column major
    v = periodic_spectral_diff(x)
    a = periodic_spectral_diff(v)

    return DiscreteClosedCurve(x, v, a)

end

function visualize(m::DiscreteClosedCurve)

    fig = Figure()
    ax = Axis(fig[1, 1])

    curve = lines!(ax, m.x[:, 1], m.x[:, 2])
    veloc = arrows2d!(ax, m.x[:, 1], m.x[:, 2], m.n[:, 1], m.n[:, 2],
        color="red",
        lengthscale=0.1,
    )
    accel = arrows2d!(ax, m.x[:, 1], m.x[:, 2], m.t[:, 1], m.t[:, 2],
        color="blue",
        lengthscale=0.1,
    )

    Legend(fig[1, 2],
        [curve, veloc, accel],
        ["curve", "veloc", "accel"],
    )

    return fig

end



"""
    periodic_spectral_diff(d)

periodic spectral derivative

# Arguments
- `f`: matrix containing datapoints along curve
"""
function periodic_spectral_diff(f)


    n = size(f, 1)

    f_hat = fft(f, 1)

    # TODO: replace by fftfreq, fftshift
    if iseven(n)
        k = [0; 1im * (1:n÷2-1); 0; 1im * (-n÷2+1:-1)]
    else
        k = [0; 1im * (1:(n-1)÷2); 1im * (-(n - 1)÷2:-1)]
    end

    f_prime_hat = f_hat .* k

    f_prime = real(ifft(f_prime_hat, 1))

    return f_prime
end


end
