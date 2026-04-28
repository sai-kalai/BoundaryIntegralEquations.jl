module Manifolds


export Manifold, visualize


using FFTW, LinearAlgebra, GLMakie

# IDEA:
# specialize the concept of manifold. e.g. geometric manifold has tangents, etc.
# curve, closed curve, 2d, 3d, surface, closed surface, ...

struct Manifold{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    # TODO: all matrices are required to be the same type, is this too restrictive?
    x::M # locations of points in the manifold # TODO: maybe rename to r?
    v::M # velocities
    a::M # accelerations
    s::V # speeds
    t::M # unit tangential vectors
    n::M # unit normal vectors
    k::M # curvatures # TODO: think 2d vs 3d
    w::V # weights


end


function visualize(m::Manifold)

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



function Manifold(x, v, a)

    # TODO: assert shapes


    N = size(x, 1)


    s = norm.(eachrow(v))
    t = v ./ s

    # normal is rotated tangential
    n = hcat(t[:, 2], -t[:, 1])

    k = sum(a .* n, dims=2) ./ s .^ 2

    w = 2pi / N .* s .* ones(N)

    return Manifold{Float64,Matrix{Float64},Vector{Float64}}(x, v, a, s, t, n, k, w)

end

# construct from number of points and parametrization
# using standard containers
function Manifold(n_points::Int, rho::Function)

    # range [0, 2pi) to evaluate parametrization
    theta = (0:n_points-1) * 2pi / n_points


    x = stack(rho, theta)' # TODO: check this #BUG: actually performance sink
    v = periodic_spectral_diff(x)
    a = periodic_spectral_diff(v)

    return Manifold(x, v, a)

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
