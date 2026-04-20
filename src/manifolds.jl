module Manifolds

export Manifold, visualize

using FFTW, LinearAlgebra, GLMakie





struct Manifold{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
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

    fig

end


function Manifold{T}(x, v, a) where {T}

    # TODO: assert shapes


    N = size(x, 1)

    s = norm.(eachrow(v))
    t = v ./ s

    # normal is rotated tangential
    n = hcat(-t[:, 2], t[:, 1])

    k = sum(a .* n, dims=2) ./ s.^2

    w = 2pi / N .* s .* ones(N)


    Manifold{T, Matrix{T}, Vector{T}}(x, v, a, s, t, n, k, w)

end

# construct from number of points and parametrization
# using standard containers
function Manifold{T}(n_points::Int, rho::Function) where {T<:Real}

    # range [0, 2pi) to evaluate parametrization
    theta = range(0, stop=2pi, length=n_points)[1:end-1]


    x = stack(rho, theta)'
    v = periodic_spectral_diff(x)
    a = periodic_spectral_diff(v)

    Manifold{T}(x, v, a)

end



# constructors:
# - only points
# - parametrization function and number of points

# construct using only point locations, use periodic spectral diff.
function Manifold{T}(x::Matrix{T}) where {T<:Real}

    Manifold{T, Matrix{T}, Vector{T}}(x, v, a, n, t, k, w)
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
        k = [0; 1im*(1:n÷2-1); 0; 1im*(-n÷2+1:-1)]
    else
        k = [0; 1im*(1:(n-1)÷2); 1im*(-(n-1)÷2:-1)]
    end

    f_prime_hat = f_hat .* k

    f_prime = real(ifft(f_prime_hat, 1))

    return f_prime
end


end
