abstract type AbstractManifold end # TODO: move to models

# IDEA:
# specialize the concept of manifold. e.g. geometric manifold has tangents, etc.
# curve, closed curve, 2d, 3d, surface, closed surface, ...

# TODO: maybe set upper bounds as <: AbstractMatrix{<:Number}} for all
struct DiscreteClosedCurve{
    T<:Real,
    TX<:AbstractMatrix{<:T},
    TV<:AbstractMatrix{<:T},
    TA<:AbstractMatrix{<:T},
    TS<:AbstractVector{<:T}, # scalar
    TT<:AbstractMatrix{<:T},
    TN<:AbstractMatrix{<:T},
    TK<:AbstractVector{<:T}, # scalar
    TW<:AbstractVector{<:T}, # scalar
    CW<:AbstractVector{<:Complex{T}}, # scalar
} <: AbstractManifold
    x::TX # locations of points in the manifold
    v::TV # velocities
    a::TA # accelerations
    s::TS # speeds
    t::TT # unit tangential vectors
    n::TN # unit normal vectors
    k::TK # curvatures # TODO: think 2d vs 3d
    w::TW # weights # TODO: enforce that these be vectors
    cw::CW
end

# # dummy, TODO: consider an explicit type for set of target points
# function DiscreteClosedCurve(x)
#
#     n, dim_x = size(x)
#
#     one_1d = ones(n)
#     zero_nd = zeros((n, dim_x))
#     zero_1d = zeros(n)
#
#     return DiscreteClosedCurve(
#         x,
#         zero_nd, #v
#         zero_nd, #a
#         zero_1d, #s
#         zero_nd, #t
#         zero_nd, #n
#         zero_1d, #k
#         one_1d, #w
#     )
#
# end

function DiscreteClosedCurve(x::AbstractMatrix, v::AbstractMatrix, a::AbstractMatrix)

    # TODO: assert shape

    s = vec(sqrt.(sum(abs2, v; dims=2))) # TODO: make this vec() produce a container accordingly to container type of x, v, a
    t = v ./ s


    # normal is rotated tangential
    n = similar(t)
    n[:, 1], n[:, 2] = t[:, 2], -t[:, 1]

    k = vec(-sum(a .* n, dims=2) ./ s .^ 2)

    N = size(x, 1)

    w = (2π / N) .* s # WARN: discretization in parameter space h is hardcoded here

    # complex weights
    # cw = (2π / N) .* reinterpret(ComplexF64, v')'
    cw = (2π / N) .* ComplexF64.(v[:, 1], v[:, 2])

    return DiscreteClosedCurve(x, v, a, s, t, n, k, w, cw)

end

function DiscreteClosedCurve(x::AbstractMatrix)
    v = periodic_spectral_diff(x)
    a = periodic_spectral_diff(v)
    return DiscreteClosedCurve(x, v, a)

end

function DiscreteClosedCurve(θ::AbstractVector, ρ::Function)

    # range [0, 2pi) to evaluate parametrization
    x = Matrix(stack(ρ, θ)') # TODO: don't transpose, work with column major

    return DiscreteClosedCurve(x)

end

# construct from number of points and parametrization
# using equispaced parameter
function DiscreteClosedCurve(n_points::Int, ρ::Function)
    # range [0, 2pi) to evaluate parametrization
    θ = range(0, 2π; length=n_points + 1)[1:(end-1)]
    return DiscreteClosedCurve(θ, ρ)

end

function visualize(m::DiscreteClosedCurve)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())

    curve = lines!(ax, m.x[:, 1], m.x[:, 2])

    veloc = arrows2d!(ax, m.x[:, 1], m.x[:, 2], m.n[:, 1], m.n[:, 2],
        color="red",
        lengthscale=0.1,
    )

    accel = arrows2d!(ax, m.x[:, 1], m.x[:, 2], m.t[:, 1], m.t[:, 2],
        color="blue",
        lengthscale=0.1,
    )

    Legend(fig[1, 1][1, 2],
        [curve, veloc, accel],
        ["curve", "normal", "tangent"];
        # tellwidth=false,
        # halign=:left,
        # valign=:bottom,
    )

    return fig, ax

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
        k = [0; 1im * (1:(n÷2-1)); 0; 1im * ((-n÷2+1):-1)]
    else
        k = [0; 1im * (1:((n-1)÷2)); 1im * ((-(n-1)÷2):-1)]
    end

    f_prime_hat = f_hat .* k

    f_prime = real(ifft(f_prime_hat, 1))

    return f_prime
end


