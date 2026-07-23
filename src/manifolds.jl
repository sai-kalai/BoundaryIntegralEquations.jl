abstract type AbstractManifold end # TODO: move to models

# IDEA:
# specialize the concept of manifold. e.g. geometric manifold has tangents, etc.
# curve, closed curve, 2d, 3d, surface, closed surface, ...


function Base.size(m::AbstractManifold, dims...)
    return size(m.x, dims...)
end


# TODO: maybe set upper bounds as <: AbstractMatrix{<:Number}} for all
struct DiscreteClosedCurve{
    T<:Real,
    TX<:AbstractMatrix{<:T},
    TN<:AbstractMatrix{<:T},
    TK<:AbstractVector{<:T}, # scalar
    TW<:AbstractVector{<:T}, # scalar
    CW<:AbstractVector{<:Complex{T}}, # scalar
} <: AbstractManifold
    x::TX # locations of points in the manifold
    n::TN # unit normal vectors
    k::TK # curvatures # TODO: think 2d vs 3d
    w::TW # weights # TODO: enforce that these be vectors
    cw::CW

    function DiscreteClosedCurve(
        x::TX,
        n::TN,
        k::TK,
        w::TW,
        cw::CW,
    ) where {
        T<:Real,
        TX<:AbstractMatrix{<:T},
        TN<:AbstractMatrix{<:T},
        TK<:AbstractVector{<:T},
        TW<:AbstractVector{<:T},
        CW<:AbstractVector{<:Complex{T}},
    }

        d, N = size(x)

        @assert size(n) == (d, N) "normal vectors must have same shape as x"
        @assert length(k) == N "curvature vector must have one entry per point"
        @assert length(w) == N "weight vector must have one entry per point"
        @assert length(cw) == N "complex weight vector must have one entry per point"

        @assert all(isfinite, x) "x contains non-finite values"
        @assert all(isfinite, n) "n contains non-finite values"
        @assert all(isfinite, k) "k contains non-finite values"
        @assert all(isfinite, w) "w contains non-finite values"

        # Optional: enforce unit normals
        # @assert all(abs(norm(n[:, i]) - one(T)) ≤ sqrt(eps(T)) for i in 1:N) "normals must be unit length"

        new{T,TX,TN,TK,TW,CW}(x, n, k, w, cw)
    end
end

function length_scale(c::DiscreteClosedCurve)
    xmin, xmax = extrema(@view c.x[1, :])
    ymin, ymax = extrema(@view c.x[2, :])
    hypot(xmax - xmin, ymax - ymin)
end

function make_dummy_curve(x)

    dim_x, n = size(x)

    one_1d = ones(n)
    zero_nd = zeros((dim_x, n))
    zero_1d = zeros(n)
    zero_cmp=zeros(ComplexF64, n)


    return DiscreteClosedCurve(
        x,
        zero_nd, #n
        zero_1d, #k
        one_1d, #w
        zero_cmp,
    )

end


"""
    DiscreteClosedCurve(x::AbstractMatrix, v::AbstractMatrix, a::AbstractMatrix)

given positions, velocities, and accelerations of the curve parametrization,
compute the remaining parameters

# Arguments
- `x::AbstractMatrix`: location of the points
- `v::AbstractMatrix`: velocity of the curve at each point
- `a::AbstractMatrix`: acceleration of the curve at each point
"""
function DiscreteClosedCurve(x::AbstractMatrix, v::AbstractMatrix, a::AbstractMatrix)

    # TODO: assert shape

    s = vec(sqrt.(sum(abs2, v; dims=1))) # TODO: make this vec() produce a container accordingly to container type of x, v, a

    t = v ./ s' # NOTE: i don't like these transposes that are coming from switching to column-major for enabling bradcasting ...


    # normal is rotated tangential
    n = similar(t)
    n[1, :], n[2, :] = -t[2, :], t[1, :]

    k = vec(-sum(a .* n, dims=1) ./ s' .^ 2)

    N = size(x, 2)

    w = (2π / N) .* s # WARN: discretization in parameter space h is hardcoded here

    # complex weights
    cw = (2π / N) .* ComplexF64.(v[1, :], v[2, :])

    return DiscreteClosedCurve(x, n, k, w, cw)

end

"""
    DiscreteClosedCurve(x::AbstractMatrix)

given positions, compute the velocities and accelerations using periodic spectral differentiation

# Arguments
- `x::AbstractMatrix`: locations of the points
"""
function DiscreteClosedCurve(x::AbstractMatrix)
    v = periodic_spectral_diff(x)
    a = periodic_spectral_diff(v)
    return DiscreteClosedCurve(x, v, a)

end

"""
    DiscreteClosedCurve(θ::AbstractVector, ρ::Function)

construct curve given a list of parameter values and a parametrization

# Arguments
- `θ::AbstractVector`: list of nodes in parameter space
- `ρ::Function`: function that parametrizes the curve
"""
function DiscreteClosedCurve(θ::AbstractVector, ρ::Function)

    # range [0, 2pi) to evaluate parametrization
    x = Matrix(stack(ρ, θ)) # TODO: don't transpose, work with column major

    return DiscreteClosedCurve(x)

end

# construct from number of points and parametrization
# using equispaced parameter
"""
    DiscreteClosedCurve(n_points::Int, ρ::Function)

construct curve given a number of points and a parametrization using equispaced
nodes in parameter space

# Arguments
- `n_points::Int`: number of nodes
- `ρ::Function`: function that parametrizes the curve
"""
function DiscreteClosedCurve(n_points::Int, ρ::Function)
    # range [0, 2pi) to evaluate parametrization
    θ = range(0, 2π; length=n_points + 1)[1:(end-1)]
    return DiscreteClosedCurve(θ, ρ)

end





"""
    periodic_spectral_diff(d)

periodic spectral derivative

# Arguments
- `f`: matrix containing datapoints along curve
"""
function periodic_spectral_diff(f)

    n = size(f, 2)

    f_hat = fft(f, 2)

    # TODO: replace by fftfreq, fftshift
    if iseven(n)
        k = [0; 1im * (1:(n÷2-1)); 0; 1im * ((-n÷2+1):-1)]
    else
        k = [0; 1im * (1:((n-1)÷2)); 1im * ((-(n-1)÷2):-1)]
    end



    f_prime_hat = f_hat .* k'

    f_prime = real(ifft(f_prime_hat, 2))

    return f_prime
end


