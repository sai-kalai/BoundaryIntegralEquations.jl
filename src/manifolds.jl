
# IDEA:
# specialize the concept of manifold. e.g. geometric manifold has tangents, etc.
# curve, closed curve, 2d, 3d, surface, closed surface, ...

using BoundaryIntegralEquations: make_svector2

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
    n[1, :], n[2, :] = t[2, :], -t[1, :]

    k = vec(-sum(a .* n, dims=1) ./ s' .^ 2)

    N = size(x, 2)

    w = (2π / N) .* s # WARN: discretization in parameter space h is hardcoded here

    # complex weights
    # orientation of surface was flipped?
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
    x = Matrix(stack(ρ, θ))

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

@doc raw"""
    make_dummy_curve(x)

Construct storing only the node locations, using unit weights and no information
normals, curvatures, or complex weights. Used for computing manufactured solutions.

# Arguments
- `x`: locations of the nodes
"""
function make_dummy(x)
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


@doc raw"""
    make_offset(c::DiscreteClosedCurve, distance::Real)

Construct a parallel curve by displacing the vertices in the normal direction

# Arguments
- `c::DiscreteClosedCurve`: base curve
- `distance::Real`: distance to move the curve (positive number for inflation,
negative number for deflation)
"""
function make_offset(c::DiscreteClosedCurve, distance::Real)

    x = similar(c.x)
    @views for col in axes(c.x, 2)
        x[:, col] = c.x[:, col] + distance * c.n[:, col]
    end


    return DiscreteClosedCurve(x)

end


function Base.show(io::IO, ::MIME"text/plain", c::DiscreteClosedCurve{T}) where {T}
    print(io, "DiscreteClosedCurve{", T, "...} with ", size(c, 2), " nodes")
end
function Base.show(io::IO, ::Type{<:DiscreteClosedCurve{T}}) where {T}
    print(io, "DiscreteClosedCurve{", T, "...}")
end


@doc raw"""
    Base.extrema(c::DiscreteClosedCurve)

compute the minimum and maximum of x and y coordinates of the curve

# Arguments
- `c::DiscreteClosedCurve`: curve to compute the extrema
"""
function Base.extrema(c::DiscreteClosedCurve)
    (xmin, xmax), (ymin, ymax) = extrema(c.x, dims=2)
    return xmin, xmax, ymin, ymax
end

@doc raw"""
    length_scale(c::DiscreteClosedCurve)

compute the diagonal length of the axis-aligned rectangle where the curve is
inscribed. This gives an idea of the characteristic length of the domain.

# Arguments
- `c::DiscreteClosedCurve`: curve to compute the characteristic length
"""
function lengthscale(c::DiscreteClosedCurve)
    xmin, xmax, ymin, ymax = extrema(c)
    hypot(xmax - xmin, ymax - ymin)
end


@doc raw"""
    polygon(c::DiscreteClosedCurve)

Construct a polgon compatible with PolygonOps.jl

# Arguments
- `c::DiscreteClosedCurve`: Curve that defines the polygon
"""
function polygon(c::DiscreteClosedCurve)
    # TODO: attempt to avoid allocating here
    points = [(col[1], col[2]) for col in eachcol(c.x)]
    # close loop, since PolygonOps requires that the first and last points be
    # the same
    push!(points, points[1])

    return points
end


@doc raw"""
    mask(c::DiscreteClosedCurve, x::AbstractMatrix, s::DomainSide)

Compute a boolean mask to decide if points are in the correct side

# Arguments
- `c::DiscreteClosedCurve`: Boundary of the domain
- `x::AbstractMatrix`: Points, stored in a column-major matrix of size (2, N)
- `s::DomainSide`: Decide to set `true` for inner or outer points
"""
function mask(c::DiscreteClosedCurve, x::AbstractMatrix, s::DomainSide,)
    poly = polygon(c)
    hit = s isa Interior ? 1 : 0
    return [
        inpolygon((x[1, row], x[2, row]), poly) == hit
        for row in axes(x, 2)
    ]
end


@doc raw"""
    KDTree(c::DiscreteClosedCurve)

Construct a 2d tree for fast geometric operations on the nodes of a boundary

# Arguments
- `c::DiscreteClosedCurve`: Boundary of the domain
"""
function NearestNeighbors.KDTree(c::DiscreteClosedCurve)
    # TODO: cache tree inside boundary
    return KDTree(c.x)
end
function NearestNeighbors.BallTree(c::DiscreteClosedCurve)
    return BallTree(c.x)
end

@doc raw"""
    mask(c::DiscreteClosedCurve, x::AbstractMatrix, d::Real)

Compute a boolean mask to decide if points in `x` are within a given distance of the
boundary `c`

# Arguments
- `c::DiscreteClosedCurve`: Boundary of the domain
- `x::AbstractMatrix`: Points, stored in a column-major matrix of size (2, N)
- `d::Real`: Distance to boundary
"""
function mask(c::DiscreteClosedCurve, x::AbstractMatrix, d::Real)
    tree = KDTree(c)
    inside_cutoff_idxs = inrange(tree, x, d)
    inside_cutoff_mask = .!isempty.(inside_cutoff_idxs)
    return inside_cutoff_mask
end


@doc raw"""
    findall(c::DiscreteClosedCurve, x::AbstractMatrix, d::Real)

finds the indices of points within a distance of the boundary

# Arguments
- `c::DiscreteClosedCurve`: boundary of the domain
- `x::AbstractMatrix`: point data as a matrix of size `2xm`
- `d::Real`: distance
"""
function Base.findall(c::DiscreteClosedCurve, x::AbstractMatrix, d::Real, strategy=:tree)
    if strategy == :tree
        tree = KDTree(c)
        idxs = inrangecount(tree, x, d) .|> !iszero |> findall
        return idxs
    elseif strategy == :brute
        idxs = Int[]
        sizehint!(idxs, size(x, 2))
        for j in axes(x, 2), i in axes(c.x, 2)
            if norm(make_svector2(c.x, i) - make_svector2(x, j)) <= d
                push!(idxs, j)
            end
        end
        return idxs
    else
        error("Expected $strategy to be one of (`:tree`, `:brute`)")
    end

end

@doc raw"""
    classify(c::DiscreteClosedCurve, x::AbstractMatrix, s::DomainSide, d::Real)

classifies query points, returning vectors of indices of points in each region

# Arguments
- `c::DiscreteClosedCurve`: boundary of the domain
- `x::AbstractMatrix`: point data as a matrix of size `2xn`
- `s::DomainSide`: [TODO:description]
- `d::Real`: non-negative cutoff distance to discern near and far points
# Returns
- vectors of indices of points laying on:
    - near boundary
    - far from boundary
    - outside of valid region (bad points)
"""
function classify(
    c::DiscreteClosedCurve, x::AbstractMatrix, s::DomainSide, d::Real=0.)

    # allocate enough space for storing indices
    m = size(x, 2)
    near_idxs = sizehint!(Int[], m)
    far_idxs = sizehint!(Int[], m)
    bad_idxs = sizehint!(Int[], m)

    # construct polygon for domain side check
    poly = polygon(c)
    hit = s isa Interior ? 1 : 0

    @assert d >= 0

    # construct tree for in range query only if d > 0
    if iszero(d)
        tree = nothing
    else
        tree = KDTree(c)
    end


    # classify every query point accordingly
    for i in axes(x, 2)
        xi = make_svector2(x, i)

        if !(inpolygon(xi, poly) == hit)
            # not correct side of domain
            push!(bad_idxs, i)
        else
            # correct side of domain
            if iszero(d)
                # if cutoff distance is zero, all points are "far"
                push!(far_idxs, i)
            elseif iszero(inrangecount(tree, xi, d))
                push!(far_idxs, i)
            else
                push!(near_idxs, i)
            end
        end
    end

    return near_idxs, far_idxs, bad_idxs
end

@doc raw"""
    findall(c::DiscreteClosedCurve, x::AbstractMatrix, s::DomainSide)

finds the indices of points that lie in the correct slide of the domain

# Arguments
- `c::DiscreteClosedCurve`: boundary of the domain
- `x::AbstractMatrix`: point data as a matrix of size `2xm`
- `s::DomainSide`: interior or exterior domain
"""
function Base.findall(c::DiscreteClosedCurve, x::AbstractMatrix, s::DomainSide)
    poly = polygon(c)
    hit = s isa Interior ? 1 : 0
    return [
        row
        for row in axes(x, 2)
        if inpolygon((x[1, row], x[2, row]), poly) == hit
    ]
end

# TODO: this doesn't belong to manifolds
@doc raw"""
    periodic_spectral_diff(d)

periodic spectral derivative

# Arguments
- `f`: matrix containing datapoints along curve
"""
function periodic_spectral_diff(f)

    dim = ndims(f)

    n = size(f, dim)
    eachslice

    f_hat = fft(f, dim)

    # TODO: replace by fftfreq, fftshift
    if iseven(n)
        k = [0; 1im * (1:(n÷2-1)); 0; 1im * ((-n÷2+1):-1)]
    else
        k = [0; 1im * (1:((n-1)÷2)); 1im * ((-(n-1)÷2):-1)]
    end

    # HACK:
    # fix later
    # make k broadcast correctly along the differentiation dimension
    if dim == 1
        f_prime_hat = f_hat .* k
    elseif dim == 2
        f_prime_hat = f_hat .* transpose(k) # NOTE: this mysterious minus appeared after changing to col-major
    # NOTE: the culprit was the ' operator which gives the adjoing in the case of
    # complex valued vectors... transpose solves this
    else
        error("dimension $dim not supported")
    end

    # TODO:: try writing loop
    # work out why the normal is flipped
    # loop over dimensions performing 1D fft on each instead of passing matrix


    f_prime = real(ifft(f_prime_hat, dim))


    return f_prime
end


