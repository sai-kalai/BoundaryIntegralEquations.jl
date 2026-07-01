

# NOTE: is this the julian way for a getter/public api?
function matrix(op::IntegralOperator)::AbstractMatrix
    return op.matrix
end

function Base.:*(op::IntegralOperator, v::AbstractArray)
    return matrix(op) * v
end


function Base.:+(op::IntegralOperator, v::AbstractArray)
    return matrix(op) + v
end

function Base.:*(v::AbstractArray, op::IntegralOperator)
    return op * v
end

function Base.:+(v::AbstractArray, op::IntegralOperator)
    return op + v
end

# a.k.a S
struct SingleLayer{
    P<:BoundaryValueProblem,
    C<:Union{SingularCorrection,Nothing},
    M<:AbstractMatrix{<:Number}, # TODO: change order of members/constructor arguments to match order of generic parameters
} <: IntegralOperator
    problem::P
    correction::C
    matrix::M
end

# a.k.a D a.k.a. ∂S/∂ny
struct DoubleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a  D* a.k.a. ∂S/∂nx
struct AdjointDoubleLayer{
    P<:BoundaryValueProblem,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    problem::P
    matrix::M
end

# a.k.a  N a.k.a. ∂S²/∂nx∂ny
struct Hypersingular{
    P<:BoundaryValueProblem,
    C<:HypersingularCorrection,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    problem::P
    correction::C
    matrix::M
end

# construct empty operator
function SingleLayer(
    problem::P,
    correction::C,
    m::Int, n::Int;
    allocator=(m, n) -> zeros(Float64, m, n) # default to Matrix{Float64}
) where {
    P<:BoundaryValueProblem,
    C<:Union{SingularCorrection,Nothing}
}
    mat = allocator(m, n)
    return SingleLayer(problem, correction, mat)
end

# construct Laplace SLP from a source manifold and a list of target points
function SingleLayer(
    problem::Laplace,
    targets::AbstractMatrix, # target points to compute operator
    boundary::AbstractManifold; # source manifold e.g. domain boundary
    allocator=(m, n) -> zeros(Float64, m, n) # default to Matrix{Float64}
)
    m, dim_t = size(targets)
    n, dim_x = size(boundary.x)

    op = SingleLayer(problem, nothing, m, n; allocator=allocator)

    populate_matrices!(boundary, targets, op)
    # op.matrix .*= boundary.w'

    return op
end

# self interaction
function SingleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    order::Int; # order of kapur rokhlin singular correction
    allocator=(m, n) -> zeros(Float64, m, n) # default to Matrix{Float64}
)
    n, dim_x = size(boundary.x)
    op = SingleLayer(problem, KapurRokhlin(order), n, n; allocator=allocator)

    populate_matrices!(boundary, op)
    # op.matrix .*= boundary.w'

    return op
end

# construct empty operator
function DoubleLayer(
    problem::P,
    m::Int,
    n::Int;
    allocator=(m, n) -> zeros(Float64, m, n) # default to Matrix{Float64}
) where {
    P<:BoundaryValueProblem,
}
    mat = allocator(m, n)
    return DoubleLayer(problem, mat)
end




# construct empty operator
function AdjointDoubleLayer(
    problem::P,
    m::Int,
    n::Int;
    allocator=(m, n) -> zeros(Float64, m, n) # default to Matrix{Float64}
) where {
    P<:BoundaryValueProblem,
}
    mat = allocator(m, n)
    return AdjointDoubleLayer(problem, mat)
end



# construct empty operator
function Hypersingular(
    problem::P,
    correction::C,
    m::Int,
    n::Int;
    allocator=(m, n) -> zeros(Float64, m, n) # default to Matrix{Float64}
) where {
    P<:BoundaryValueProblem,
    C<:HypersingularCorrection
}
    mat = allocator(m, n)
    return Hypersingular(problem, correction, mat)
end




# store data to avoid recomputing
mutable struct PairwiseCache{T<:AbstractFloat}
    r::SVector{2,T}
    r_norm_sq::T
    r_dot_nx::T
    r_dot_ny::T
    nx_dot_ny::T

    function PairwiseCache{T}() where {T<:AbstractFloat}
        cache = new{T}()
        reset!(cache)
        return cache
    end
end

function reset!(cache::PairwiseCache{T}) where {T}
    nan = T(NaN)
    cache.r = SA[nan, nan]
    cache.r_norm_sq = nan
    cache.r_dot_nx = nan
    cache.r_dot_ny = nan
    cache.nx_dot_ny = nan
end

function get_r!(d::PairwiseCache, x::SVector, y::SVector)
    if isnan(d.r[1]) || isnan(d.r[2])
        d.r = x - y
    end
    return d.r
end

function get_r_norm_sq!(d::PairwiseCache, x::SVector, y::SVector)
    if isnan(d.r_norm_sq)
        r = get_r!(d, x, y)
        d.r_norm_sq = dot(r, r)
    end
    return d.r_norm_sq
end

function get_r_dot_nx!(d::PairwiseCache, x::SVector, y::SVector, nx::SVector)
    if isnan(d.r_dot_nx)
        r = get_r!(d, x, y)
        d.r_dot_nx = dot(r, nx)
    end
    return d.r_dot_nx
end

function get_r_dot_ny!(d::PairwiseCache, x::SVector, y::SVector, ny::SVector)
    if isnan(d.r_dot_ny)
        r = get_r!(d, x, y)
        d.r_dot_ny = dot(r, ny)
    end
    return d.r_dot_ny
end

function get_nx_dot_ny!(d::PairwiseCache, nx::SVector, ny::SVector)
    if isnan(d.nx_dot_ny)
        d.nx_dot_ny = dot(nx, ny)
    end
    return d.nx_dot_ny
end




function make_svector2(matrix, row)
    return SVector{2}(matrix[row, 1], matrix[row, 2])
end

# not self interaction
function compute_entry!(
    i::Int,
    j::Int,
    op::SingleLayer{Laplace,Nothing},
    d::PairwiseCache,
    b::DiscreteClosedCurve,
    t::DiscreteClosedCurve, # target points
)

    x = make_svector2(t.x, i)
    y = make_svector2(b.x, j)

    op.matrix[i, j] = kernel(
        op,
        get_r_norm_sq!(d, x, y)
    ) * b.w[j]
end


function compute_entry!(
    i::Int,
    j::Int,
    op::DoubleLayer{Laplace}, # WARN: no explicit indication to separate types corresponding to self vs target interaction
    d::PairwiseCache,
    b::DiscreteClosedCurve,
    t::DiscreteClosedCurve,
)
    x = make_svector2(t.x, i) # target point
    y = make_svector2(b.x, j) # source point
    ny = make_svector2(b.n, j) # normal at y


    op.matrix[i, j] = kernel(
        op,
        get_r_norm_sq!(d, x, y),
        get_r_dot_ny!(d, x, y, ny),
    ) * b.w[j]
end


function compute_entry!(
    # WARN: Massive hack
    i::Int,
    j::Int,
    op::AdjointDoubleLayer{Laplace},
    d::PairwiseCache,
    b::DiscreteClosedCurve, # outside points in this case
    t::DiscreteClosedCurve, # boundary in this case
)
    x = make_svector2(t.x, i) # target point
    y = make_svector2(b.x, j) # source point
    nx = make_svector2(t.n, i) # normal at x


    val = kernel(
        op,
        get_r_norm_sq!(d, x, y),
        dot(x - y, nx)#get_r_dot_nx!(d, x, y, nx),
    )


    op.matrix[i, j] = val * b.w[j]

end


# self interaction
function compute_entry!(
    i::Int,
    j::Int,
    op::SingleLayer{Laplace,KapurRokhlin},
    d::PairwiseCache,
    b::DiscreteClosedCurve
)

    if j < i
        # lower  triangular sweep
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)

        val = kernel(
            op,
            get_r_norm_sq!(d, x, y)
        )

        op.matrix[i, j] = val * b.w[j]
        op.matrix[j, i] = val * b.w[i]

    elseif j == i
        #diagonal
        op.matrix[i, i] = -0.5 * log(b.w[i]) / π * b.w[i]

    elseif j > i
        return

    end

end

function compute_entry!(
    i::Int,
    j::Int,
    op::DoubleLayer{Laplace},
    d::PairwiseCache,
    b::DiscreteClosedCurve
)
    if j == i
        #diagonal
        op.matrix[i, i] = -0.25 * b.k[i] / π * b.w[i]
    else
        # off-diagonal sweep
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        ny = make_svector2(b.n, j)

        op.matrix[i, j] = kernel(
            op,
            get_r_norm_sq!(d, x, y),
            get_r_dot_ny!(d, x, y, ny),
        ) * b.w[j]
    end
end
function compute_entry!(
    i::Int,
    j::Int,
    op::AdjointDoubleLayer{Laplace},
    d::PairwiseCache,
    b::DiscreteClosedCurve
)
    if j == i
        #diagonal
        op.matrix[i, i] = -0.25 * b.k[i] / π * b.w[i]
    else
        # off-diagonal sweep
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        nx = make_svector2(b.n, i)

        op.matrix[i, j] = kernel(
            op,
            get_r_norm_sq!(d, x, y),
            get_r_dot_nx!(d, x, y, nx),
        ) * b.w[j]
    end
end

function compute_entry!(
    i::Int,
    j::Int,
    op::Hypersingular{Laplace,Sidi},
    d::PairwiseCache,
    b::DiscreteClosedCurve
)


    if j == i
        # diagonal correction
        op.matrix[i, i] = -π / 4 / b.w[i]

    elseif iseven(i) == iseven(j)
        # checkered pattern zero-out
        op.matrix[i, j] = 0.

    elseif j > i
        # skip upper triangular
        return

    else
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        nx = make_svector2(b.n, i)
        ny = make_svector2(b.n, j)


        val = 2 * kernel(
            op,
            get_r_norm_sq!(d, x, y),
            get_r_dot_nx!(d, x, y, nx),
            get_r_dot_ny!(d, x, y, ny),
            get_nx_dot_ny!(d, nx, ny),
        )

        op.matrix[i, j] = val * b.w[j]
        op.matrix[j, i] = val * b.w[i]
    end
end

function compute_entry!(
    i::Int,
    j::Int,
    op::Hypersingular{Laplace,Zeta},
    d::PairwiseCache,
    b::DiscreteClosedCurve
)

    if j == i
        #diagonal
        val = -π / 6 / b.w[i] + b.k[i]^2 * b.w[i] / 4π
        op.matrix[i, i] = val

    elseif j > i
        # skipped symmetric
        return

    else
        # off-diagonal sweep
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        nx = make_svector2(b.n, i)
        ny = make_svector2(b.n, j)


        val = kernel(
            op,
            get_r_norm_sq!(d, x, y),
            get_r_dot_nx!(d, x, y, nx),
            get_r_dot_ny!(d, x, y, ny),
            get_nx_dot_ny!(d, nx, ny),
        )

        op.matrix[i, j] = val * b.w[j]
        op.matrix[j, i] = val * b.w[i]
    end

end

# store stencils of possibly several orders
mutable struct StencilCache{
    I<:Integer,
    V<:AbstractVector{<:AbstractFloat}
}
    kr::Dict{I,V} # kapur rokhlin
    fd::Dict{I,V} # finite differences
end



function get_kr!(c::StencilCache, k::Int)
    get!(c.kr, k) do
        stencil = krcoeffs(k + 1)
        [stencil[end:-1:2]; stencil] # TODO: make this prettier
    end
end

function get_fd!(d::StencilCache, k::Int)
    get!(d.fd, k) do
        stencil = fdcoeffs(2, k)
        [stencil[end:-1:2]; stencil] # TODO: make this prettier
    end
end

function apply_correction!(
    i::Int,
    op::IntegralOperator,
    s::StencilCache,
    b::DiscreteClosedCurve,
)
    return
end

function apply_correction!(
    i::Int,
    op::SingleLayer{Laplace,KapurRokhlin},
    s::StencilCache,
    b::DiscreteClosedCurve
)
    ord = op.correction.order
    m = size(b.x, 1)
    k = clamp((ord - 1) ÷ 2, 0, (m - 1) ÷ 2)

    stencil = get_kr!(s, k)

    for dj in (-k):k
        j = mod1(i + dj, m)
        val = stencil[dj+k+1] * 0.5 / pi
        op.matrix[i, j] += val * b.w[j]

    end

end

function apply_correction!(
    i::Int,
    op::Hypersingular{Laplace,Zeta},
    s::StencilCache,
    b::DiscreteClosedCurve
)
    m = size(b.x, 1)
    ord = op.correction.order
    k = (ord - 2) ÷ 2

    stencil = get_fd!(s, k)

    for dj in (-k):k

        j = mod1(i + dj, m)

        # TODO: remove norm
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        r = x - y

        r_norm_sq = dot(r, r)
        r_prime_0_x = b.w[i] * dj

        # B(j) = (r(j) ^ 2 - |ρ'(i) * j| ^ 2) / |ρ'(i) * j| ^ 2
        B = (r_norm_sq - r_prime_0_x^2) / r_prime_0_x^2

        # TODO: remove branch?
        if i == j
            B = 0.
        end

        # TODO: is it possible to use cached data, i.e. call correction inside the first loop?

        # g(j) = n(i) ⋅ n(j) |ρ'(j)|/(2π |ρ'(i)|) * (1 - B + B^2)
        nx_dot_ny = _a_dot_b(b.n[i, 1], b.n[i, 2], b.n[j, 1], b.n[j, 2])

        g = nx_dot_ny * b.w[j] / (b.w[i]^2) * (1 - B + B^2)

        op.matrix[i, j] += stencil[dj+k+1] * g / 4π

    end

end

# barrier function
function populate_matrices!(
    boundary::DiscreteClosedCurve{<:Real},
    ops::IntegralOperator..., # variadic
)
    populate_matrices!(boundary, ops)
end
# so client is responsible for allocating the zeros
# self interaction operators
function populate_matrices!(
    boundary::DiscreteClosedCurve{<:Real},
    ops, # tuple of operators with already allocated matrices
)
    n = size(boundary.x, 1)

    # client provides initialized matrices contained in ops

    # compute required stencils: {KR, FD2}
    #
    #

    stencil_cache = StencilCache{Int32,Vector{Float64}}(Dict(), Dict())

    # if both dlp and dlp adjoint are requested, compute once and transpose before applying weights
    pairwise_cache = PairwiseCache{Float64}()

    # loop over i
    for i in n:-1:1
        for j in 1:n

            # always every pair gets a fresh cache
            reset!(pairwise_cache)
            foreach(ops) do op
                # call appropriate code for each operator kind
                compute_entry!(i, j, op, pairwise_cache, boundary)
            end

        end


        foreach(ops) do op
            apply_correction!(i, op, stencil_cache, boundary)
        end

    end

end

function populate_matrices!(
    boundary::DiscreteClosedCurve{T},
    targets::DiscreteClosedCurve{T},
    ops::IntegralOperator... # only
) where {T<:AbstractFloat}

    populate_matrices!(boundary, targets, ops)

end

# target interaction operators
function populate_matrices!(
    boundary::DiscreteClosedCurve{T},
    targets::DiscreteClosedCurve{T},
    ops # only
) where {T<:AbstractFloat}
    n = size(boundary.x, 1)
    m = size(targets.x, 1)

    pairwise_cache = PairwiseCache{Float64}()
    # loop over i
    #
    # Ideas:
    # - replace double nested loop by cartesian?
    for i in m:-1:1
        for j in 1:n

            # always every pair gets a fresh cache

            # call appropriate code for each operator kind
            reset!(pairwise_cache)
            foreach(ops) do op
                # for op in ops
                compute_entry!(i, j, op, pairwise_cache, boundary, targets)
            end
        end
    end
end



