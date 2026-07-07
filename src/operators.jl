

# NOTE: is this the julian way for a getter/public api?
function matrix(op::IntegralOperator)::AbstractMatrix
    return op.matrix
end

# add scalar to the diagonal
function Base.:+(op::IntegralOperator, s::Number)
    return matrix(op) + I * s
end
function Base.:+(s::Number, op::IntegralOperator)
    return op + s
end
function Base.:-(op::IntegralOperator, s::Number)
    return op + (-1 * s)
end
function Base.:-(s::Number, op::IntegralOperator)
    return op - s
end


# function Base.:*(op::IntegralOperator, v::AbstractArray)
#     return matrix(op) * v
# end
#
# function Base.:+(op::IntegralOperator, v::AbstractArray)
#     return matrix(op) + v
# end
#
#
# function Base.:*(v::AbstractArray, op::IntegralOperator)
#     return op * v
# end
#
# function Base.:+(v::AbstractArray, op::IntegralOperator)
#     return op + v
# end



# a.k.a S
struct SingleLayer{
    E<:DifferentialEquation,
    C<:Union{SingularCorrection,Nothing},
    M<:AbstractMatrix{<:Number}, # TODO: change order of members/constructor arguments to match order of generic parameters
    # TODO: include bitpattern of floating point representation as a type param
} <: IntegralOperator
    equation::E
    correction::C
    matrix::M
end

default_allocator = (_m, _n) -> Matrix{Float64}(undef, _m, _n)

# source-target interaction
function SingleLayer(
    equation::Laplace,
    source::AbstractManifold, # source manifold e.g. domain boundary
    target::AbstractMatrix; # target points to compute operator
    matrix_factory::Function=default_allocator,
)
    m, dim_t = size(target)
    n, dim_x = size(source.x)

    matrix = matrix_factory(m, n)::AbstractMatrix
    op = SingleLayer(equation, nothing, matrix)

    populate_matrices!(source, target, op)
    return op
end

# self interaction
function SingleLayer(
    equation::Laplace,
    source::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    correction::SingularCorrection; # order of kapur rokhlin singular correction
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(source.x)

    matrix = matrix_factory(n, n)::AbstractMatrix # allocate memory
    op = SingleLayer(equation, correction, matrix)

    populate_matrices!(source, op)
    return op
end

# a.k.a D a.k.a. ∂S/∂ny
struct DoubleLayer{
    E<:DifferentialEquation,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    equation::E
    matrix::M
end

# source-target interaction
function DoubleLayer(
    equation::Laplace,
    source::AbstractManifold, # source manifold e.g. domain boundary
    target::AbstractMatrix; # target points to compute operator
    matrix_factory::Function=default_allocator,
)
    m, dim_t = size(target)
    n, dim_x = size(source.x)

    matrix = matrix_factory(m, n)::AbstractMatrix
    op = DoubleLayer(equation, matrix)

    populate_matrices!(source, target, op)
    return op
end

# self interaction
function DoubleLayer(
    equation::Laplace,
    source::AbstractManifold;
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(source.x)
    matrix = matrix_factory(n, n)::AbstractMatrix
    op = DoubleLayer(equation, matrix)

    populate_matrices!(source, op)
    return op
end

# a.k.a  D* a.k.a. ∂S/∂nx
struct AdjointDoubleLayer{
    E<:DifferentialEquation,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    equation::E
    matrix::M
end

# self interaction
function AdjointDoubleLayer(
    equation::Laplace,
    source::AbstractManifold; # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(source.x)
    matrix = matrix_factory(n, n)::AbstractMatrix
    op = AdjointDoubleLayer(equation, matrix)

    populate_matrices!(source, op)
    return op
end

# source-target interaction: edge case for manufactured solution
function AdjointDoubleLayer(
    equation::Laplace,
    source::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    target::AbstractMatrix,
    target_normals::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)

    m, dim_x = size(target)
    n, dim_x = size(source.x)

    matrix = matrix_factory(m, n)::AbstractMatrix
    op = AdjointDoubleLayer(equation, matrix)

    populate_matrices!(source, target, op; target_normals=target_normals)
    return op
end

# a.k.a  N a.k.a. ∂S²/∂nx∂ny
struct Hypersingular{
    E<:DifferentialEquation,
    C<:HypersingularCorrection,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    equation::E
    correction::C
    matrix::M
end

# self interaction
function Hypersingular(
    equation::Laplace,
    source::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    correction::HypersingularCorrection;
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(source.x)

    matrix = matrix_factory(n, n)::AbstractMatrix
    op = Hypersingular(equation, correction, matrix)
    populate_matrices!(source, op)
    return op
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

# TODO: move cache stuff to another file
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
    op::SingleLayer{Laplace,Nothing},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve, # source manifold
    t::AbstractMatrix, # target points
    target_normals, # ignored
)

    x = make_svector2(t, i)
    y = make_svector2(s.x, j)

    op.matrix[i, j] = kernel(
        op,
        get_r_norm_sq!(c, x, y)
    ) * s.w[j]
end


function compute_entry!(
    op::DoubleLayer{Laplace}, # WARN: no explicit indication to separate types corresponding to self vs target interaction
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve,
    t::AbstractMatrix,
    target_normals, # ignored
)
    x = make_svector2(t, i) # target point
    y = make_svector2(s.x, j) # source point
    ny = make_svector2(s.n, j) # normal at y


    op.matrix[i, j] = kernel(
        op,
        get_r_norm_sq!(c, x, y),
        get_r_dot_ny!(c, x, y, ny),
    ) * s.w[j]
end


function compute_entry!(
    op::AdjointDoubleLayer{Laplace},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve, # outside points in this case
    t::AbstractMatrix, # boundary in this case
    target_normals::AbstractMatrix,
)
    x = make_svector2(t, i) # target point
    y = make_svector2(s.x, j) # source point
    nx = make_svector2(target_normals, i) # normal at x


    val = kernel(
        op,
        get_r_norm_sq!(c, x, y),
        dot(x - y, nx)#get_r_dot_nx!(d, x, y, nx),
    )
    op.matrix[i, j] = val * s.w[j]
end


# self interaction
function compute_entry!(
    op::SingleLayer{Laplace,KapurRokhlin},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve,
)

    if j < i
        # lower  triangular sweep
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)

        val = kernel(
            op,
            get_r_norm_sq!(c, x, y)
        )

        op.matrix[i, j] = val * s.w[j]
        op.matrix[j, i] = val * s.w[i]

    elseif j == i
        #diagonal
        op.matrix[i, i] = -0.5 * log(s.w[i]) / π * s.w[i]

    elseif j > i
        return

    end

end

function compute_entry!(
    op::DoubleLayer{Laplace},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve,
)
    if j == i
        #diagonal
        op.matrix[i, i] = -0.25 * s.k[i] / π * s.w[i]
    else
        # off-diagonal sweep
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        ny = make_svector2(s.n, j)

        op.matrix[i, j] = kernel(
            op,
            get_r_norm_sq!(c, x, y),
            get_r_dot_ny!(c, x, y, ny),
        ) * s.w[j]
    end
end

function compute_entry!(
    op::AdjointDoubleLayer{Laplace},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve
)
    if j == i
        #diagonal
        op.matrix[i, i] = -0.25 * s.k[i] / π * s.w[i]
    else
        # off-diagonal sweep
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        nx = make_svector2(s.n, i)

        op.matrix[i, j] = kernel(
            op,
            get_r_norm_sq!(c, x, y),
            get_r_dot_nx!(c, x, y, nx),
        ) * s.w[j]
    end
end

function compute_entry!(
    op::Hypersingular{Laplace,Sidi},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve
)


    if j == i
        # diagonal correction
        op.matrix[i, i] = -π / 4 / s.w[i]

    elseif iseven(i) == iseven(j)
        # checkered pattern zero-out
        op.matrix[i, j] = 0.

    elseif j > i
        # skip upper triangular
        return

    else
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        nx = make_svector2(s.n, i)
        ny = make_svector2(s.n, j)


        val = 2 * kernel(
            op,
            get_r_norm_sq!(c, x, y),
            get_r_dot_nx!(c, x, y, nx),
            get_r_dot_ny!(c, x, y, ny),
            get_nx_dot_ny!(c, nx, ny),
        )

        op.matrix[i, j] = val * s.w[j]
        op.matrix[j, i] = val * s.w[i]
    end
end

function compute_entry!(
    op::Hypersingular{Laplace,Zeta},
    c::PairwiseCache,
    i::Int,
    j::Int,
    s::DiscreteClosedCurve
)

    if j == i
        #diagonal
        val = -π / 6 / s.w[i] + s.k[i]^2 * s.w[i] / 4π
        op.matrix[i, i] = val

    elseif j > i
        # skipped symmetric
        return

    else
        # off-diagonal sweep
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        nx = make_svector2(s.n, i)
        ny = make_svector2(s.n, j)


        val = kernel(
            op,
            get_r_norm_sq!(c, x, y),
            get_r_dot_nx!(c, x, y, nx),
            get_r_dot_ny!(c, x, y, ny),
            get_nx_dot_ny!(c, nx, ny),
        )

        op.matrix[i, j] = val * s.w[j]
        op.matrix[j, i] = val * s.w[i]
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

function get_fd!(c::StencilCache, k::Int)
    get!(c.fd, k) do
        stencil = fdcoeffs(2, k)
        [stencil[end:-1:2]; stencil] # TODO: make this prettier
    end
end

function apply_correction!(
    op::IntegralOperator,
    c::StencilCache,
    i::Int,
    s::DiscreteClosedCurve,
)
    return
end

function apply_correction!(
    op::SingleLayer{Laplace,KapurRokhlin},
    c::StencilCache,
    i::Int,
    s::DiscreteClosedCurve
)
    ord = op.correction.order
    m = size(s.x, 1)
    k = clamp((ord - 1) ÷ 2, 0, (m - 1) ÷ 2)

    stencil = get_kr!(c, k)

    for dj in (-k):k
        j = mod1(i + dj, m)
        val = stencil[dj+k+1] * 0.5 / pi
        op.matrix[i, j] += val * s.w[j]

    end

end

function apply_correction!(
    op::Hypersingular{Laplace,Zeta},
    c::StencilCache,
    i::Int,
    s::DiscreteClosedCurve
)
    m = size(s.x, 1)
    ord = op.correction.order
    k = (ord - 2) ÷ 2

    stencil = get_fd!(c, k)

    for dj in (-k):k

        j = mod1(i + dj, m)

        # TODO: remove norm
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        r = x - y

        r_norm_sq = dot(r, r)
        r_prime_0_x = s.w[i] * dj

        # B(j) = (r(j) ^ 2 - |ρ'(i) * j| ^ 2) / |ρ'(i) * j| ^ 2
        B = (r_norm_sq - r_prime_0_x^2) / r_prime_0_x^2

        # TODO: remove branch?
        if i == j
            B = 0.
        end

        # TODO: is it possible to use cached data, i.e. call correction inside the first loop?

        # g(j) = n(i) ⋅ n(j) |ρ'(j)|/(2π |ρ'(i)|) * (1 - B + B^2)
        nx_dot_ny = _a_dot_b(s.n[i, 1], s.n[i, 2], s.n[j, 1], s.n[j, 2])

        g = nx_dot_ny * s.w[j] / (s.w[i]^2) * (1 - B + B^2)

        op.matrix[i, j] += stencil[dj+k+1] * g / 4π

    end

end

# barrier function
function populate_matrices!(
    source::DiscreteClosedCurve{<:Real},
    ops::IntegralOperator..., # variadic
)
    populate_matrices!(source, ops)
end

# so client is responsible for allocating the zeros
# self interaction operators
function populate_matrices!(
    source::DiscreteClosedCurve{<:Real},
    ops, # tuple of operators with already allocated matrices
)
    n = size(source.x, 1)

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
                compute_entry!(op, pairwise_cache, i, j, source)
            end

        end


        foreach(ops) do op
            apply_correction!(op, stencil_cache, i, source)
        end

    end

end

function populate_matrices!(
    source::DiscreteClosedCurve{T},
    target::AbstractMatrix{T},
    ops::IntegralOperator...; # only
    target_normals::Union{AbstractMatrix{T},Nothing}=nothing, # the AdjointDoubleLayer operator needs the normal vectors at the target locations
) where {T<:AbstractFloat}

    populate_matrices!(source, target, ops, ; target_normals=target_normals)

end

# target interaction operators
function populate_matrices!(
    source::DiscreteClosedCurve{T},
    target::AbstractMatrix{T},
    ops;
    target_normals::Union{AbstractMatrix{T},Nothing}=nothing,
) where {T<:AbstractFloat}


    if any(x -> x isa AdjointDoubleLayer, ops) && isnothing(target_normals)
        error("Requested AdjointDoubleLayer, but provided no unit outward normals at the target points")
    end


    n = size(source.x, 1)
    m = size(target, 1)

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
                compute_entry!(op, pairwise_cache, i, j, source, target, target_normals)
            end
        end
    end
end



