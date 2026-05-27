module Operators

using StaticArrays



include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")


using LinearAlgebra


import ..Kernels
using ..Manifolds
using ..Models

export
    matrix,
    populate_matrices!,
    compute_laplace_slp_matrix,
    compute_laplace_slp_matrix_and_normal_derivative,
    compute_laplace_dlp_adjoint_matrix,
    compute_laplace_hypersingular_matrix,
    compute_laplace_dlp_matrix


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


# construct empty operator
function Models.SingleLayer(
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
function Models.SingleLayer(
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
function Models.SingleLayer(
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
function Models.DoubleLayer(
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

function Models.DoubleLayer(
    problem::Laplace,
    target::AbstractMatrix, # target points to compute operator
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    mat = compute_laplace_dlp_matrix(
        target,
        boundary.x,
        boundary.n
    ) .* boundary.w'
    return DoubleLayer(problem, mat)
end

# self interaction
function Models.DoubleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    mat = compute_laplace_dlp_matrix(
        boundary.x,
        boundary.n,
        boundary.k
    ) .* boundary.w'
    return DoubleLayer(problem, mat)
end


# construct empty operator
function Models.AdjointDoubleLayer(
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
# self interaction
function Models.AdjointDoubleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # source manifold e.g. domain boundary
)
    mat = compute_laplace_dlp_adjoint_matrix(
        boundary.x,
        boundary.n,
        boundary.k
    ) .* boundary.w' # TODO: check if weights do go here
    return AdjointDoubleLayer(problem, mat)
end

# construct empty operator
function Models.Hypersingular(
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
# self interaction using zeta correction
function Models.Hypersingular(
    problem::Laplace,
    boundary::AbstractManifold, # TODO: abstract manifold is no good, need to ensure that passed object has n, k, ...
    correction::Zeta,
)
    mat = compute_laplace_hypersingular_matrix(
        boundary.x,
        boundary.n,
        vec(boundary.k),
        vec(boundary.w),
        correction.order,
    )

    mat[diagind(mat)] .+= -π / 6 ./ boundary.w + boundary.k .^ 2 .* boundary.w ./ 4π

    return Hypersingular(problem, correction, mat,)
end

# self interaction using Sidi correction
function Models.Hypersingular(
    problem::Laplace,
    boundary::AbstractManifold,
    correction::Sidi,
)
    mat = compute_laplace_hypersingular_matrix(
        boundary.x,
        boundary.n
    ) .* boundary.w'

    mat[diagind(mat)] .= -pi / 4 ./ boundary.w

    return Hypersingular(problem, correction, mat)
end



function compute_laplace_slp_matrix(
    x, # list of x points (targets)
    y, # list of y points (source, integration variable)
)

    # kernel is symmetric, so source/target distinction is not meaningful??
    # shape of kernel matters though, e.g. apply operator to function acting on source/target points? in this case, function acts on the boundary (source points)
    # but to compute solution at arbitrary points, we are talking about m "target" points
    # so kernel is nxm and is applied to m x ...
    m, dim_x = size(x)
    n, dim_y = size(y)

    A = zeros(Float64, m, n)

    @inbounds for i in 1:m, j in 1:n
        A[i, j] = Kernels.kernel(
            view(x, i, :),
            view(y, j, :)
        )
    end
    return A
end

# store data to avoid recomputing
mutable struct PairwiseCache{T<:AbstractFloat}
    r::SVector{2,T}
    r_norm_sq::T
    r_dot_nx::T
    r_dot_ny::T
    nx_dot_ny::T
end

function PairwiseCache(T)
    nan = T(NaN)
    PairwiseCache(SA[nan, nan], nan, nan, nan, nan)
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

    op.matrix[i, j] = Kernels.kernel(
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


    op.matrix[i, j] = Kernels.kernel(
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


    val = Kernels.kernel(
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

        val = Kernels.kernel(
            op,
            get_r_norm_sq!(d, x, y)
        )

        op.matrix[i, j] += val * b.w[j]
        op.matrix[j, i] += val * b.w[i]

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

        op.matrix[i, j] = Kernels.kernel(
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

        op.matrix[i, j] = Kernels.kernel(
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

    if j > i
        return
    elseif j == i
        op.matrix[i, i] = -π / 4 / b.w[i]

    elseif isodd(i) ⊻ isodd(j)
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        nx = make_svector2(b.n, i)
        ny = make_svector2(b.n, j)


        val = 2 * Kernels.kernel(
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
        op.matrix[i, i] += val

    elseif j > i
        return
    else

        # off-diagonal sweep
        x = make_svector2(b.x, i)
        y = make_svector2(b.x, j)
        nx = make_svector2(b.n, i)
        ny = make_svector2(b.n, j)


        val = Kernels.kernel(
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

    for dj in -k:k
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

    for dj in -k:k

        j = mod1(i + dj, m)

        r_norm_sq = norm(b.x[i, :] - b.x[j, :])^2
        r_prime_0_x = b.w[i] * dj

        # B(j) = (r(j) ^ 2 - |ρ'(i) * j| ^ 2) / |ρ'(i) * j| ^ 2
        B = (r_norm_sq - r_prime_0_x^2) / r_prime_0_x^2

        # TODO: remove branch?
        if i == j
            B = 0.
        end

        # TODO: is it possible to use cached data, i.e. call correction inside the first loop?

        # g(j) = n(i) ⋅ n(j) |ρ'(j)|/(2π |ρ'(i)|) * (1 - B + B^2)
        nx_dot_ny = Kernels._a_dot_b(b.n[i, 1], b.n[i, 2], b.n[j, 1], b.n[j, 2])

        g = nx_dot_ny * b.w[j] / (b.w[i]^2) * (1 - B + B^2)

        op.matrix[i, j] += stencil[dj+k+1] * g / 4π

    end

end

# so client is responsible for allocating the zeros
# self interaction operators
function populate_matrices!(
    boundary::DiscreteClosedCurve,
    ops::IntegralOperator..., # tuple of operators with already allocated matrices
)
    n, dim_x = size(boundary.x)

    # client provides initialized matrices contained in ops

    # compute required stencils: {KR, FD2}
    #
    #

    stencil_cache = StencilCache{Int32,Vector{Float64}}(Dict(), Dict())

    # if both dlp and dlp adjoint are requested, compute once and transpose before applying weights

    # loop over i
    for i in n:-1:1
        for j in 1:n

            # always every pair gets a fresh cache
            pairwise_cache = PairwiseCache(Float64)


            # call appropriate code for each operator kind
            for op in ops
                compute_entry!(i, j, op, pairwise_cache, boundary)
            end

        end

        for op in ops
            apply_correction!(i, op, stencil_cache, boundary)
        end

    end



end


# target interaction operators
function populate_matrices!(
    boundary::DiscreteClosedCurve{T},
    targets::DiscreteClosedCurve{T},
    ops::IntegralOperator..., # only
) where {T<:AbstractFloat}
    n, dim_x = size(boundary.x)
    m, dim_t = size(targets.x)

    # loop over i
    for i in m:-1:1
        for j in 1:n

            # always every pair gets a fresh cache
            pairwise_cache = PairwiseCache(Float64)

            # call appropriate code for each operator kind
            for op in ops
                compute_entry!(i, j, op, pairwise_cache, boundary, targets)
            end
        end
    end
end


# self interaction  using kapur rokhlin
function compute_laplace_slp_matrix(
    x::AbstractMatrix, # list of x points (targets), matrix
    weights::AbstractVector, # list of weights
    order::Int, # quadrature accuracy order
)

    m, dim_x = size(x)

    A = zeros(Float64, m, m)

    k = clamp((order - 1) ÷ 2, 0, (m - 1) ÷ 2)

    stencil = krcoeffs(k + 1)
    stencil = [stencil[end:-1:2]; stencil] # TODO: make this prettier


    for i in m:-1:1 # loop bacwards

        #diagonal term
        # TODO: consider doing this outside
        A[i, i] = -log(weights[i]) / 2pi

        for j in 1:i-1
            ker = Kernels.laplace_slp(
                view(x, i, :),
                view(x, j, :)
            )

            A[i, j] += ker
            A[j, i] += ker

        end

        for dj in -k:k
            j = mod1(i + dj, m)
            A[i, j] += stencil[dj+k+1] / 2pi
        end


    end
    return A
end


# self interaction
function compute_laplace_dlp_adjoint_matrix(
    x::AbstractMatrix, # points of interest
    nx::AbstractMatrix, # unitary normal vectors at the y points
    curvatures::AbstractVector, # curvature at x
)

    m, dim_x = size(x)

    dA_dn = zeros(Float64, m, m)

    @inbounds for i in 1:m

        # diagonal limit
        # -1/2 * curvature * 1/2pi
        dA_dn[i, i] = -0.25 / pi * curvatures[i]

        for j in 1:i-1
            val = Kernels.laplace_slp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )
            dA_dn[i, j] = val
        end

        for j in i+1:m
            val = Kernels.laplace_slp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )
            dA_dn[i, j] = val
        end

    end


    return dA_dn
end

# obsolete-ish
function compute_laplace_slp_matrix_and_normal_derivative(
    x::AbstractMatrix, # points of interest
    y::AbstractMatrix, # domain boundary manifold
    nx::AbstractMatrix, # unitary normal vectors at the y points
)
    m, dim_x = size(x)
    n, dim_y = size(y)

    # TODO: assert shapes

    A = zeros(Float64, m, n)
    dA_dn = zeros(Float64, m, n)

    # zero

    # TODO: s macro broken somehow
    @inbounds for i in 1:m, j in 1:n
        A[i, j], dA_dn[i, j] = Kernels.laplace_slp_and_dn(
            # TODO: maybe better to construct r as svector here
            view(x, i, :),
            view(y, j, :),
            view(nx, i, :)
        )
    end

    # TODO profileview.jl
    # benchmarktools.jl

    return A, dA_dn

end

# self interaction
function compute_laplace_slp_matrix_and_normal_derivative(
    x::AbstractMatrix, # points of interest
    nx::AbstractMatrix, # unitary normal vectors at the y points
    curvatures::AbstractVector, # curvature at x
)
    m, dim_x = size(x)

    # TODO: assert shapes

    A = zeros(Float64, m, m)
    dA_dn = zeros(Float64, m, m)

    @inbounds for i in 1:m

        A[i, i] = 0.
        dA_dn[i, i] = -0.5 * curvatures[i]

        for j in 1:i-1

            slp, slp_dn = Kernels.laplace_slp_and_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )

            A[i, j] = slp
            A[j, i] = slp

            dA_dn[i, j] = slp_dn

        end
        # NOTE: normal derivative isn't symmetric
        for j in i+1:m
            dA_dn[i, j] = Kernels.laplace_slp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :)
            )
        end
    end

    return A, dA_dn

end


function compute_laplace_dlp_matrix(
    x::AbstractMatrix,
    y::AbstractMatrix,
    ny::AbstractMatrix, # unitary normals at source
)
    m, dim_x = size(x)
    n, dim_y = size(y)

    A = zeros(Float64, m, n)

    @inbounds for i in 1:m, j in 1:n
        A[i, j] = Kernels.laplace_dlp(
            view(x, i, :),
            view(y, j, :),
            view(ny, j, :)
        )
    end
    return A
end

# self interaction
function compute_laplace_dlp_matrix(
    x::AbstractMatrix,
    nx::AbstractMatrix, # unitary normals at source
    curvatures::AbstractVector
)

    m, dim_x = size(x)

    D = zeros(Float64, m, m)

    @inbounds for i in 1:m

        D[i, i] = -0.25 / pi * curvatures[i]

        for j in 1:i-1
            val = Kernels.laplace_dlp(
                view(x, i, :),
                view(x, j, :),
                view(nx, j, :) # index with j
            )
            D[i, j] = val
        end
        for j in i+1:m
            val = Kernels.laplace_dlp(
                view(x, i, :),
                view(x, j, :),
                view(nx, j, :)
            )
            D[i, j] = val
        end
    end
    return D
end


# self interaction using Sidi's / Richarson's method
function compute_laplace_hypersingular_matrix(
    x::AbstractMatrix,
    nx::AbstractMatrix,
)

    m, dim_x = size(x)

    dD_dn = zeros(Float64, m, m)

    @inbounds for i in 1:m

        # TODO: measure: is it faster to do it here or outside the loop?
        # or leave diagonal empty and let quadrature client handle diagonal
        # dD_dn[i, i] = -pi/4 # NOTE: weights and dirichlet need to be multiplied to diagonal for computing the quadrature

        for j in (mod(i, 2)+1):2:i-1

            # twice weights for staggered grid
            val = 2 * Kernels.laplace_dlp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :),
                view(nx, j, :),
            )
            dD_dn[i, j] = val
            dD_dn[j, i] = val

        end
    end

    return dD_dn


end

# self interaction using FD correction
function compute_laplace_hypersingular_matrix(
    x::AbstractMatrix,
    nx::AbstractMatrix,
    curvatures::AbstractVector,
    weights::AbstractVector,
    order::Int, # accuracy order
)

    m, dim_x = size(x)

    dD_dn = zeros(Float64, m, m)

    # FD stencil for second derivative
    k = (order - 2) ÷ 2

    stencil = fdcoeffs(2, k)

    stencil = [stencil[end:-1:2]; stencil] # TODO: make this prettier

    # `x` in the paper, i.e. for each point in the manifold, each row in the matrix
    @inbounds for i in m:-1:1

        # dD_dn[i, i] = -π / 6 / weights[i] + curvatures[i]^2 * weights[i] / 4π

        # first sum: compute for other points  in the manifold the dlp normal derivative
        for j in 1:i-1
            ker = Kernels.laplace_dlp_dn(
                view(x, i, :),
                view(x, j, :),
                view(nx, i, :),
                view(nx, j, :),
            )

            # this way asymmetric weighting can be applied
            dD_dn[i, j] = ker * weights[j]
            dD_dn[j, i] = ker * weights[i]

        end

        # apply banded correction
        for dj in -k:k

            j = mod1(i + dj, m)

            r_norm_sq = norm(x[i, :] - x[j, :])^2
            r_prime_0_x = weights[i] * dj

            # B(j) = (r(j) ^ 2 - |ρ'(i) * j| ^ 2) / |ρ'(i) * j| ^ 2
            B = (r_norm_sq - r_prime_0_x^2) / r_prime_0_x^2

            # TODO: remove branch
            if i == j
                B = 0.
            end

            # g(j) = n(i) ⋅ n(j) |ρ'(j)|/(2π |ρ'(i)|) * (1 - B + B^2)
            nx_dot_ny = Kernels._a_dot_b(nx[i, 1], nx[i, 2], nx[j, 1], nx[j, 2])

            g = nx_dot_ny * weights[j] / (weights[i]^2) * (1 - B + B^2)

            dD_dn[i, j] += stencil[dj+k+1] * g / 4π

        end


    end


    return dD_dn
end


end


