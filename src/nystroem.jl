# Operator assembly via Nystroem discretization


# rename: RichardsonExtrapolation; subtype: ZetaQuadrature
struct Sidi <: HypersingularCorrection end
# rename: FiniteDifferences
struct Zeta <: HypersingularCorrection
    order::Int
end
struct KapurRokhlin <: SingularCorrection
    order::Int
end
# for D and D* in 2d with C2 curve
struct AnalyticalExpression <: SingularCorrection end
order(c::Zeta) = c.order
order(c::KapurRokhlin) = c.order
order(::Sidi) = Inf # unsure about this value

# Type alias to help with dispatch
# alternative: make abstract type parametric on E, C, D
const NystroemOperator = Union{
    SingleLayer{<:DifferentialEquation,<:Union{SingularCorrection,Nothing},Nystroem},
    DoubleLayer{<:DifferentialEquation,<:Union{SingularCorrection,Nothing},Nystroem},
    AdjointDoubleLayer{<:DifferentialEquation,<:Union{SingularCorrection,Nothing},Nystroem},
    Hypersingular{<:DifferentialEquation,HypersingularCorrection,Nystroem}
}

# forward call from instances to types
@inline function compute_entry(op::IntegralOperator, args...)
    compute_entry(typeof(op), args...)
end
# forward call to inplace mutating version
@inline function compute_entry!(
    op::NystroemOperator,
    i::Int,
    j::Int,
    args...
)
    op.matrix[i, j] = compute_entry(op, i, j, args...)
end

@inline function make_svector2(matrix, col)
    return @inbounds SVector{2}(matrix[1, col], matrix[2, col])
end

###
#Source-Target interaction
###
@inline function compute_entry(
    op_t::Type{<:SingleLayer{Laplace,Nothing,Nystroem}},
    i::Int,
    j::Int,
    s::DiscreteClosedCurve, # source manifold
    t::AbstractMatrix, # target points
    target_normals, # ignored
)
    x = make_svector2(t, i)
    y = make_svector2(s.x, j)

    return kernel(op_t, x, y) * s.w[j]
end

@inline function compute_entry(
    op_t::Type{<:DoubleLayer{Laplace,Nothing,Nystroem}},
    i::Int,
    j::Int,
    s::DiscreteClosedCurve,
    t::AbstractMatrix,
    target_normals, # ignored
)
    x = make_svector2(t, i) # target point
    y = make_svector2(s.x, j) # source point
    ny = make_svector2(s.n, j) # normal at y

    return kernel(op_t, x, y, ny) * s.w[j] # NOTE: expensive fetch of w[j]
end
@inline function compute_entry(
    op_t::Type{<:AdjointDoubleLayer{Laplace,Nothing,Nystroem}},
    i::Int,
    j::Int,
    s::DiscreteClosedCurve, # outside points in this case
    t::AbstractMatrix, # boundary in this case
    target_normals::AbstractMatrix,
)
    x = make_svector2(t, i) # target point
    y = make_svector2(s.x, j) # source point
    nx = make_svector2(target_normals, i) # normal at x
    return kernel(op_t, x, y, nx) * s.w[j]
end

###
#Self interaction operators
###
#
# NOTE: for now, only inplace, dense is supported. Figure out how to implement
# as matrix-free operators exploiting symmetry

@doc raw"""
    compute_entry!(op::SingleLayer{Laplace,KapurRokhlin}, c::Union{PairwiseCache,Nothing}, i::Int, j::Int, s::DiscreteClosedCurve)

Apply the correction for separable logarithmic singularities

[1] Kapur and Rokhlin, High-Order Corrected Trapezoidal Quadrature Rules for
Singular Functions, 1997

"""
@inline function compute_entry!(
    op::SingleLayer{Laplace,KapurRokhlin,Nystroem},
    i::Int,
    j::Int,
    s::DiscreteClosedCurve,
)

    if j < i
        # lower  triangular sweep
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)

        val = kernel(op, x, y)

        op.matrix[i, j] = val * s.w[j]
        op.matrix[j, i] = val * s.w[i]

    elseif j == i
        #diagonal correction: τ₀ term from [1], eqn. (81), (82), (114)
        op.matrix[i, i] = -0.5 * log(s.w[i]) / π * s.w[i]

    elseif j > i
        return
    end

end

@inline function compute_entry!(
    op::DoubleLayer{Laplace,AnalyticalExpression,Nystroem},
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

# NOTE: why is this float multiplication so expensive? how to make setindex and getindex faster?
        op.matrix[i, j] = kernel(op, x, y, ny) * s.w[j]
    end
end

@inline function compute_entry!(
    op::AdjointDoubleLayer{Laplace,AnalyticalExpression,Nystroem},
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

        op.matrix[i, j] = kernel(op, x, y, nx) * s.w[j]
    end
end

@inline function compute_entry!(
    op::Hypersingular{Laplace,Sidi,Nystroem},
    i::Int,
    j::Int,
    s::DiscreteClosedCurve
)
    if j == i
        # diagonal correction
        op.matrix[i, i] = -π / 4 / s.w[i]

    elseif iseven(i) == iseven(j)
        # checkered pattern zero-out
# NOTE: it's unexpectedly expensive to store a zero
        op.matrix[i, j] = 0.

    elseif j > i
        # skip upper triangular
        return

    else
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        nx = make_svector2(s.n, i)
        ny = make_svector2(s.n, j)

        val = 2 * kernel(op, x, y, nx, ny)

        op.matrix[i, j] = val * s.w[j]
        op.matrix[j, i] = val * s.w[i]
    end
end

# TODO: simplify logic by adding trait is_self_adjoint()

@inline function compute_entry!(
    op::Hypersingular{Laplace,Zeta,Nystroem},
    i::Int,
    j::Int,
    s::DiscreteClosedCurve
)
    if j == i
        #diagonal
        val = -π / 6 / s.w[i] + s.k[i]^2 * s.w[i] / 4π
        op.matrix[i, i] = val

    elseif j > i
        return
    else
        # off-diagonal sweep
        x = make_svector2(s.x, i)
        y = make_svector2(s.x, j)
        nx = make_svector2(s.n, i)
        ny = make_svector2(s.n, j)

        val = kernel(op, x, y, nx, ny)

        op.matrix[i, j] = val * s.w[j]
        op.matrix[j, i] = val * s.w[i]
    end

end

# TODO: stencils should be statically available, caching is redundant
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

@doc raw"""
    apply_correction!(op::SingleLayer{Laplace,KapurRokhlin}, c::StencilCache, i::Int, s::DiscreteClosedCurve)

Apply the correction for separable logarithmic singularities

[1] Kapur and Rokhlin, High-Order Corrected Trapezoidal Quadrature Rules for
Singular Functions, 1997
"""
function apply_correction!(
    op::SingleLayer{Laplace,KapurRokhlin,Nystroem},
    c::StencilCache,
    i::Int,
    s::DiscreteClosedCurve
)
    ord = op.correction.order
    m = size(s, 2)

    #
    k = clamp(
        (ord - 1) ÷ 2,
        0,
        (m - 1) ÷ 2
    )

    stencil = get_kr!(c, k)

    # apply band from [1], eqn. (82)
    for dj in (-k):k
        # TODO: replace by abs
        j = mod1(i + dj, m)
        # divide by 2π = h
        val = stencil[dj+k+1] * 0.5 / pi
        op.matrix[i, j] += val * s.w[j]
    end

end

function apply_correction!(
    op::Hypersingular{Laplace,Zeta,Nystroem},
    c::StencilCache,
    i::Int,
    s::DiscreteClosedCurve
)
    m = size(s, 2)
    ord = op.correction.order
    k = (ord - 2) ÷ 2

    stencil = get_fd!(c, k)

    for dj in (-k):k

        j = mod1(i + dj, m)

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

        nx = make_svector2(s.n, i)
        ny = make_svector2(s.n, j)

        g = dot(nx, ny) * s.w[j] / (s.w[i]^2) * (1 - B + B^2)

        op.matrix[i, j] += stencil[dj+k+1] * g / 4π

    end

end

