
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


# construct Laplace SLP from a source manifold and a list of target points
function SingleLayer(
    problem::Laplace,
    targets::AbstractMatrix, # target points to compute operator
    boundary::AbstractManifold; # source manifold e.g. domain boundary
)
    mat = compute_laplace_slp_matrix(targets, boundary.x) .* boundary.w'
    return SingleLayer(problem, nothing, mat)

end

# self interaction
function SingleLayer(
    problem::Laplace,
    boundary::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    order::Int; # order of kapur rokhlin singular correction
)

    mat = compute_laplace_slp_matrix(boundary.x, boundary.w, order) .* boundary.w'

    return SingleLayer(problem, KapurRokhlin(order), mat)

end


# self interaction
function DoubleLayer(
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

# following operators are only self-interaction because they require normal derivative at x

# self interaction
function AdjointDoubleLayer(
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

# self interaction using zeta correction
function Hypersingular(
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

 function DoubleLayer(
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

# self interaction using Sidi correction
function Hypersingular(
    problem::Laplace,
    boundary::AbstractManifold,
    correction::Sidi,
)
    mat = compute_laplace_hypersingular_matrix(
        boundary.x,
        boundary.n
    ) .* boundary.w'

    # mat[diagind(mat)] .= -pi / 4 ./ boundary.w

    return Hypersingular(problem, correction, mat)
end



function make_svector2(matrix, row)
    return SVector{2}(matrix[row, 1], matrix[row, 2])
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
        r = make_svector2(x, i) - make_svector2(y, j)
        A[i, j] = kernel(
            SingleLayer{Laplace},
            dot(r, r)
        )
    end
    return A
end

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

        for j in 1:(i-1)
            r = make_svector2(x, i) - make_svector2(x, j)
            ker = kernel(
                SingleLayer{Laplace},
                dot(r, r)
            )

            A[i, j] += ker
            A[j, i] += ker

        end

        for dj in (-k):k
            j = mod1(i + dj, m)
            A[i, j] += stencil[dj+k+1] / 2pi
        end


    end
    return A

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

        r = make_svector2(x, i) - make_svector2(y, j)
        nyj = make_svector2(ny, j)

        A[i, j] = kernel(
            DoubleLayer{Laplace},
            dot(r, r),
            dot(r, nyj),
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

        for j in Iterators.flatten((1:(i-1), (i+1):m))
            r = make_svector2(x, i) - make_svector2(x, j)
            nxj = make_svector2(nx, j)

            val = kernel(
                DoubleLayer{Laplace},
                dot(r, r),
                dot(r, nxj),
            )
            D[i, j] = val
        end

    end
    return D
end


# not self interaction: special case when making a manufactured solution

function compute_laplace_dlp_adjoint_matrix(
    x,
    y,
    nx
)
    m, dim_x = size(x)
    n, dim_y = size(y)
    dA_dn = zeros(Float64, m, n)

    for i in 1:m, j in 1:n
        xi = make_svector2(x, i)
        yj = make_svector2(y, j)
        nxi = make_svector2(nx, i)
        r = xi - yj

        dA_dn[i, j] = kernel(
            AdjointDoubleLayer{Laplace},
            dot(r, r),
            dot(r, nxi),
        )
    end

    return dA_dn
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

        for j in Iterators.flatten((1:(i-1), (i+1):m))
            xi = make_svector2(x, i)
            xj = make_svector2(x, j)
            nxi = make_svector2(nx, i)
            r = xi - xj

            val = kernel(
                AdjointDoubleLayer{Laplace},
                dot(r, r),
                dot(r, nxi),
            )
            dA_dn[i, j] = val
        end
    end
    return dA_dn
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
        dD_dn[i, i] = -pi/4 # NOTE: weights and dirichlet need to be multiplied to diagonal for computing the quadrature

        for j in (mod(i, 2)+1):2:(i-1)

            xi = make_svector2(x, i)
            xj = make_svector2(x, j)
            nxi = make_svector2(nx, i)
            nxj = make_svector2(nx, j)
            r = xi - xj

            # twice weights for staggered grid
            val = 2 * kernel(
                Hypersingular{Laplace},
                dot(r, r),
                dot(r, nxi),
                dot(r, nxj),
                dot(nxi, nxj),
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
        for j in 1:(i-1)

            xi = make_svector2(x, i)
            xj = make_svector2(x, j)
            nxi = make_svector2(nx, i)
            nxj = make_svector2(nx, j)
            r = xi - xj

            ker = kernel(
                Hypersingular{Laplace},
                dot(r, r),
                dot(r, nxi),
                dot(r, nxj),
                dot(nxi, nxj),
            )

            # this way asymmetric weighting can be applied
            dD_dn[i, j] = ker * weights[j]
            dD_dn[j, i] = ker * weights[i]

        end

        # apply banded correction
        for dj in (-k):k

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
            nxi = make_svector2(nx, i)
            nxj = make_svector2(nx, j)
            nx_dot_ny = dot(nxi, nxj)

            g = nx_dot_ny * weights[j] / (weights[i]^2) * (1 - B + B^2)

            dD_dn[i, j] += stencil[dj+k+1] * g / 4π

        end


    end


    return dD_dn
end



