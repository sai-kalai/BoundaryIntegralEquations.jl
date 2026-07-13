
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

# TODO: fix undef allocator
default_allocator = (_m, _n) -> zeros(_m, _n)

# source-target interaction
function SingleLayer(
    equation::Laplace,
    source::AbstractManifold, # source manifold e.g. domain boundary
    target::AbstractMatrix; # target points to compute operator
    matrix_factory::Function=default_allocator,
)
    mat = compute_laplace_slp_matrix(
        target,
        source.x;
        matrix_factory=matrix_factory
    ) .* source.w'
    return SingleLayer(equation, nothing, mat)
end

# self interaction
function SingleLayer(
    equation::Laplace,
    source::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    correction::SingularCorrection; # order of kapur rokhlin singular correction
    matrix_factory::Function=default_allocator,
)
    mat = compute_laplace_slp_matrix(
        source.x,
        source.w,
        correction.order;
        matrix_factory
    ) .* source.w'
    return SingleLayer(equation, correction, mat)

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
    mat = compute_laplace_dlp_matrix(
        target,
        source.x,
        source.n;
        matrix_factory=matrix_factory,
    ) .* source.w'

    return DoubleLayer(equation, mat)

end

# self interaction
function DoubleLayer(
    equation::Laplace,
    source::AbstractManifold;
    matrix_factory::Function=default_allocator,
)

    mat = compute_laplace_dlp_matrix(
        source.x,
        source.n,
        source.k;
        matrix_factory=matrix_factory,
    ) .* source.w'

    return DoubleLayer(equation, mat)
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

    mat = compute_laplace_dlp_adjoint_matrix(
        source.x,
        source.n,
        source.k;
        matrix_factory=matrix_factory,
    ) .* source.w' # TODO: check if weights do go here
    return AdjointDoubleLayer(equation, mat)
end

# source-target interaction: edge case for manufactured solution
function AdjointDoubleLayer(
    equation::Laplace,
    source::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    target::AbstractMatrix,
    target_normals::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)


    mat = compute_laplace_dlp_adjoint_matrix(
        target,
        source.x,
        target_normals,
        ;
        matrix_factory=matrix_factory)


    return AdjointDoubleLayer(equation, mat)

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



# self interaction using zeta correction
function Hypersingular(
    equation::Laplace,
    # TODO: change order to equation -> correction -> source to be more similar to other constructor
    source::AbstractManifold, # differentiate 2d vs 3d here by dispatching on DiscreteClosedCurve vs DiscreteClosedSurface
    correction::Zeta;
    matrix_factory::Function=default_allocator,
)
    mat = compute_laplace_hypersingular_matrix(
        source.x,
        source.n,
        vec(source.k),
        vec(source.w),
        correction.order;
        matrix_factory=matrix_factory,
    )

    # mat[diagind(mat)] .+= -π / 6 ./ source.w + source.k .^ 2 .* source.w ./ 4π

    return Hypersingular(equation, correction, mat,)

end


# self interaction using Sidi correction
function Hypersingular(
    equation::Laplace,
    source::AbstractManifold,
    correction::Sidi;
    matrix_factory::Function=default_allocator,
)
    mat = compute_laplace_hypersingular_matrix(
        source.x,
        source.n;
        matrix_factory,
    ) .* source.w'

    mat[diagind(mat)] .= -pi / 4 ./ source.w

    return Hypersingular(equation, correction, mat)
end


function make_svector2(matrix, row)
    return SVector{2}(matrix[row, 1], matrix[row, 2])
end


function compute_laplace_slp_matrix(
    x, # list of x points (targets)
    y; # list of y points (source, integration variable)
    matrix_factory::Function=default_allocator,
)

    # kernel is symmetric, so source/target distinction is not meaningful??
    # shape of kernel matters though, e.g. apply operator to function acting on source/target points? in this case, function acts on the boundary (source points)
    # but to compute solution at arbitrary points, we are talking about m "target" points
    # so kernel is nxm and is applied to m x ...
    m, dim_x = size(x)
    n, dim_y = size(y)

    mat = matrix_factory(m, n)

    for i in 1:m, j in 1:n
        r = make_svector2(x, i) - make_svector2(y, j)
        mat[i, j] = kernel(
            SingleLayer{Laplace},
            dot(r, r)
        )
    end
    return mat
end

function compute_laplace_slp_matrix(
    y::AbstractMatrix, # list of x points (targets), matrix
    weights::AbstractVector, # list of weights
    order::Int; # quadrature accuracy order
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(y)

    mat = matrix_factory(n, n)

    k = clamp((order - 1) ÷ 2, 0, (n - 1) ÷ 2)

    stencil = krcoeffs(k + 1)
    stencil = [stencil[end:-1:2]; stencil] # TODO: make this prettier


    for i in n:-1:1 # loop backwards

        #diagonal term
        # TODO: consider doing this outside
        mat[i, i] = -log(weights[i]) / 2pi

        for j in 1:(i-1)
            r = make_svector2(y, i) - make_svector2(y, j)
            ker = kernel(
                SingleLayer{Laplace},
                dot(r, r)
            )

            mat[i, j] = ker
            mat[j, i] = ker

        end

        for dj in (-k):k
            j = mod1(i + dj, n)
            mat[i, j] += stencil[dj+k+1] / 2pi
        end


    end
    return mat

end


function compute_laplace_dlp_matrix(
    x::AbstractMatrix,
    y::AbstractMatrix,
    ny::AbstractMatrix; # unitary normals at source
    matrix_factory::Function=default_allocator,
)
    m, dim_x = size(x)
    n, dim_y = size(y)


    mat = matrix_factory(m, n)

    @inbounds for i in 1:m, j in 1:n

        r = make_svector2(x, i) - make_svector2(y, j)
        nyj = make_svector2(ny, j)

        mat[i, j] = kernel(
            DoubleLayer{Laplace},
            dot(r, r),
            dot(r, nyj),
        )
    end
    return mat
end

# self interaction
function compute_laplace_dlp_matrix(
    y::AbstractMatrix,
    ny::AbstractMatrix, # unitary normals at source
    curvatures::AbstractVector;
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(y)

    mat = matrix_factory(n, n)

    @inbounds for i in 1:n

        mat[i, i] = -0.25 / pi * curvatures[i]


        for j in Iterators.flatten((1:(i-1), (i+1):n))
            r = make_svector2(y, i) - make_svector2(y, j)
            nxj = make_svector2(ny, j)

            val = kernel(
                DoubleLayer{Laplace},
                dot(r, r),
                dot(r, nxj),
            )
            mat[i, j] = val

        end

    end
    return mat
end


# not self interaction: special case when making a manufactured solution
function compute_laplace_dlp_adjoint_matrix(
    x,
    y,
    nx;
    matrix_factory::Function=default_allocator,
)
    m, dim_x = size(x)
    n, dim_y = size(y)
    mat = matrix_factory(m, n)

    for i in 1:m, j in 1:n
        xi = make_svector2(x, i)
        yj = make_svector2(y, j)
        nxi = make_svector2(nx, i)
        r = xi - yj

        mat[i, j] = kernel(
            AdjointDoubleLayer{Laplace},
            dot(r, r),
            dot(r, nxi),
        )
    end

    return mat
end

# self interaction
function compute_laplace_dlp_adjoint_matrix(
    y::AbstractMatrix, # points of interest
    ny::AbstractMatrix, # unitary normal vectors at the y points
    curvatures::AbstractVector; # curvature at x
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(y)

    mat = matrix_factory(n, n)

    @inbounds for i in 1:n

        # diagonal limit
        # -1/2 * curvature * 1/2pi
        mat[i, i] = -0.25 / pi * curvatures[i]

        for j in Iterators.flatten((1:(i-1), (i+1):n))
            xi = make_svector2(y, i)
            xj = make_svector2(y, j)
            nxi = make_svector2(ny, i)
            r = xi - xj

            val = kernel(
                AdjointDoubleLayer{Laplace},
                dot(r, r),
                dot(r, nxi),
            )
            mat[i, j] = val
        end
    end
    return mat
end

# self interaction using Sidi's / Richarson's method
function compute_laplace_hypersingular_matrix(
    y::AbstractMatrix,
    ny::AbstractMatrix;
    matrix_factory::Function=default_allocator,
)
    n, dim_x = size(y)
    mat = matrix_factory(n, n)

    for i in 1:n

        # TODO: measure: is it faster to do it here or outside the loop?
        # or leave diagonal empty and let quadrature client handle diagonal
        # dD_dn[i, i] = -pi/4 # NOTE: weights and dirichlet need to be multiplied to diagonal for computing the quadrature

        for j in (mod(i, 2)+1):2:(i-1)

            xi = make_svector2(y, i)
            xj = make_svector2(y, j)
            nxi = make_svector2(ny, i)
            nxj = make_svector2(ny, j)
            r = xi - xj

            # twice weights for staggered grid
            val = 2 * kernel(
                Hypersingular{Laplace},
                dot(r, r),
                dot(r, nxi),
                dot(r, nxj),
                dot(nxi, nxj),
            )

            mat[i, j] = val
            mat[j, i] = val

        end

        # zero-out other entries for supporting undef initializer
        for j in (2-mod(i, 2)):2:(i-1)

            mat[i, j] = 0
            mat[j, i] = 0

        end
    end

    return mat


end

# self interaction using FD correction
function compute_laplace_hypersingular_matrix(
    y::AbstractMatrix,
    ny::AbstractMatrix,
    curvatures::AbstractVector,
    weights::AbstractVector,
    order::Int; # accuracy order
    matrix_factory::Function=default_allocator,
)

    n, dim_x = size(y)

    mat = matrix_factory(n, n)

    # FD stencil for second derivative
    k = (order - 2) ÷ 2

    stencil = fdcoeffs(2, k)

    stencil = [stencil[end:-1:2]; stencil] # TODO: make this prettier

    # `x` in the paper, i.e. for each point in the manifold, each row in the matrix
    for i in n:-1:1

        mat[i, i] = -π / 6 / weights[i] + curvatures[i]^2 * weights[i] / 4π

        # first sum: compute for other points  in the manifold the dlp normal derivative
        for j in 1:(i-1)

            xi = make_svector2(y, i)
            xj = make_svector2(y, j)
            nxi = make_svector2(ny, i)
            nxj = make_svector2(ny, j)
            r = xi - xj

            ker = kernel(
                Hypersingular{Laplace},
                dot(r, r),
                dot(r, nxi),
                dot(r, nxj),
                dot(nxi, nxj),
            )

            # this way asymmetric weighting can be applied
            mat[i, j] = ker * weights[j]
            mat[j, i] = ker * weights[i]

        end

        # apply banded correction
        for dj in (-k):k

            j = mod1(i + dj, n)

            r_norm_sq = norm(y[i, :] - y[j, :])^2
            r_prime_0_x = weights[i] * dj

            # B(j) = (r(j) ^ 2 - |ρ'(i) * j| ^ 2) / |ρ'(i) * j| ^ 2
            B = (r_norm_sq - r_prime_0_x^2) / r_prime_0_x^2

            # TODO: remove branch
            if i == j
                B = 0.
            end

            # g(j) = n(i) ⋅ n(j) |ρ'(j)|/(2π |ρ'(i)|) * (1 - B + B^2)
            nxi = make_svector2(ny, i)
            nxj = make_svector2(ny, j)
            nx_dot_ny = dot(nxi, nxj)

            g = nx_dot_ny * weights[j] / (weights[i]^2) * (1 - B + B^2)

            mat[i, j] += stencil[dj+k+1] * g / 4π

        end



    end


    return mat

end



