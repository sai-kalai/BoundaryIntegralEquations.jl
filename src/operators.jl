"""
    matrix(op::IntegralOperator)

retrieve the matrix representation

# Arguments
- `op::IntegralOperator`: operator
"""
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
# TODO: include bitpattern of floating point representation as a type param
@doc raw"""
    SingleLayer


Finite-dimensional approximation of a single-layer potential integral operator
using the Nyström approach.

This operator maps a density on the boundary of a domain to the potential
that it produces.

```math
S: C^1(\Gamma) \to C^1(\bar \Omega)\\

S[u](x) = \int_\Gamma {k(x, y) \varphi(y) dS(y)}, x \in \bar \Omega, y \in \Gamma

```
"""
struct SingleLayer{
    E<:DifferentialEquation,
    C<:Union{SingularCorrection,Nothing},
    D<:AbstractDiscretization,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    equation::E
    correction::C
    discretization::D
    matrix::M
end

const default_allocator = (_m, _n) -> Matrix{Float64}(undef, _m, _n)

# source-target interaction
"""
    SingleLayer(equation::Laplace, source::AbstractManifold, # source manifold e.g. domain boundary, target::AbstractMatrix, # target points to compute operator, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source-target interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined, i.e. boundary of the domain
- `target::AbstractMatrix`: Points where the potential is computed
"""
function SingleLayer(
    equation::Laplace,
    source::AbstractManifold,
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator, # how to reserve memory
    populate_matrix::Bool=true, # should entries be computed
    discretization::AbstractDiscretization=Nystroem(), # what to use to compute entries
)
    m = size(target, 2)
    n = size(source, 2)

    matrix = matrix_factory(m, n)::AbstractMatrix
    op = SingleLayer(equation, nothing,
        discretization,
        matrix,
    )

    if populate_matrix
        populate_matrices!(source, target, op)
    end
    return op
end

# self interaction
@doc raw"""
    SingleLayer(equation::Laplace, source::AbstractManifold, correction::SingularCorrection, # order of kapur rokhlin singular correction, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source self interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined and where the potential is computed, i.e. boundary of the domain
- `correction::SingularCorrection`: Approach to deal with the singularity
"""
function SingleLayer(
    equation::Laplace,
    source::AbstractManifold,
    correction::SingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
    populate_matrix::Bool=false,
    discretization::AbstractDiscretization=Nystroem(),
)

    n = size(source, 2)

    matrix = matrix_factory(n, n)::AbstractMatrix # allocate memory
    op = SingleLayer(equation, correction,
        discretization,
        matrix,
    )

    if populate_matrix
        populate_matrices!(source, op)
    end
    return op
end

# a.k.a D a.k.a. ∂S/∂ny
@doc raw"""
    DoubleLayer

Finite-dimensional approximation of a double-layer potential integral operator
using the Nyström approach.

This operator maps a density on the boundary of a domain to potential
that it produces.

```math
D: C^1(\Gamma) \to C^1(\bar \Omega)\\
D[u](x) = \int_\Gamma {\frac{\partial}{\partial n_y}k(x, y) \varphi(y) dS(y)}, x \in \bar \Omega, y \in \Gamma

```

"""
struct DoubleLayer{
    E<:DifferentialEquation,
    C<:Union{SingularCorrection,Nothing},
    D<:AbstractDiscretization,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    equation::E
    correction::C
    discretization::D
    matrix::M
end

# source-target interaction
@doc raw"""
    DoubleLayer(equation::Laplace, source::AbstractManifold, # source manifold e.g. domain boundary, target::AbstractMatrix, # target points to compute operator, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source-target interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined, i.e. boundary of the domain
- `target::AbstractMatrix`: Points where the potential is computed
"""
function DoubleLayer(
    equation::Laplace,
    source::AbstractManifold, # source manifold e.g. domain boundary
    target::AbstractMatrix,
    ;
    matrix_factory::Function=default_allocator,
    populate_matrix::Bool=true,
    discretization::AbstractDiscretization=Nystroem(),
)
    m = size(target, 2)
    n = size(source, 2)

    matrix = matrix_factory(m, n)::AbstractMatrix
    op = DoubleLayer(equation, nothing,
        discretization,
        matrix,
    )

    if populate_matrix
        populate_matrices!(source, target, op)
    end
    return op
end

# self interaction
@doc raw"""
    DoubleLayer(equation::Laplace, source::AbstractManifold, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source self interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined and where the potential is computed, i.e. boundary of the domain
"""
function DoubleLayer(
    equation::Laplace,
    source::AbstractManifold,
    ;
    matrix_factory::Function=default_allocator,
    populate_matrix::Bool=false,
    discretization::AbstractDiscretization=Nystroem(),
)

    n = size(source, 2)
    matrix = matrix_factory(n, n)::AbstractMatrix
    op = DoubleLayer(equation, AnalyticalExpression(),
        discretization,
        matrix,
    )

    if populate_matrix
        populate_matrices!(source, op)
    end
    return op
end

# a.k.a  D* a.k.a. ∂S/∂nx
@doc raw"""
    AdjointDoubleLayer

Finite-dimensional approximation of the adjoint operator of a double-layer potential integral operator
using the Nyström approach.

This operator maps a density on the boundary of a domain to the potential
that it produces.

```math
D^*: C^1(\Gamma) \to C^1(\bar \Omega) \\
D^*[u](x) = \int_\Gamma {\frac{\partial}{\partial n_x}k(x, y) \varphi(y) dS(y)}, x \in \Gamma, y \in \bar \Omega

```

"""
struct AdjointDoubleLayer{
    E<:DifferentialEquation,
    C<:Union{SingularCorrection,Nothing},
    D<:AbstractDiscretization,
    M<:AbstractMatrix{<:Number}
} <: IntegralOperator
    equation::E
    correction::C
    discretization::D
    matrix::M
end

# self interaction
@doc raw"""
    AdjointDoubleLayer(equation::Laplace, source::AbstractManifold, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source self interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined and where the potential is computed, i.e. boundary of the domain
"""
function AdjointDoubleLayer(
    equation::Laplace,
    source::AbstractManifold,
    ;
    matrix_factory::Function=default_allocator,
    populate_matrix::Bool=false,
    discretization::AbstractDiscretization=Nystroem(),
)
    n = size(source, 2)
    matrix = matrix_factory(n, n)::AbstractMatrix
    op = AdjointDoubleLayer(equation, AnalyticalExpression(),
        discretization,
        matrix,
    )

    if populate_matrix
        populate_matrices!(source, op)
    end
    return op
end

# source-target interaction: edge case for manufactured solution
@doc raw"""
    AdjointDoubleLayer(equation::Laplace, source::AbstractManifold, target::AbstractMatrix, target_normals::AbstractMatrix, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source-target interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined, i.e. points inside the domain
- `target::AbstractMatrix`: Points where the potential is computed, i.e. points in the boundary of the domain
- `target_normals::AbstractMatrix`: outward unit normal vectors at the target points
"""
function AdjointDoubleLayer(
    equation::Laplace,
    source::AbstractManifold,
    target::AbstractMatrix,
    target_normals::Union{AbstractMatrix,Nothing}=nothing,
    ;
    matrix_factory::Function=default_allocator,
    populate_matrix::Bool=false,
    discretization::AbstractDiscretization=Nystroem(),
)

    m = size(target, 2)
    n = size(source, 2)

    matrix = matrix_factory(m, n)::AbstractMatrix
    op = AdjointDoubleLayer(equation, nothing,
        discretization,
        matrix,
    )

    if populate_matrix
        populate_matrices!(source, target, op; target_normals=target_normals)
    end
    return op
end

# a.k.a  N a.k.a. ∂S²/∂nx∂ny
@doc raw"""
    Hypersingular

Finite-dimensional approximation of the hypersingular integral operator
using the Nyström approach.

This operator maps a density on the boundary of a domain to the potential
that it produces.

```math
H: C^1(\Gamma) \to C^1(\Gamma)\\
H[u](x) = \int_\Gamma{\frac{\partial^2}{\partial n_x \partial n_y}k(x, y) \varphi(y) dS(y)}, x\in \Gamma, y \in \Gamma
```

"""
struct Hypersingular{
    E<:DifferentialEquation,
    C<:HypersingularCorrection,
    D<:AbstractDiscretization,
    M<:AbstractMatrix{<:Number},
} <: IntegralOperator
    equation::E
    correction::C
    discretization::D
    matrix::M
end

# self interaction
@doc raw"""
    Hypersingular(equation::Laplace, source::AbstractManifold, correction::HypersingularCorrection, matrix_factory::Function=default_allocator)

Compute the operator coefficients for source self interaction

# Arguments
- `equation::Laplace`: Partial Differential Equation corresponding to this operator
- `source::AbstractManifold`: Manifold where the density is defined and where the potential is computed, i.e. boundary of the domain
- `correction::HypersingularCorrection`: Approach to handle the singularity
"""
function Hypersingular(
    equation::Laplace,
    source::AbstractManifold,
    correction::HypersingularCorrection,
    ;
    matrix_factory::Function=default_allocator,
    populate_matrix::Bool=false,
    discretization::AbstractDiscretization=Nystroem(),
)
    n = size(source, 2)

    matrix = matrix_factory(n, n)::AbstractMatrix
    op = Hypersingular(equation, correction,
        discretization,
        matrix,
    )
    if populate_matrix
        populate_matrices!(source, op)
    end
    return op
end

# TODO: update this for DL
function correction(::IntegralOperator)
    return nothing
end
function correction(op::SingleLayer{E,C}) where {E,C}
    return op.correction
end
function correction(op::Hypersingular{E,C}) where {E,C}
    return op.correction
end




