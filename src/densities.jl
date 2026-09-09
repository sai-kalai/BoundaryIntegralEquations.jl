
function data(density::BoundaryDensity)
    density.φ
end
function data(density::Dirichlet)
    density.σ
end
function data(density::Neumann)
    density.τ
end

# apply an operator to a density
function Base.:*(A::IntegralOperator, φ::AbstractBoundaryDensity)
    return matrix(A) * data(φ)
end
# matrix-free application of operator
function apply(
    op_t::Type{<:IntegralOperator},
    source::AbstractManifold,
    target::AbstractMatrix,
    density::AbstractBoundaryDensity
)

    n = size(source, 2)
    m = size(target, 2)

    u = similar(target, m)

    # matrix-vector product
    for i in 1:m
        acc = zero(eltype(u))
        for j in 1:n
            op_ij = compute_entry(op_t, nothing, i, j,
                source, target, nothing)
            acc += op_ij * data(density)[j]
        end
        u[i] = acc
    end
    return u
end

# add two densities
function Base.:+(φ::AbstractBoundaryDensity, ψ::AbstractBoundaryDensity)
    return data(φ) + data(ψ)
end

# add density to raw vector
function Base.:+(φ::AbstractBoundaryDensity, v::AbstractArray)
    return data(φ) + v
end
function Base.:+(v::AbstractArray, φ::AbstractBoundaryDensity)
    return φ + v
end
function Base.:-(φ::AbstractBoundaryDensity, v::AbstractArray)
    return data(φ) - v
end
function Base.:-(v::AbstractArray, φ::AbstractBoundaryDensity)
    return v - data(φ)
end



# Allow any AbstractBoundaryDensity subtype to be constructed from another
(::Type{T})(density::AbstractBoundaryDensity) where {T<:AbstractBoundaryDensity} = T(data(density))


# actually allowing implicit conversion goes against type safety
# # Define the conversion rule
# Base.convert(::Type{T}, density::AbstractBoundaryDensity) where {T<:AbstractBoundaryDensity} = T(density)
#
# # Fallback to prevent infinite loops if it's already the right type
# Base.convert(::Type{T}, density::T) where {T<:AbstractBoundaryDensity} = density

Base.convert(::Type{T}, density::AbstractBoundaryDensity) where {T<:AbstractArray} = T(data(density))

# Base.length(density::AbstractBoundaryDensity) = length(data(density))

