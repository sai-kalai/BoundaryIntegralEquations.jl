
abstract type AbstractBoundaryDensity end

# TODO type stability
struct BoundaryDensity <: AbstractBoundaryDensity
    φ::AbstractVector
end
function data(density::BoundaryDensity)
    density.φ
end
abstract type BoundaryCondition <: AbstractBoundaryDensity end


struct Dirichlet <: BoundaryCondition
    σ::AbstractVector
end

function data(density::Dirichlet)
    density.σ
end
struct Neumann <: BoundaryCondition
    τ::AbstractVector
end
function data(density::Neumann)
    density.τ
end

# apply an operator to a density
function Base.:*(A::IntegralOperator, φ::AbstractBoundaryDensity)
    return matrix(A) * data(φ)
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

Base.convert(::Type{T}, density::AbstractBoundaryDensity) where {T<:AbstractVector} = T(data(density))

# Base.length(density::AbstractBoundaryDensity) = length(data(density))

