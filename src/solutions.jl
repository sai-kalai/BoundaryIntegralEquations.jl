# Definitions of solutions

struct BIESolution{
    A<:Approach,
    T<:Number, # floating point representation
    P<:BoundaryValueProblem, # BVP that this helps solve
    U<:AbstractVector{T}, # NOTE: maybe U should be AbstractBoundaryDensity?
    AL<:AbstractBIEAlgorithm{A},
    BP<:BIEProblem{A,P}, # BIE that this solves NOTE: superfluous
} <: AbstractSolution{A,T,P}
    # TODO: Link this with Linear Solution, wrapping or inheriting
    u::U # boundary density that solves the bie
    prob::BP
    alg::AL
end
function BIESolution(
    u,
    prob::BIEProblem{A,P},
    alg::AbstractBIEAlgorithm{A},
) where {A,P}

    return BIESolution{
        A,
        eltype(u),
        P,
        typeof(u),
        typeof(alg),
        typeof(prob),
    }(
        u,
        prob,
        alg
    )
end

struct BVPSolution{
    A<:Approach,
    T<:Number,
    P<:BoundaryValueProblem,
    U<:AbstractVector{T},
    X<:AbstractMatrix{T},
    AL<:AbstractBVPAlgorithm{A},
    BS<:BIESolution{A,T,P},
} <: AbstractSolution{A,T,P}
    u::U # potential generated from the boundary density in the domain
    x::X # evaluation positions
    bie_solution::BS # keep solution to the associated BIE
    prob::P
    alg::AL
    # TODO: missing: information about spatial discretization
end
function BVPSolution(
    u,
    x,
    bie_solution::BIESolution{A,T,P},
    prob::P,
    alg::AbstractBVPAlgorithm{A},
) where {A,T,P}
    return BVPSolution{
        A,
        eltype(u),
        P,
        typeof(u),
        typeof(x),
        typeof(alg),
        typeof(bie_solution),
    }(
        u,
        x,
        bie_solution,
        prob,
        alg,
    )

end

@doc raw"""
    BDPSolution

Solution to a Boundary Data Problem, where the unknown Cauchy data of a Boundary
value problem is needed.

"""
struct BDPSolution{
    A<:Approach,
    T<:Number,
    P<:BoundaryValueProblem,
    U<:AbstractVector{T},
    AL<:AbstractBDPAlgorithm{A},
    DP<:BDProblem{A,P},
    BS<:BIESolution{A,T,P}, # solution of the bie
} <: AbstractSolution{A,T,P}
    u::U # potential generated from the boundary density in the domain
    bie_solution::BS # keep solution to the associated BIE
    prob::DP
    alg::AL
end
function BDPSolution(
    u,
    bie_solution::BIESolution{A,T,P},
    prob::BDProblem{A,P},
    alg::AbstractBDPAlgorithm{A},
) where {A,T,P}
    return BDPSolution{
        A,
        eltype(u),
        P,
        typeof(u), # WARN: actually, eltype(u) and T are tied... make this explicit
        typeof(alg),
        typeof(prob),
        typeof(bie_solution),
    }(
        u,
        bie_solution,
        prob,
        alg,
    )

end

const NumericalSolution{
    A,T,P
} = Union{
    BVPSolution{A,T,P},
    BIESolution{A,T,P},
    BDPSolution{A,T,P},
}

function Base.show(io::IO, ::Type{<:BVPSolution{A,T,P}}) where {A,T,P}
    print(io, "BVPSolution")
end
function Base.show(io::IO, ::Type{<:BIESolution{A,T,P}}) where {A,T,P}
    print(io, "BIESolution")
end
function Base.show(io::IO, ::Type{<:BDPSolution{A,T,P}}) where {A,T,P}
    print(io, "BDPSolution")
end

# solution/algorithm traits

numpoints(s::AbstractSolution) = size(bvp(s).boundary.x, 2)
bvp(s::BIESolution) = s.prob.bvp
bvp(s::BDPSolution) = s.prob.bvp
bvp(s::BVPSolution) = s.prob
boundary_condition(::AbstractSolution{A,T,<:BoundaryValueProblem{E,C}}) where {A,T,E,C} = C

