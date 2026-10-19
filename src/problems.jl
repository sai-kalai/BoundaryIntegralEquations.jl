# Definitions of the problems

struct Laplace <: DifferentialEquation end
struct Helmholtz <: DifferentialEquation end
struct Stokes <: DifferentialEquation end


struct Interior <: DomainSide end
struct Exterior <: DomainSide end
# add: both sides

@doc raw"""
    BoundaryValueProblem

Represents a boundary value problem

"""
struct BoundaryValueProblem{
    E<:DifferentialEquation,
    C<:BoundaryCondition,
    S<:DomainSide,
    B<:AbstractManifold,
}
    equation::E
    bc::C
    side::S
    boundary::B
end


@doc raw"""
    BIEProblem

Represents the boundary integral equation associated with a BVP

"""
struct BIEProblem{A<:Approach,P<:BoundaryValueProblem}
    # NOTE: this already depends on approach! approach is much more fundamental than i was considering it...
    # also consider caching operators here the same way that LinearProblem caches A and b
    # CommonSolve.init returns a "cache" object that contains the problem and algorithm
    bvp::P

    function BIEProblem{A}(p::P) where {A,P}
        return new{A,P}(p)
    end
end

@doc raw"""
    BDProblem

Represents the problem of finding the missing cauchy data in a boundary value
problem, i.e. finding the Neumann data given a Dirichlet problem and vice-versa


"""
struct BDProblem{A<:Approach,P<:BoundaryValueProblem}
    bvp::P

    function BDProblem{A}(p::P) where {A,P}
        return new{A,typeof(p)}(p)
    end
end
