# dispatch for solving bvp

# convenience for internal bie solution computation
function solve(
    prob::BoundaryValueProblem,
    alg::AbstractBVPAlgorithm{A},
    bie_alg::AbstractBIEAlgorithm{A},
    ;
    target::AbstractMatrix,
)::BVPSolution where {A}

    bie_sol = solve(BIEProblem{A}(prob), bie_alg)

    return solve(prob, alg; bie_sol=bie_sol, target=target)
end
# Dirichlet | Direct
function solve(
    prob::P,
    alg::AbstractBVPAlgorithm{A},
    ;
    bie_sol::BIESolution{A,T,P}, # if nothing, compute internally
    target::AbstractMatrix,
)::BVPSolution where {
    A<:Direct,
    BC<:Dirichlet,
    P<:BoundaryValueProblem{Laplace,BC,Interior,<:DiscreteClosedCurve},
    T,
}

    near_id, far_id, bad_id = classify(prob.boundary, target, prob.side,)

    @assert length(near_id) == 0

    m = size(target, 2)
    u = similar(target, m)

    #NOTE: this is very brittle: boundary density still used by apply.

    a = apply(SingleLayer{Laplace,Nothing,Nystroem}, prob.boundary, target[:, far_id], bie_sol.u)
    b = apply(DoubleLayer{Laplace,Nothing,Nystroem}, prob.boundary, target[:, far_id], prob.bc)

    u[far_id] .= a - b
    u[bad_id] .= NaN

    return BVPSolution(u, target, bie_sol, prob, alg)
end
# Neumann | Direct
function solve(
    prob::P,
    alg::DirectBVPAlgorithm{D},
    ;
    bie_sol::BIESolution{A,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    A<:Direct,
    BC<:Neumann,
    P<:BoundaryValueProblem{Laplace,BC,Interior,<:DiscreteClosedCurve},
    T,D,
}

    m = size(target, 2)
    u = similar(target, m)

    near_id, far_id, bad_id = classify(prob.boundary, target, prob.side, 0.)

    @assert length(near_id) == 0

    a = apply(SingleLayer{Laplace,Nothing,D}, prob.boundary, target[:, far_id], prob.bc)
    b = apply(DoubleLayer{Laplace,Nothing,D}, prob.boundary, target[:, far_id], bie_sol.u)

    u[far_id] .= a - b
    u[bad_id] .= NaN

    return BVPSolution(u, target, bie_sol, prob, alg)
end

### Indirect approach


# unsafe helpers
function _solve_ci(prob, bie_sol, target)
    v_lim = holomorphism_boundary_limit(
        prob,
        bie_sol,
    )
    v = cauchy_integral(prob.boundary, target, v_lim)
    return v
end
function _solve_pt(bdry, target, bie_sol, disc_t)
    apply(DoubleLayer{Laplace,Nothing,disc_t}, bdry, target, bie_sol)
end

# Dirichlet | Indirect
# helper
distance(::PotentialTheory, args...) = 0
distance(::CauchyIntegral, args...) = Inf
distance(m::DistancePolicy, b::DiscreteClosedCurve) = lengthscale(b) * m.cutoff
# Distance Policy
function solve(
    prob::P,
    alg::IndirectBVPAlgorithm{D,M},
    ;
    bie_sol::BIESolution{A,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    A<:Indirect,
    BC<:Dirichlet,
    M,
    P<:BoundaryValueProblem{Laplace,BC,Interior,<:DiscreteClosedCurve},
    D,T,
}
    m = size(target, 2)
    u = similar(target, m)
    # find points in the correct side of the domain and within cutoff distance
    # NOTE: this looks more and more like a init/cache implementation for bvp points
    # -> bvp doesn't know about x points, cache does and preprocesses
    d = distance(alg.evalmethod, prob.boundary)
    near_idxs, far_idxs, bad_idxs = classify(prob.boundary, target,
        prob.side, d)

    u_far = _solve_pt(prob.boundary, target[:, far_idxs], bie_sol.u, D)
    u_near = _solve_ci(prob, bie_sol.u, target[:, near_idxs]) .|> real

    # TODO: how to make this classification available to the solution?
    u[far_idxs] .= u_far
    u[near_idxs] .= u_near
    u[bad_idxs] .= NaN

    return BVPSolution(u, target, bie_sol, prob, alg)
end
# Neumann | Indirect
function solve(
    prob::P,
    alg::IndirectBVPAlgorithm{D,M},
    ;
    bie_sol::BIESolution{Indirect,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    T,D,M<:PotentialTheory,
    P<:BoundaryValueProblem{Laplace,<:Neumann,Interior,<:DiscreteClosedCurve},
}
    m = size(target, 2)

    u = similar(target, m)

    near_id, far_id, bad_id = classify(prob.boundary, target, prob.side, 0.)

    @assert length(near_id) == 0

    # TODO: avoid allocating u inside apply as well as here
    u[far_id] .= apply(SingleLayer{Laplace,Nothing,D}, prob.boundary, target[:, far_id], bie_sol.u)
    u[bad_id] .= NaN

    return BVPSolution(u, target, bie_sol, prob, alg)
end

