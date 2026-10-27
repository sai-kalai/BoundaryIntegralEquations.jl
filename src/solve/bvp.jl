include("util.jl")
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
function solve(
    prob::P,
    alg::AbstractBVPAlgorithm{Direct},
    ;
    bie_sol::BIESolution{Direct,T,P}, # if nothing, compute internally
    target::AbstractMatrix,
)::BVPSolution where {
    T,P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve},
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

    return BVPSolution(u, bie_sol, prob, alg)
end


### Indirect approach

function solve(
    prob::P,
    alg::IndirectBVPAlgorithm{D,PotentialTheory},
    ;
    bie_sol::BIESolution{Indirect,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    D,T,P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve}
}
    u = apply(DoubleLayer{Laplace,Nothing,D}, prob.boundary, target, bie_sol.u)
    return BVPSolution(u, bie_sol, prob, alg)
end

function solve(
    prob::P,
    alg::IndirectBVPAlgorithm{D,CauchyIntegral},
    ;
    bie_sol::BIESolution{Indirect,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    D,T,P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve}
}
    # NOTE: here D and alg are kind of ignored?

    v_lim = holomorphism_boundary_limit(
        prob,
        bie_sol.u,
    )
    v = cauchy_integral(prob.boundary, target, v_lim)

    return BVPSolution(real.(v), bie_sol, prob, alg)
end

function solve(
    prob::P,
    alg::IndirectBVPAlgorithm{D,<:DistancePolicy},
    ;
    bie_sol::BIESolution{Indirect,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    D,T,P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve}
}

    m = size(target, 2)

    # find points in the correct side of the domain and within cutoff distance
    distance = lengthscale(prob.boundary) * alg.evalmethod.cutoff

    near_idxs, far_idxs, bad_idxs = classify(prob.boundary, target,
        prob.side, distance)

    u_far = solve(
        prob,
        IndirectBVPAlgorithm(alg.discretization, PotentialTheory())
        ;
        target=target[:, far_idxs],
        bie_sol=bie_sol
    ).u

    u_near = solve(
        prob,
        IndirectBVPAlgorithm(alg.discretization, CauchyIntegral())
        ;
        target=target[:, near_idxs],
        bie_sol=bie_sol
    ).u

    # TODO: how to make this classification available to the solution?
    u = similar(target, m)
    u[far_idxs] .= u_far
    u[near_idxs] .= u_near
    u[bad_idxs] .= NaN

    return BVPSolution(u, bie_sol, prob, alg)
end
function solve(
    prob::P,
    alg::DirectBVPAlgorithm{D},
    ;
    bie_sol::BIESolution{Direct,T,P},
    target::AbstractMatrix,
)::BVPSolution where {
    T,D,
    P<:BoundaryValueProblem{Laplace,<:Neumann,Interior,<:DiscreteClosedCurve},
}

    m = size(target, 2)
    u = similar(target, m)

    near_id, far_id, bad_id = classify(prob.boundary, target, prob.side, 0.)

    @assert length(near_id) == 0

    a = apply(SingleLayer{Laplace,Nothing,D}, prob.boundary, target[:, far_id], prob.bc)
    b = apply(DoubleLayer{Laplace,Nothing,D}, prob.boundary, target[:, far_id], bie_sol.u)

    u[far_id] .= a - b
    u[bad_id] .= NaN

    return BVPSolution(u, bie_sol, prob, alg)
end
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

    return BVPSolution(u, bie_sol, prob, alg)
end

