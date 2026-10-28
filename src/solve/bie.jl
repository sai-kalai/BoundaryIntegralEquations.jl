# dispatch for solving bie
#
# TODO: instead, overload commonsolve
function solve_linear_system(A, b; algorithm)
    pbm = LinearSolve.LinearProblem(A, b)
    sln = LinearSolve.solve(pbm, algorithm)
    return sln.u

    # NOTE: LinearSolve is needed for Enzyme rules, but it chooses a solver
    # (MKU) that degrades performance. Investigate how to pick a better solver.
    # return A \ b
    # update: RFLU from RecursiveFactorization.jl works as well as \ did,
    # solving convergence_study(20:20:200) in about 10 ms
end

### Direct Approach

#
# Laplace - Interior - Dirichlet - Direct
#
function solve(
    prob::BIEProblem{Direct,P},
    alg::AbstractBIEAlgorithm{Direct}
)::BIESolution where {
    P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve}
}
    # TODO: replace accessors by  getter or getattrib/trait
    bvprob = prob.bvp

    D_star = AdjointDoubleLayer(bvprob.equation, bvprob.boundary; discretization=alg.discretization)
    H = Hypersingular(bvprob.equation, bvprob.boundary, alg.correction; discretization=alg.discretization)
    populate_matrices!(bvprob.boundary, D_star, H)

    # TODO: work in place to avoid allocating a new matrix
    panic_if_garbage(D_star, H)
    A = -0.5 + D_star
    τ = solve_linear_system(A, (H * bvprob.bc); algorithm=alg.linsolver)

    return BIESolution(τ, prob, alg) # NOTE: for now, return raw vector type. Losing type safety tied to physical meaning.
end

#
# Laplace - Interior - Neumann - Direct
#

function solve(
    prob::BIEProblem{Direct,P},
    alg::AbstractBIEAlgorithm{Direct}
)::BIESolution where {
    P<:BoundaryValueProblem{Laplace,<:Neumann,Interior,<:DiscreteClosedCurve}
}

    bvprob = prob.bvp

    S = SingleLayer(bvprob.equation, bvprob.boundary, alg.correction; discretization=alg.discretization)
    D = DoubleLayer(bvprob.equation, bvprob.boundary; discretization=alg.discretization)
    populate_matrices!(bvprob.boundary, S, D)

    panic_if_garbage(S, D)

    # add 1 to kill 1d nullspace
    σ = solve_linear_system((0.5 + D) .+ bvprob.boundary.w, (S * bvprob.bc),
        algorithm=alg.linsolver)
    return BIESolution(σ, prob, alg)
end

### Indirect Approach

#
# Laplace - Interior - Dirichlet - Indirect
#
function solve(
    prob::BIEProblem{Indirect,P},
    alg::AbstractBIEAlgorithm{Indirect},
    ;
)::BIESolution where {
    P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve}
}

    bvprob = prob.bvp

    D = DoubleLayer(bvprob.equation, bvprob.boundary; populate_matrix=true, discretization=alg.discretization)

    panic_if_garbage(D)

    φ = solve_linear_system(-0.5 + D, bvprob.bc.σ; algorithm=alg.linsolver)

    return BIESolution(φ, prob, alg)
end

#
# Laplace - Interior - Neumann - Indirect
#

function solve(
    prob::BIEProblem{Indirect,P},
    alg::AbstractBIEAlgorithm{Indirect},
    ;
)::BIESolution where {
    P<:BoundaryValueProblem{Laplace,<:Neumann,Interior,<:DiscreteClosedCurve}
}

    bvprob = prob.bvp

    D_star = AdjointDoubleLayer(bvprob.equation, bvprob.boundary;
        populate_matrix=true, discretization=alg.discretization)

    panic_if_garbage(D_star)

    ψ = solve_linear_system((0.5 + D_star) .+ bvprob.boundary.w, bvprob.bc.τ,
        algorithm=alg.linsolver)

    return BIESolution(ψ, prob, alg)
end

