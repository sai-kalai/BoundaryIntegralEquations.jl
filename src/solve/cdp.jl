# dispatch for solving cauchy data problems

function solve(
    prob::BDProblem{A,<:BoundaryValueProblem},
    alg::AbstractBDPAlgorithm{A},
    bie_alg::AbstractBIEAlgorithm{A},
)::BDPSolution where {A}
    bie_sol = solve(BIEProblem{A}(prob.bvp), bie_alg)
    return solve(prob, alg; bie_sol=bie_sol)
end

function solve(
    prob::BDProblem{Direct,P},
    alg::AbstractBDPAlgorithm{Direct},
    ;
    bie_sol::BIESolution{Direct,T,P}
)::BDPSolution where {
    T,P<:BoundaryValueProblem{Laplace,<:BoundaryCondition,Interior,<:DiscreteClosedCurve}
}
    # Direct
    return BDPSolution(bie_sol.u, bie_sol, prob, alg)
end

function solve(
    prob::BDProblem{Indirect,P},
    alg::IndirectBDPAlgorithm{D,C},
    ;
    bie_sol::BIESolution{Indirect,T,P}
)::BDPSolution where {
    P<:BoundaryValueProblem{Laplace,<:Dirichlet,Interior,<:DiscreteClosedCurve},
    D,C,T,
}
    # Indirect | Dirichlet
    bvprob = prob.bvp
    H = Hypersingular(
        bvprob.equation, bvprob.boundary, alg.correction;
        discretization=alg.discretization, populate_matrix=true)
    panic_if_garbage(H)

    τ = matrix(H) * bie_sol.u
    return BDPSolution(τ, bie_sol, prob, alg)
end
function solve(
    prob::BDProblem{Indirect,P},
    alg::IndirectBDPAlgorithm{D,C},
    ;
    bie_sol::BIESolution{Indirect,T,P}
)::BDPSolution where {
    P<:BoundaryValueProblem{Laplace,<:Neumann,Interior,<:DiscreteClosedCurve},
    D,C,T,
}
    # Indirect | Neumann
    bvprob = prob.bvp
    S = SingleLayer(
        bvprob.equation, bvprob.boundary, alg.correction;
        discretization=alg.discretization, populate_matrix=true)
    panic_if_garbage(S)

    σ = matrix(S) * bie_sol.u
    return BDPSolution(σ, bie_sol, prob, alg)
end

