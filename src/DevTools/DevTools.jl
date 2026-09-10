module DevTools

using BenchmarkTools # keeping as dependency for now, find better way
using LinearAlgebra
using StaticArrays

using ..BoundaryIntegralEquations

export ConvergenceResult, SolverParameters, SolutionMetadata,
    SolutionWithMetadata, add_solutions!, SolutionGroup,
    solutions, metadatas, trials, times, gctimes, manufactured_solution, errors


export run_all_simulations, Fixtures

include("Fixtures.jl")

@doc raw"""
    SolverParameters

identifies the parameters used for running one solver

"""
struct SolverParameters
    approach_t::Type{<:Approach}
    bdrycond_t::Type{<:BoundaryCondition}
    solution_t::Type{<:NumericalSolution}
    correction::AbstractSingularCorrection
    evalmethod::EvaluationMethod
end
# function Base.show(io::IO, param::SolverParameters)
#     print(io, "SolverParameters: ", )
# end

# compare keys
function Base.isless(a::SolverParameters, b::SolverParameters)

    if a.approach_t <: Direct && b.approach_t <: Indirect
        return false
    elseif a.approach_t <: Indirect && b.approach_t <: Direct
        return true
    end

    if cutoff(a.evalmethod) != cutoff(b.evalmethod)
        return cutoff(a.evalmethod) < cutoff(b.evalmethod)
    end

    @show typeof(a.correction), typeof(b.correction)
    if !(typeof(a.correction) <: typeof(b.correction))
        @show typeof(a.correction), typeof(b.correction)
        if a.correction isa Zeta
            return true
        elseif a.correction isa Sidi
            return true
        elseif a.correction isa KapurRokhlin
            return false
        end
    end

    return return order(a.correction) < order(b.correction)
end

@doc raw"""
    SolutionMetadata

Contains information about a simulation such as runtime

"""
struct SolutionMetadata
    # initial sketch, maybe include Tryal instance here
    trial::Union{BenchmarkTools.Trial,Nothing}
end
const SolutionWithMetadata = Tuple{NumericalSolution,SolutionMetadata}
SolutionWithMetadata(s, m) = SolutionWithMetadata((s, m))
const SolutionGroup = Vector{SolutionWithMetadata}
# iterators for broadcasting
solutions(g::SolutionGroup) = (s for (s, _) in g)
metadatas(g::SolutionGroup) = (m for (_, m) in g)
trials(g::SolutionGroup) = (m.trial for (_, m) in g)
times(g::SolutionGroup) = (t.times for t in trials(g))
gctimes(g::SolutionGroup) = (t.gctimes for t in trials(g))


@doc raw"""
    ConvergenceResult

Stores the metadata and data associated with a group of simulation runs with
different parameters for several discretization sizes

"""
mutable struct ConvergenceResult{T}
    # metadata: parameters used during the runs
    n_vals::Vector{Int}
    cutoff_vals::Vector{T}
    fd_acc_vals::Vector{Int}
    kr_acc_vals::Vector{Int}

    x::Matrix{T} # x locations in col-major

    u_exact::Vector{T} # exact solution at x points
    neumann_exact::Dict{Int,Vector{T}} # exact neumann data for each n val
    dirichlet_exact::Dict{Int,Vector{T}}

    # results of the simulations for several n values, grouped by solver parameters
    solutions::Dict{SolverParameters,SolutionGroup}
end
function ConvergenceResult(
    n_vals::Vector{Int},
    cutoff_vals::Vector{T},
    fd_acc_vals::Vector{Int},
    kr_acc_vals::Vector{Int},
    x::Matrix{T},
    u_exact::Vector{T},
) where {T}
    return ConvergenceResult{eltype(T)}(
        n_vals,
        cutoff_vals,
        fd_acc_vals,
        kr_acc_vals,
        x,
        u_exact,
        Dict{Int,Vector{T}}(), # exact neumann and dirichlet data for each n value
        Dict{Int,Vector{T}}(), # exact neumann and dirichlet data for each n value
        Dict{SolverParameters,Vector{SolutionWithMetadata}}(),
    )
end
function add_solutions!(res::ConvergenceResult, correction, evalmethod, sols_with_md...)
    foreach(sols_with_md) do sol_with_md
        (sol, md) = sol_with_md
        push!(
            get!(
                res.solutions,
                SolverParameters(
                    approach(sol.alg),
                    boundary_condition(sol),
                    typeof(sol),
                    correction,
                    evalmethod,
                ),
                Vector{Float64}() # why vector of float...?
            ),
            sol_with_md
        )
    end

end
function Base.filter!(
    res::ConvergenceResult,
    by_key=(k)->true,
)

    filter!(((k, v),) -> begin
            by_key(k)
        end, res.solutions
    )
    fd_ords = [
        order(k.correction)
        for k in keys(res.solutions)
        if k.correction isa Zeta
    ]
    kr_ords = [
        order(k.correction)
        for k in keys(res.solutions)
        if k.correction isa KapurRokhlin
    ]

    intersect!(res.fd_acc_vals, fd_ords)

    intersect!(res.kr_acc_vals, kr_ords)

    intersect!(res.cutoff_vals,
        (
            cutoff(k.evalmethod)
            for k in keys(res.solutions)
        )
    )
end

@doc raw"""
    errors(k::SolverParameters, res::ConvergenceResult, group::SolutionGroup)

Return a vector of errors for all mesh sizes of a particular solution type
in a convergence result
"""
function errors(
    key::SolverParameters,
    res::ConvergenceResult,
    group::SolutionGroup
)
    sols = solutions(group)

    errs = if key.solution_t <: BVPSolution
        [
            begin
                # exact solution at incorrect side contains correct solution;
                # numerical solution at incorrect side contains NaN
                # TODO: not always the three are needed
                _, far_ids, _ = classify(s.prob.boundary, res.x, s.prob.side, 0.)
                norm(s.u[far_ids] - res.u_exact[far_ids], Inf)
            end
            for s in sols
        ]
    elseif key.solution_t <: BDPSolution
        if key.bdrycond_t <: Dirichlet
            [norm(s.u - res.neumann_exact[numpoints(s)], Inf) for s in sols]
        elseif key.bdrycond_t <: Neumann
            [norm(s.u - res.dirichlet_exact[numpoints(s)], Inf) for s in sols]
        else
            error("invalid bc type $(key.bdrycond_t)")
        end
    else
        error("invalid solution type $(key.solution_t)")
    end
    if any(isnan.(errs))
        @warn "NaN found in errs:"
        @show key
        @show errs
    end
    return errs
end


function Base.show(io::IO, ::MIME"text/plain", res::ConvergenceResult{T}) where {T}
    println(io, "ConvergenceResult{", T, "}:")
    println(io, "  n_vals:          ", res.n_vals)
    println(io, "  cutoff_vals:     ", res.cutoff_vals)
    println(io, "  fd_acc_vals:     ", res.fd_acc_vals)
    println(io, "  kr_acc_vals:     ", res.kr_acc_vals)
    println(io, "  x:               ", summary(res.x))
    println(io, "  u_exact:         ", summary(res.u_exact))
    println(io, "  neumann_exact:   Dict with ", length(res.neumann_exact), " entries")
    println(io, "  dirichlet_exact: Dict with ", length(res.dirichlet_exact), " entries")
    println(io, "  solutions:       ", length(res.solutions), "-element ", typeof(res.solutions))
end


@doc raw"""
    manufactured_solution()

computes known solution at given points and useful information at point sources

Returns tuple of:
- source dummy curve
- density at sources
- exact solution at test points

"""
function manufactured_solution(eqn::Laplace, x_test,)
    # TODO:
    x_source, density_source = Fixtures.point_sources()
    density_source = BoundaryDensity(density_source)

    # operators for exact solution at test and plot points
    Γ_source = make_dummy(x_source)
    S_source = SingleLayer(eqn, Γ_source, x_test, populate_matrix=true)
    u_exact = S_source * density_source # exact solution at test points
    return Γ_source, BoundaryDensity(density_source), u_exact
end

function find_farthest(target, bdry)
    max_cost = Inf
    farthest_idx = -1

    for i in 1:size(target, 2)
        cost = 0
        for j in 1:size(bdry, 2)
            cost += sum(abs2, view(target, :, i) .- view(bdry, :, j))
        end
        if cost < max_cost
            max_cost = cost
            farthest_idx = i
            # @show max_cost
        end
    end

    return farthest_idx, sqrt(max_cost)
end

@doc raw"""

run all methods with different parameters

"""
function run_all_simulations(
    x_test::AbstractMatrix, # test locations
    ;
    n_vals=20:20:400,
    cutoff_vals=[0.0, 0.01, 0.05, 0.1, 0.5],
    fd_acc_vals=[4, 8, 16, 32],
    kr_acc_vals=copy(fd_acc_vals),
    approach_types=[Direct, Indirect],
    bc_types=[Dirichlet, Neumann],
    # indicate how to reserve memory
    allocator=(_m, _n) -> Matrix{Float64}(undef, _m, _n),
    benchmark_kwargs=nothing,
)
    @show n_vals
    @show cutoff_vals
    @show fd_acc_vals
    @show kr_acc_vals
    @show benchmark_kwargs

    # common variables
    laplace = Laplace()
    interior = Interior()
    exterior = Exterior()
    direct = Direct()
    indirect = Indirect()

    Γ_source, density_source, u_exact = manufactured_solution(laplace, x_test)
    # accumulate results
    res = ConvergenceResult(collect(n_vals), cutoff_vals, fd_acc_vals, kr_acc_vals,
        x_test, u_exact)

    # verify that results match MATLAB version
    u_exact_reference = Fixtures.reference_exact_solution()
    # move this to test module
    # @test u_exact[1:length(u_exact_reference)] ≈ u_exact_reference atol = 1e-15

    # storage for produced solutions
    nan_count = 0
    function validate_nan(sols...)
        foreach(sols) do sol
            @info "validating $(typeof(sol))"
            @show sol.alg
            if any(isnan, sol.u)
                @warn "NaN found in solution"
                # @show sol.u
                nan_count += 1
            end
        end
    end

    for n in n_vals

        @info @show n

        # boundary discretization
        Γ = DiscreteClosedCurve(n, starfish)

        # operators for computing boundary conditions from point sources
        S_source = SingleLayer(laplace, Γ_source, Γ.x)
        D_star_source = AdjointDoubleLayer(laplace, Γ_source, Γ.x)
        populate_matrices!(Γ_source, Γ.x, S_source, D_star_source; target_normals=Γ.n)
        # TODO: test this
        # @assert D_star_source.matrix ≈ AdjointDoubleLayer(laplace, Γ.x, Γ.n, Γ_source; matrix_factory=allocator).matrix
        σ_exact = S_source * density_source # Dirichlet BC
        τ_exact = D_star_source * density_source # Neumann BC exact solution


        res.dirichlet_exact[n] = σ_exact
        res.neumann_exact[n] = τ_exact

        for side in [interior,], bc in [Dirichlet(σ_exact), Neumann(τ_exact)]

            @show side, typeof(bc)

            if bc isa Neumann
                # find farthest point from boundary in the correct side of the
                # domain to reconstruct the integration constant
                near_id, far_id, bad_id = classify(Γ, x_test, side, 0.0)
                farthest_idx, farthest_dist = find_farthest(x_test[:, far_id], Γ.x)
                farthest_idx = far_id[farthest_idx]

                # farthest_idx = size(x_test, 2) ÷ 2
                # @error x_test[:, farthest_idx]

                # fig = Main.Figure()
                # ax = Main.Axis(fig[1, 1])
                # Main.scatter!(ax, x_test[:, near_id], color=:blue)
                # Main.scatter!(ax, x_test[:, far_id], color=:green)
                # Main.scatter!(ax, x_test[:, bad_id], color=:red)
                # Main.scatter!(ax, x_test[:, farthest_idx]..., color=:black)
                # fig |> display |> wait
            end

            if !any(T -> bc isa T, bc_types)
                continue
            end

            pb = BoundaryValueProblem(laplace, bc, side, Γ)

            # NOTE: actually, operators of different orders can be precomputed
            # in parallel using tuple comprenhension
            # ops = (
            #   (SingleLayer(laplace, Γ, KapurRokhlin(x)) for x kr_acc_vals)...,
            #   (Hypersingular(laplace, Γ, Zeta(x)) for x fd_acc_vals)...,
            #   Sidi(),
            #   )
            # populate_matrices!(Γ, ops)
            # for op in ops
            #    solve_and_evaluate(
            #       prob, approach, (op, op isa SingleLayer ? D_star : op isa Hypersingular?  etc...
            #    ) -> requires putting correct args
            # end

            corrections = if bc isa Dirichlet
                [Sidi(); [Zeta(x) for x in fd_acc_vals]]
            elseif bc isa Neumann
                [KapurRokhlin(x) for x in kr_acc_vals]
            else
                error("invalid bc")
            end

            for correction in corrections
                @show correction

                if Direct in approach_types
                    @show direct
                    # direct approach
                    u, cauchy_data = solve_and_evaluate(
                        pb,
                        direct,
                        correction,
                        x_test,
                        ;
                        matrix_factory=allocator
                    )

                    if !isnothing(benchmark_kwargs)
                        b = @benchmarkable solve_and_evaluate(
                            $pb,
                            $direct,
                            $correction,
                            $x_test,
                            ;
                            matrix_factory=($allocator)
                        )
                        @time begin
                            trial = run(b; benchmark_kwargs...)
                        end
                        display(trial)
                    else
                        trial = nothing
                    end

                    if bc isa Neumann
                        offset = u_exact[farthest_idx] - u[farthest_idx]
                        u .+= offset
                        data(cauchy_data) .+= offset # TODO: put this inside solver maybe and user passes integration constant
                    end


                    # dummy placeholder for now
                    bie_sln = BIESolution(
                        zeros(n),
                        BIEProblem{Direct}(pb),
                        BIEAlgorithm{Direct}(correction),
                    )

                    bvp_sln = BVPSolution(
                        u,
                        bie_sln,
                        pb,
                        BVPAlgorithm{Direct}(),
                    )
                    bdp_sln = BDPSolution(
                        data(cauchy_data),
                        bie_sln,
                        BDProblem{Direct}(pb),
                        BDPAlgorithm{Direct}(),
                    )
                    validate_nan(bvp_sln, bie_sln, bdp_sln)

                    add_solutions!(res, correction, PotentialTheory(),
                        SolutionWithMetadata(bvp_sln, SolutionMetadata(trial)),
                        SolutionWithMetadata(bdp_sln, SolutionMetadata(trial))
                    )
                end

                if Indirect in approach_types
                    @show indirect
                    # indirect approach: cutoff is available
                    for cutoff in cutoff_vals

                        method = cutoff == 0. ? PotentialTheory() :
                                 isinf(cutoff) ? CauchyIntegral() :
                                 DistancePolicy(cutoff)

                        try
                            u, cauchy_data = solve_and_evaluate(
                                pb,
                                indirect,
                                correction,
                                x_test,
                                cutoff,
                                ;
                                matrix_factory=allocator
                            )
                        catch e
                            @error e
                            continue
                        end

                        if bc isa Neumann
                            #recover integration constant
                            offset = u_exact[farthest_idx] - u[farthest_idx]
                            u .+= offset
                            data(cauchy_data) .+= offset # TODO: put this inside solver maybe and user passes integration constant
                        end

                        @show method
                        if !isnothing(benchmark_kwargs)
                            b = @benchmarkable solve_and_evaluate(
                                $pb,
                                $indirect,
                                $correction,
                                $x_test,
                                $cutoff,
                                ;
                                matrix_factory=($allocator)
                            )
                            @time begin
                                trial = run(b; benchmark_kwargs...)
                            end
                            display(trial)
                        else
                            trial = nothing
                        end

                        # dummy placeholder
                        bie_sln = BIESolution(
                            zeros(n),
                            BIEProblem{Indirect}(pb),
                            BIEAlgorithm{Indirect}(),
                        )


                        bvp_sln = BVPSolution(
                            u,
                            bie_sln,
                            pb,
                            BVPAlgorithm{Indirect}(method),
                        )

                        bdp_sln = BDPSolution(
                            data(cauchy_data),
                            bie_sln,
                            BDProblem{Indirect}(pb),
                            BDPAlgorithm{Indirect}(correction),
                        )

                        validate_nan(bvp_sln, bie_sln, bdp_sln)

                        add_solutions!(res, correction, method,
                            SolutionWithMetadata(bvp_sln, SolutionMetadata(trial)),
                            SolutionWithMetadata(bdp_sln, SolutionMetadata(trial))
                        )
                    end
                end
            end
        end
    end

    @info @show nan_count
    return res
end


end
