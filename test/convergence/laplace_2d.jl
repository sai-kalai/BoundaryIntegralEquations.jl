
# TEST_LAP2D_HYPER_BIE
# Test hypersingular zeta-corrected trapezoidal rule for Laplace layer
# potentials on smooth geometries by solving the BVPs using a direct
# approach to BIE:
#      	int Laplace ansatz: u = S*(du/dn) - D*u
#       int Calder?n projection:     u =  (1/2-D)*u  +         S*(du/dn)
using LinearAlgebra: exactdiv
#                                du/dn =       -T*u  + (1/2+D^*)*(du/dn)
#      	ext Laplace ansatz: u = D*u - S*(du/dn) + omega
#       ext Calder?n projection:     u =  (1/2+D)*u  -         S*(du/dn)
#                                du/dn =        T*u  + (1/2-D^*)*(du/dn)
#
# c.f. Hsiao-Wendland 2008, Sec.1.3-1.4

using Test
using Revise

using Enzyme



using LinearAlgebra

using BimDiff



include("../fixtures.jl")

abstract type Solution end
abstract type NumericalSolution{S,A} end


mutable struct DirichletSolution{S<:DomainSide,A<:Approach,C<:HypersingularCorrection} <: NumericalSolution{S,A}
    n
    u::Vector{Float64}
    τ::Vector{Float64}
    correction::C
    u_err
    τ_err
end

struct NeumannSolution{S<:DomainSide,A<:Approach} <: NumericalSolution{S,A}
    n
    u::Vector{Float64}
    σ::Vector{Float64}
    u_err
    σ_err
end

struct ExactSolution{S<:DomainSide} <: Solution
    n
    u::Vector{Float64}
    σ::Vector{Float64}
    τ::Vector{Float64}
end


function get_trace_err(s::DirichletSolution)
    return s.τ_err
end
function get_trace_err(s::NeumannSolution)
    return s.σ_err
end

solution_label(sol) = begin
    bc =
        sol isa DirichletSolution ? "Dirichlet" :
        sol isa NeumannSolution ? "Neumann" :
        "Unknown"

    approach =
        nameof(typeof(sol).parameters[2])

    correction =
        sol isa DirichletSolution ?
        string(nameof(typeof(sol).parameters[3])) :
        ""

    isempty(correction) ?
    "$bc / $approach" :
    "$bc / $approach / $correction"
end

get_color(::DirichletSolution{S,A,Sidi}) where
{S<:DomainSide,A<:Approach} = :magenta
get_color(::DirichletSolution{S,A,Zeta}) where
{S<:DomainSide,A<:Approach} = :red
get_color(::NeumannSolution) = :blue

get_linestyle(::NumericalSolution{S,Direct}) where
{S<:DomainSide} = :solid
get_linestyle(::NumericalSolution{S,Indirect}) where
{S<:DomainSide} = :dash

function solution_style(sol)

    color = get_color(sol)
    linestyle = get_linestyle(sol)

    return (; linestyle, color,)
end


function plot_errors(
    solutions::Vector{NumericalSolution},
)
    # ----------------------------
    # Group by configuration
    # ----------------------------

    groups = Dict{String,Vector{NumericalSolution}}()

    for sol in solutions

        key = solution_label(sol)

        if !haskey(groups, key)
            groups[key] = NumericalSolution[]
        end

        push!(groups[key], sol)
    end

    # ----------------------------
    # Plot
    # ----------------------------

    fig = Figure(size=(900, 600))


    ax = Axis(
        fig[1, 1],
        xlabel="n",
        ylabel="∞-error",
        yscale=log10,
        xscale=log10,
    )

    for (label, sols) in groups

        sort!(sols, by=s -> s.n)

        ns = [s.n for s in sols]

        u_errs = [s.u_err for s in sols]

        st = solution_style(first(sols))

        scatterlines!(
            ax,
            ns,
            u_errs,
            label=label,
            linestyle=st.linestyle,
            color=st.color,
            marker=:circle,
            markersize=12,
            linewidth=2,
        )

        trace_errs = [
            s isa DirichletSolution ? s.τ_err : s.σ_err
            for s in sols
        ]

        scatterlines!(
            ax,
            ns,
            trace_errs,
            label="$label trace",
            linestyle=st.linestyle,
            color=st.color,
            marker=:rect,
            markersize=12,
            linewidth=2,
        )

    end


    Legend(
        fig[1, 1],
        [
            LineElement(linestyle=:solid),
            LineElement(linestyle=:dash),
            MarkerElement(color=:blue, marker=:circle),
            MarkerElement(color=:magenta, marker=:circle),
            MarkerElement(color=:red, marker=:circle),
            MarkerElement(color=:black, marker=:circle),
            MarkerElement(color=:black, marker=:rect),
        ],
        [
            "Direct",
            "Indirect",
            "Dirichlet (Zeta)",
            "Dirichlet (Sidi)",
            "Neumann",
            "Solution",
            "Boundary Trace"
        ], "Legend";
        tellwidth=false,
        halign=:left,
        valign=:bottom
    )

    fig

    # order_offset = ord - 2
    # # trendlines
    # lines!(
    #     ax,
    #     n_vals,
    #     (n_vals ./ (n_vals[1])) .^ float(-order_offset),
    #     label="O(h^$(order_offset))",
    #     linestyle=:dash,
    #     color=:black)
    # exponential_decay = 0.1
    # lines!(ax,
    #     n_vals,
    #     exp.(-exponential_decay .* (n_vals .- n_vals[1])),
    #     label="O(exp(-$exponential_decay / h))",
    #     linestyle=:dot,
    #     color=:black
    # )

end



function convergence_study(n_vals=20:20:200, accuracy_order=32)


    # useful constants
    laplace = Laplace()
    interior = Interior()
    exterior = Exterior()
    kapur_rokhlin = KapurRokhlin(accuracy_order)
    zeta = Zeta(accuracy_order)
    sidi = Sidi()
    direct = Direct()
    indirect = Indirect()

    # indicate how to reserve memory
    allocator = (_m, _n) -> zeros(_m, _n)

    x_test = test_locations()


    x_source, density_source = point_sources()

    density_source = BoundaryDensity(density_source)

    n_source = size(x_source, 1)


    Γ_source = make_dummy_curve(x_source)

    S_manuf = SingleLayer(laplace, Γ_source, x_test; matrix_factory=allocator)

    # matrix = compute_laplace_slp_matrix(x_test, x_source)
    u_exact = S_manuf * density_source # exact solution at test points
    u_exact_reference = reference_exact_solution()

    # @assert norm(u_exact - u_exact_reference) < 1e-15
    @test u_exact ≈ u_exact_reference atol=1e-15

    x_test = [x_test; ball(0.1, 10); ball(0.3, 30); ball(0.6, 60); Matrix(stack((t) -> starfish(t, 0.9), 0:0.1:2pi))']

    # scatter!(ax, x_test[:, 1], x_test[:, 2], color=u_exact)

    # println("Printing max-norm errors")
    # println("Interior")


    num_solutions = Vector{NumericalSolution}()

    for (i, n) ∈ enumerate(n_vals)

        Γ = DiscreteClosedCurve(n, starfish) # boundary of the domain

        # fig, ax = visualize(Γ)

        # target: domain boundary, source: manufactured solution point sources
        S_source = SingleLayer(laplace, Γ_source, Γ.x; matrix_factory=allocator)
        D_star_source = AdjointDoubleLayer(laplace, Γ_source, Γ.x, Γ.n; matrix_factory=allocator)


        σ = S_source * density_source # Dirichlet BC
        τ_exact = D_star_source * density_source # Neumann BC exact solution



        S = SingleLayer(laplace, Γ, kapur_rokhlin)
        D = DoubleLayer(laplace, Γ)
        D_star = AdjointDoubleLayer(laplace, Γ)
        H_zeta = Hypersingular(laplace, Γ, zeta)
        H_sidi = Hypersingular(laplace, Γ, sidi)


        S_target = SingleLayer(laplace, Γ, x_test,)
        D_target = DoubleLayer(laplace, Γ, x_test,)

        # Dirichlet Zeta Direct

        s = (x) -> begin
            bc = Dirichlet(σ)
            pb = BoundaryValueProblem(
                laplace,
                bc,
                interior,
                Γ
            )

            return solve_and_evaluate(
                pb,
                Indirect(),
                Sidi(),
                x,
            )
        end



        @time u, τ = s(x_test)




        prob = BoundaryValueProblem(
            laplace,
            Dirichlet(σ),
            interior,
            Γ
        )


        bie_solution = solve(prob, direct, D_star, H_zeta)

        # primal run of solution evaluation
        bvp_solution = evaluate(prob, direct, bie_solution, x_test)



        # diff of only evaluation of solution

        @time begin

            dx_test_1 = Enzyme.make_zero(x_test)
            dx_test_2 = Enzyme.make_zero(x_test)
            dprob = Enzyme.make_zero(prob)

            dbie_solution = Enzyme.make_zero(bie_solution)

            dx_test_1[:, 1] .= 1.;
            dx_test_2[:, 2] .= 1.;

            f1 = autodiff(
                ForwardWithPrimal,
                Const(evaluate),
                Duplicated(prob, dprob),
                Const(direct),
                Duplicated(bie_solution, dbie_solution),
                Duplicated(x_test, dx_test_1),
            )

            f2 = autodiff(
                ForwardWithPrimal,
                Const(evaluate),
                Duplicated(prob, dprob),
                Const(direct),
                Duplicated(bie_solution, dbie_solution),
                Duplicated(x_test, dx_test_2),
            )

            g4 = [collect(f1[1][1]);; collect(f2[1][1])]

        end

        @show size(Γ)
        @show size(x_test)

        exact_gradient = solution_derivative(
            direct,
            x_test,
            Γ.x,
            Γ.n,
            Γ.w,
            data(τ),
            σ,
        )

        @test g4 ≈ exact_gradient atol=1e-5

        # @time begin
        #     g2 = spatial_gradient(Forward, prob, x_test, indirect, zeta)
        # end
        #
        # @time begin
        #     g3 = spatial_gradient(Enzyme.Reverse, prob, x_test, indirect, zeta)
        # end
        #
        # @test g2 ≈ g3

        # scatter!(ax, x_test[:, 1], x_test[:, 2]; color=u, markersize=20)
        # arrows2d!(ax, x_test[:, 1], x_test[:, 2], g2[:, 1], g2[:, 2]; lengthscale=0.1)
        # wait(display(fig))


        break

        # push!(
        #     num_solutions,
        #     DirichletSolution{Interior,Direct,Zeta}(
        #         n,
        #         u,
        #         τ,
        #         zeta,
        #         norm(u_exact - u, Inf),
        #         norm(τ_exact - τ, Inf)
        #     )
        # )

        # Dirichlet Zeta Indirect
        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            indirect,
            D,
            H_zeta,
            D_target,
        )
        push!(
            num_solutions,
            DirichletSolution{Interior,Indirect,Zeta}(
                n,
                u,
                τ,
                zeta,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )

        # Dirichlet Sidi Direct
        # hypersingular operator using Sidi's staggered grid
        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            direct,
            D_star,
            H_sidi,
            S_target,
            D_target,
        )

        push!(
            num_solutions,
            DirichletSolution{Interior,Direct,Sidi}(
                n,
                u,
                τ,
                sidi,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )
        # Dirichlet Sidi Indirect
        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            indirect,
            D,
            H_sidi,
            D_target,
        )
        push!(
            num_solutions,
            DirichletSolution{Interior,Indirect,Sidi}(
                n,
                u,
                τ,
                sidi,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )


        # Neumann problem
        # swap bdry conditions
        σ_exact = σ
        τ = τ_exact

        # Neumann Direct
        u, σ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Neumann(τ),
                interior,
                Γ
            ),
            direct,
            S,
            D,
            S_target,
            D_target,
        )
        # "recover constant" in the original code
        offset = u_exact[1] - u[1]
        u .+= offset
        data(σ) .+= offset # TODO: put this inside solver maybe and user passes integration constant
        push!(
            num_solutions,
            NeumannSolution{Interior,Direct}(
                n,
                u,
                σ,
                norm(u_exact - u, Inf),
                norm(σ_exact - σ, Inf))
        )

        # Neumann indirect
        u, σ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Neumann(τ),
                interior,
                Γ
            ),
            indirect,
            S,
            D_star,
            S_target,
        )
        # "recover constant" in the original code...
        offset = u_exact[1] - u[1]
        u .+= offset
        data(σ) .+= offset # TODO: put this inside solver maybe and user passes integration constant
        push!(
            num_solutions,
            NeumannSolution{Interior,Indirect}(
                n,
                u,
                σ,
                norm(u_exact - u, Inf),
                norm(σ_exact - σ, Inf)
            )
        )

    end

    return num_solutions


end



if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie
    wait(display(plot_errors(convergence_study())))
end
