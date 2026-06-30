
# TEST_LAP2D_HYPER_BIE
# Test hypersingular zeta-corrected trapezoidal rule for Laplace layer
# potentials on smooth geometries by solving the BVPs using a direct
# approach to BIE:
#      	int Laplace ansatz: u = S*(du/dn) - D*u
#       int Calder?n projection:     u =  (1/2-D)*u  +         S*(du/dn)
#                                du/dn =       -T*u  + (1/2+D^*)*(du/dn)
#      	ext Laplace ansatz: u = D*u - S*(du/dn) + omega
#       ext Calder?n projection:     u =  (1/2+D)*u  -         S*(du/dn)
#                                du/dn =        T*u  + (1/2-D^*)*(du/dn)
#
# c.f. Hsiao-Wendland 2008, Sec.1.3-1.4

using Revise


using CairoMakie
using BimDiff

abstract type Solution end
abstract type NumericalSolution{S,A} end



mutable struct DirichletSolution{S<:Side,A<:Approach,C<:HypersingularCorrection} <: NumericalSolution{S,A}
    n
    u::Vector{Float64}
    τ::Vector{Float64}
    correction::C
    u_err
    τ_err
end

struct NeumannSolution{S<:Side,A<:Approach} <: NumericalSolution{S,A}
    n
    u::Vector{Float64}
    σ::Vector{Float64}
    u_err
    σ_err
end

struct ExactSolution{S<:Side} <: Solution
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
{S<:Side,A<:Approach} = :magenta
get_color(::DirichletSolution{S,A,Zeta}) where
{S<:Side,A<:Approach} = :red
get_color(::NeumannSolution) = :blue

get_linestyle(::NumericalSolution{S,Direct}) where
{S<:Side} = :solid
get_linestyle(::NumericalSolution{S,Indirect}) where
{S<:Side} = :dash

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


function main()

    ord = 32       # pick desired convergence order of singular quad

    # useful constants
    laplace = Laplace()
    interior = Interior()
    exterior = Exterior()
    kapur_rokhlin = KapurRokhlin(ord)
    zeta = Zeta(ord)
    sidi = Sidi()
    direct = Direct()
    indirect = Indirect()

    # indicate how to reserve memory
    allocator = (_m, _n) -> Array{Float64}(undef, _m, _n)

    # set up source geometry (starfish domain)
    R = 1 # wobble center
    a = 0.3 # wobble amplitude
    w = 5 # wobble frequency
    function ρ(t)
        z = (R + a * cos.(w * t)) * cis(t) # parametrization of boundary
        return [real(z), imag(z)] # TODO: play with static arrays
    end

    # evenly distributed points in a circumference of radius r
    function ball(r, n)
        z = r .* cis.(2pi * (1:n) / n)
        return hcat(real.(z), imag.(z))
    end


    # fig, ax = visualize(m)

    # Interior Laplace BVPs
    # data generated with octave using seed 42
    # ns = 10;                          	% num of source points
    # s_ps.x = 1.5*exp(2i*pi*rand(ns,1));	% random source location
    # s_ps.w = 1;                         % dummy wei
    # den_source = randn(ns,1);           % random source densities
    n_source = 10 # 10 source points
    x_source = [
        -0.5695003215297076 1.387684900752891
        1.193361093146339 0.9087845186646686
        -1.382332571556837 -0.5823715837960004
        0.2246549945288565 1.483081296973716
        -1.49901371413742 0.05438643974317964
        1.411332357339733 -0.5080757592385936
        -1.03965505692555 1.081257306384161
        0.7266935884719004 -1.312218132961831
        1.262260923600564 0.810368657310395
        -0.6316051289445703 -1.360542157042888
    ]

    density_source = [                       # random source densities
        0.8286315202713013
        0.2222102135419846
        -0.1199957281351089
        0.5542055368423462
        1.894909262657166
        -1.461126089096069
        1.063002705574036
        -0.8932550549507141
        0.1896218359470367
        -0.4264606237411499
    ]
    Γ_source = DiscreteClosedCurve(x_source)

    n_test = 20
    x_test = ball(0.4, n_test)  # test points in inner domain
    Γ_test = DiscreteClosedCurve(x_test)

    S_manuf = SingleLayer(laplace, nothing, n_test, n_source; allocator=allocator)

    populate_matrices!(Γ_source, Γ_test, S_manuf)


    # matrix = compute_laplace_slp_matrix(x_test, x_source)
    u_exact = S_manuf * density_source # exact solution at test points

    u_exact_reference = [ # computed with octave
        -0.225720785940647
        -0.1532182381553945
        -0.07916247368635984
        -0.009436557766342876
        0.05058053245007847
        0.095561651179005
        0.1207276487177532
        0.122854434803948
        0.1008112512034587
        0.0556379263587953
        -0.008506676421622505
        -0.08391068100223156
        -0.1614855413528088
        -0.2333796644213197
        -0.2937316378516174
        -0.3378806310084885
        -0.3616183013158484
        -0.3616603501356365
        -0.33685823224991
        -0.2895297179342787
    ]


    @assert norm(u_exact - u_exact_reference) < 1e-15


    # scatter!(ax, x_test[:, 1], x_test[:, 2], color=u_exact)

    # wait(display(fig))

    # println("Printing max-norm errors")
    # println("Interior")

    n_vals = 20:20:400

    num_solutions = Vector{NumericalSolution}()

    for (i, n) ∈ enumerate(n_vals)

        Γ = DiscreteClosedCurve(n, ρ) # boundary of the domain

        # fig = visualize(Γ)
        # wait(display(fig))
        # break


        D_star_source = AdjointDoubleLayer(laplace, n, n_source; allocator) # ok
        S_source = SingleLayer(laplace, nothing, n, n_source; allocator) # ok

        populate_matrices!(Γ_source, Γ, S_source, D_star_source)

        σ = S_source * density_source # Dirichlet BC
        τ_exact = D_star_source * density_source # Neumann BC exact solution


        S = SingleLayer(laplace, kapur_rokhlin, n, n; allocator) # ok
        D = DoubleLayer(laplace, n, n; allocator) # ok
        D_star = AdjointDoubleLayer(laplace, n, n; allocator)  # ok
        H_zeta = Hypersingular(laplace, zeta, n, n; allocator) # ok
        H_sidi = Hypersingular(laplace, sidi, n, n; allocator) # ok

        populate_matrices!(Γ, S, D, D_star, H_sidi, H_zeta)

        S_target = SingleLayer(laplace, nothing, n_test, n; allocator) # ok
        D_target = DoubleLayer(laplace, n_test, n; allocator) # ok
        populate_matrices!(Γ, Γ_test, S_target, D_target)


        # Dirichlet Zeta Direct
        u, τ = solve(
            laplace,
            interior,
            Dirichlet(σ), # TODO: since one kernel matrix can be applied to several BCs, overload accepting vector of BC
            direct,
            D_star,
            H_zeta,
            S_target,
            D_target,
        )
        push!(
            num_solutions,
            DirichletSolution{Interior,Direct,Zeta}(
                n,
                u,
                τ,
                zeta,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )

        # Dirichlet Zeta Indirect
        u, τ = solve(
            laplace,
            interior,
            Dirichlet(σ),
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
        u, τ = solve(
            laplace,
            interior,
            Dirichlet(σ),
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
        u, τ = solve(
            laplace,
            interior,
            Dirichlet(σ),
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

        u, σ = solve(
            laplace,
            interior,
            Neumann(τ),
            direct,
            S,
            D,
            S_target,
            D_target,
        )
        # "recover constant" in the original code...
        offset = u_exact[1] - u[1]
        u .+= offset
        σ .+= offset # TODO: put this inside solver maybe and user passes integration constant
        push!(
            num_solutions,
            NeumannSolution{Interior,Direct}(
                n,
                u,
                σ,
                norm(u_exact - u, Inf),
                norm(σ_exact - σ, Inf))
        )

        u, σ = solve(
            laplace,
            interior,
            Neumann(τ),
            indirect,
            S,
            D_star,
            S_target,
        )
        # "recover constant" in the original code...
        offset = u_exact[1] - u[1]
        u .+= offset
        σ .+= offset # TODO: put this inside solver maybe and user passes integration constant
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

    # wait(display(fig))

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
