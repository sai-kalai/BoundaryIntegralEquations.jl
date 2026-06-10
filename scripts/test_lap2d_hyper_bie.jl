
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
using LinearAlgebra
using Random
using BimDiff



abstract type Solution end

struct DirichletSolution{S<:Side,A<:Approach,C<:HypersingularCorrection} <: Solution
    n
    u::Vector{Float64}
    τ::Vector{Float64}
    correction::C
end

struct NeumannSolution{S<:Side,A<:Approach} <: Solution
    n
    u::Vector{Float64}
    σ::Vector{Float64}
end

struct ExactSolution{S<:Side} <: Solution
    n
    u::Vector{Float64}
    σ::Vector{Float64}
    τ::Vector{Float64}
end

# struct SolutionSequence{S<:Solution}
#     data::Vector{S}
#     SolutionSequence{S}(n::Int) = SolutionSequence(Vector{S}(undef, n))
# end


function main()

    Random.seed!(42) # seed rng for reproducibility

    ord = 32       # pick desired convergence order of singular quad

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

    x_test = ball(0.4, 20)  # test points in inner domain

    matrix = compute_laplace_slp_matrix(x_test, x_source)
    u_exact = matrix * density_source # exact solution at test points

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

    err_u = zeros(Float64, size(n_vals, 1))
    err_τ = zeros(Float64, size(n_vals, 1))

    err_u_sidi = zeros(Float64, size(n_vals, 1))
    err_τ_sidi = zeros(Float64, size(n_vals, 1))

    err_neumann_u = zeros(Float64, size(n_vals, 1))
    err_neumann_σ = zeros(Float64, size(n_vals, 1))

    solutions = Vector{Solution}()

    laplace = Laplace()
    interior = Interior()
    exterior = Exterior()
    zeta = Zeta(ord)
    sidi = Sidi()
    direct = Direct()
    indirect = Indirect()

    for (i, n) ∈ enumerate(n_vals)


        Γ = DiscreteClosedCurve(n, ρ) # boundary of the domain

        # fig = visualize(Γ)
        # wait(display(fig))
        # break


        # compute boundary conditions from manufactured solution
        # special case: using manifold as target and using manifold normals as source to compute the manufactured solution data
        S_source, dS_dn_source = compute_laplace_slp_matrix_and_normal_derivative(
            Γ.x, # locations to evaluate derivative
            x_source, # integration variable
            Γ.n # normals at the locations
        )

        σ = S_source * density_source # Dirichlet BC
        τ_exact = dS_dn_source * density_source # Neumann BC exact solution

        push!(solutions, ExactSolution{Interior}(n, u_exact, σ, τ_exact)) # TODO: may be wasteful to store the same exact solution many times, although boundary traces do change...




        S = SingleLayer(laplace, Γ, ord)
        D = DoubleLayer(laplace, Γ)
        D_star = AdjointDoubleLayer(laplace, Γ)
        H_zeta = Hypersingular(laplace, Γ, zeta)
        H_sidi = Hypersingular(laplace, Γ, sidi)

        S_target = SingleLayer(laplace, x_test, Γ,)
        D_target = DoubleLayer(laplace, x_test, Γ,)

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
        push!(solutions, DirichletSolution{Interior,Direct,Zeta}(n, u, τ, zeta))
        err_u[i] = norm(u_exact - u, Inf)
        err_τ[i] = norm(τ_exact - τ, Inf)

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
        push!(solutions, DirichletSolution{Interior,Direct,Sidi}(n, u, τ, sidi))
        err_u_sidi[i] = norm(u_exact - u, Inf)
        err_τ_sidi[i] = norm(τ_exact - τ, Inf)

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
        push!(solutions, NeumannSolution{Interior,Direct}(n, u, σ))


        err_neumann_u[i] = norm(u_exact - u, Inf)
        err_neumann_σ[i] = norm(σ_exact - σ, Inf)


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
        push!(solutions, NeumannSolution{Interior,Indirect}(n, u, σ))


    end

    # println("Dirichlet")
    # for (i, n) in enumerate(n_vals)
    #     println("N=$n\tu(zeta)=$(err_u[i])\tu(sidi)=$(err_u_sidi[i])\tτ(zeta)=$(err_τ[i])\tτ(sidi)=$(err_τ_sidi[i])")
    # end
    #
    # println("Neumann")
    # for (i, n) in enumerate(n_vals)
    #     println("N=$n\tu=$(err_neumann_u[i])\tσ=$(err_neumann_σ[i])")
    # end

    # fig = Figure()
    # ax = Axis(
    #     fig[1, 1],
    #     xscale=log10,
    #     yscale=log10,
    #     xlabel="x",
    #     ylabel="y"
    # )
    # order_offset = ord - 2
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
    # scatterlines!(ax, n_vals, err_u, label="u zeta $(ord)-th order")
    # scatterlines!(ax, n_vals, err_u_sidi, label="u sidi")
    # scatterlines!(ax, n_vals, err_τ, label="τ zeta $(ord)-th order")
    # scatterlines!(ax, n_vals, err_τ_sidi, label="τ sidi")
    # scatterlines!(ax, n_vals, err_neumann_u, label="Neumann u ")
    # scatterlines!(ax, n_vals, err_neumann_σ, label="Neumann σ ")
    # axislegend(ax)

    # wait(display(fig))

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
