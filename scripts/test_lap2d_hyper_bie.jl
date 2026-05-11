
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

using CairoMakie
using LinearAlgebra
using Random
using BimDiff


struct Result
    n_vals::Vector
    err::Vector
    label::String
    color::String
    linestyle::String
    order::Union{Int32,Nothing}

end

function main()

    Random.seed!(42) # seed rng for reproducibility

    ord = 7       # pick desired convergence order of singular quad

    # set up source geometry (starfish domain)
    R = 1 # wobble center
    a = 0.3 # wobble amplitude
    w = 5 # wobble frequency
    function ρ(t)
        z = (R + a .* cos.(w .* t)) .* exp.(1im .* t) # parametrization of boundary
        return real(z), imag(z)
    end

    # evenly distributed points in a circumference of radius r
    function ball(r, n)
        z = r .* exp.(1im * 2pi * (1:n) / n)
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


    # println("matrix: ")
    # display(matrix)
    #
    # println("u_exact: ")
    # display(u_exact)

    # scatter!(ax, x_test[:, 1], x_test[:, 2], color=u_exact)

    # wait(display(fig))

    println("Printing max-norm errors")
    println("Dirichlet problem")
    println("Interior")

    n_vals = 20:20:400

    err_u = zeros(Float64, size(n_vals, 1))
    err_τ = zeros(Float64, size(n_vals, 1))

    err_u_sidi = zeros(Float64, size(n_vals, 1))
    err_τ_sidi = zeros(Float64, size(n_vals, 1))

    for (i, n) ∈ enumerate(n_vals)

        Γ = Manifold(n, ρ) # boundary of the domain

        # fig = visualize(Γ)
        # wait(display(fig))


        # compute boundary conditions from manufactured solution
        # special case: using manifold as target and using manifold normals as source to compute the manufactured solution data
        S_source, dS_dn_source = compute_laplace_slp_matrix_and_normal_derivative(
            Γ.x, # locations to evaluate derivative
            x_source, # integration variable
            Γ.n # normals at the locations
        )

        σ = S_source * density_source # Dirichlet BC
        τ_exact = dS_dn_source * density_source # Neumann BC exact solution


        # operators with quadrature weights applied
        S = compute_laplace_slp_matrix(x_test, Γ.x) .* Γ.w'
        D = compute_laplace_dlp_matrix(x_test, Γ.x, Γ.n) .* Γ.w'

        D_star = compute_laplace_slp_matrix_normal_derivative(Γ.x, Γ.n, vec(Γ.k))
        D_star .*= Γ.w' # apply quadrature weights
        A = -0.5 * I(n) + D_star


        # direct approach

        # hypersingular operator using zeta quadrature
        H = compute_laplace_dlp_matrix_normal_derivative(
            Γ.x,
            Γ.n,
            vec(Γ.k),
            Γ.w,
            ord
        )

        τ = A \ (H * σ)
        u = S * τ - D * σ


        # hypersingular operator using Sidi's staggered grid
        H_sidi = compute_laplace_dlp_matrix_normal_derivative(Γ.x, Γ.n)
        H_sidi = H_sidi .* Γ.w' # transpose seems hacky
        # divide diagonal
        H_sidi[diagind(H_sidi)] .= -pi / 4 ./ Γ.w
        # apply quadrature weights and integrate function σ
        τ_sidi = A \ (H_sidi * σ)
        u_sidi = S * τ_sidi - D * σ

        err_u[i] = norm(u_exact - u, Inf)
        err_u_sidi[i] = norm(u_exact - u_sidi, Inf)
        err_τ[i] = norm(τ_exact - τ, Inf)
        err_τ_sidi[i] = norm(τ_exact - τ_sidi, Inf)

        println("N=$n\tu(zeta)=$(err_u[i])\tu(sidi)=$(err_u_sidi[i])\tτ(zeta)=$(err_τ[i])\tτ(sidi)=$(err_τ_sidi[i])")

        # Neumann problem

        # TODO: separate into functions

        # swap bdry conditions
        σ_exact = σ
        τ = τ_exact

        D = compute_laplace_dlp_matrix(Γ.x, Γ.n, vec(Γ.k))
        S = compute_laplace_slp_matrix(Γ.x, vec(Γ.w))


        exit()



    end

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xscale=log10,
        yscale=log10,
        xlabel="x",
        ylabel="y"
    )


    order_offset = ord - 2
    lines!(
        ax,
        n_vals,
        (n_vals ./ (5 * n_vals[1])) .^ float(-order_offset),
        label="O(h^$(order_offset))",
        linestyle=:dash,
        color=:black)

    exponential_decay = 0.1
    lines!(ax,
        n_vals,
        exp.(-exponential_decay .* (n_vals .- n_vals[1])),
        label="O(exp(-$exponential_decay / h))",
        linestyle=:dot,
        color=:black
    )

    scatterlines!(ax, n_vals, err_u, label="u zeta $(ord)-th order")
    scatterlines!(ax, n_vals, err_u_sidi, label="u sidi")
    scatterlines!(ax, n_vals, err_τ, label="τ zeta $(ord)-th order")
    scatterlines!(ax, n_vals, err_τ_sidi, label="τ sidi")

    axislegend(ax)

    wait(display(fig))

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
