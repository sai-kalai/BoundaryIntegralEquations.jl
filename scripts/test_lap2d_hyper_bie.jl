
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

using GLMakie
using BimDiff

using LinearAlgebra
using Random

Random.seed!(42)

ord = 32;       # pick desired convergence order of singular quad

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


# display(typeof(ρ))
# display(ρ)
# m = Manifold(100, ρ)
# fig, ax = visualize(m)



# Interior Laplace BVPs
# data generated with octave using seed 42
# n_source = 10                                            # num of source points
# x_source = (1.5) .* exp.(2im * pi * rand(n_source)) # random source locations
# x_source = hcat(real.(x_source), imag.(x_source))

x_source = [
    1.193361093146339 0.9087845186646686
    -1.382332571556837 -0.5823715837960004
    0.2246549945288565 1.483081296973716
    -1.49901371413742 0.05438643974317964
    1.411332357339733 -0.5080757592385936
    -1.03965505692555 1.081257306384161
    0.7266935884719004 -1.312218132961831
    1.262260923600564 0.810368657310395
    -0.6316051289445703 -1.360542157042888
    0.7139358851174307 -1.319202619744818
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

x_test = ball(0.4, 20);  # test points in inner domain

# println("density source:")
# display(density_source)
#
# println("x source:")
# display(x_source)
#
# println("x test:")
# display(x_test)

# uexac=LapSLPmat(t,s_ps)*den_source; # ref soln at test pts
matrix = compute_laplace_slp_matrix(x_test, x_source)
u_exact = matrix * density_source # exact solution at test points

u_exact_reference = [ # computed with octave
    -0.061846802418295
    -0.109851942730244
    -0.1575289208119623
    -0.2008088926380286
    -0.236092249641929
    -0.2595403469271642
    -0.2674033713559268
    -0.2578848316373349
    -0.2333258235977643
    -0.1995135476533102
    -0.1622352996967466
    -0.1248436126291707
    -0.0885177087222776
    -0.05379066211791295
    -0.02182539782726936
    0.004835351522468149
    0.02216412292532798
    0.02554376622225894
    0.01164879560087229
    -0.01910867039340977
]

@assert norm(u_exact - u_exact_reference) < 1e-15


# println("matrix: ")
# display(matrix)
#
# println("u_exact: ")
# display(u_exact)

# scatter!(ax, x_test[:, 1], x_test[:, 2], color=u_exact)

# wait(display(fig))

println("Dirichlet problem")

n_vals = 20:20:400

for n ∈ n_vals

    display(n)

    Γ = Manifold(n, ρ) # boundary of the domain

    # fig = visualize(Γ)
    # wait(display(fig))

    display("density_source")
    display(density_source)

    # compute boundary conditions from manufactured solution
    # special case: using manifold as target and using manifold normals as source to compute the manufactured solution data
    S, dS_dN = compute_laplace_slp_matrix_and_normal_derivative(
        Γ.x, # locations to evaluate derivative
        x_source, # integration variable
        Γ.n # normals at the locations
    )

    display("S")
    display(S)

    display("dS_dN")
    display(dS_dN)

    σ = S * density_source # Dirichlet BC
    τ = dS_dN * density_source # Neumann BC

    display("σ")
    display(σ)

    display("τ")
    display(τ)


    # direct approach

    At = compute_laplace_dlp_matrix_normal_derivative(Γ.x, Γ.n)
    display("At")
    display(At)






    exit()

end
