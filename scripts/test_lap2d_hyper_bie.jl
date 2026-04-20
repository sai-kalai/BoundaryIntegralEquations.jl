
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




ord = 32;       # pick desired convergence order of singular quad

# set up source geometry (starfish domain)
R = 1 # wobble center
a = .3 # wobble amplitude
w = 5 # wobble frequency

function rho(t)
    z = (R + a.* cos.(w .* t)) .* exp.(1im .* t) # parametrization of boundary

    return real(z), imag(z)

end

# evenly distributed points in a circumference of radius r
function ball(r, n)
    z =  r .* exp.(1im * collect(range(0, stop=2 * pi, length=n)))[1:end-1]
    return hcat(real.(z), imag.(z))
end


m = Manifold{Float64}(100, rho)

fig = visualize(m)

# wait(display(fig))


# Interior Laplace BVPs
# generate (random) reference solution at test points
n_source = 10                          	                  # num of source points
x_source = (R + a * 2) .* exp.(2im * pi * rand(n_source)) # random source locations
density_source = randn(n_source, 1)                       # random source densities

x_test = ball(R - a * 1.5, 20);  	# test points in inner domain


# uexac=LapSLPmat(t,s_ps)*den_source; # ref soln at test pts

display(typeof(x_test))
display(typeof(m))
u_exact = LaplaceSLP(x_test, m)
display(u_exact)
