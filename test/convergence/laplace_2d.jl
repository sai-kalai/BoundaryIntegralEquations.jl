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
# cf. Hsiao-Wendland 2008, Sec.1.3-1.4

# Acquire convergence data for solution of Laplace BVPs with several solver parameters

using Test
using Revise
using StaticArrays
using GLMakie
using LinearAlgebra
using BenchmarkTools
using JLD2

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools
using BoundaryIntegralEquations.DevTools: Fixtures

# 1. run simulations
#
# 2. check that convergence rate is as expected

x_test = Fixtures.test_locations()
x_test = [
    x_test;;
    ball(0.1, 10);;
    ball(0.3, 30);;
    ball(0.6, 60);;
    # avoid  testing close evaluation for gradient
    stack((t) -> starfish(t, 0.8), 0:0.1:2pi);;
    stack((t) -> starfish(t, 0.9), 0:0.1:2pi);;
]

result = run_all_simulations(
    x_test;
    n_vals=collect(100:40:400),
    cutoff_vals=[0.0, 0.1],
    fd_acc_vals=[4, 8, 16, 32,],
    bc_types=[
        Neumann,
        Dirichlet,
    ],
    approach_types=[
        Direct,
        Indirect,
    ],
)

# TODO: this could be automated following path of script
# There's an uncomfortable coupling between the names of plotting files in scripts/
# and the names of files in test/convergence and benchmark/
# maybe config file to unify
const F = "data/convergence_laplace_2d.jld2"

save_object(F, result)

@info "result = "
display(result)

@info "`result` saved to `$F`"

