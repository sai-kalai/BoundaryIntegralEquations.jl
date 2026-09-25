docstring = """
Usage:
    ```shell
    julia scripts/plot/$(basename(@__FILE__))
    ```
    ```julia
    julia> include("scripts/plot/$(basename(@__FILE__)))
    ```

Plot the setup used for convergence analysis of value problems
"""


using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools
using GLMakie
using CairoMakie



@info docstring

###########
# Acquire data
###########
const laplace = Laplace()

include("utils.jl")

# location of test points used for convergence study
x_test = Fixtures.test_locations()
# locations inside the domain for plotting solution
x_plot = [
    x_test;;
    ball(0.1, 10);;
    ball(0.3, 30);;
    ball(0.6, 60);;
    stack((t) -> starfish(t, 0.9), 0:0.1:2pi)
]

# obtain manufactured solution information
Γ_source, density_source, _ = manufactured_solution(laplace, x_plot)

# define domain boundary
n_boundary = 200
Γ = DiscreteClosedCurve(n_boundary, starfish)

# compute dirichlet data (SL potential of point sources at boundary)
S_source = SingleLayer(laplace, Γ_source, Γ.x; populate_matrix=true)
σ = S_source * density_source

# define problem
pb = BoundaryValueProblem(laplace, Dirichlet(σ), Interior(), Γ)

# solve using 32nd order FD correction and 5% cutoff for Cauchy integral eval
u, _ = solve_and_evaluate(pb, Indirect(), Zeta(32), x_plot, 0.05)


###########
# Make figure
###########
fig = Figure()
ax = Axis(
    fig[1, 1],
    ;
    aspect=DataAspect(),
)

###########
# Add plots
###########
# plot shape of boundary
visualize!(ax, Γ, false, false; linewidth=3)

# plot solution at internal points
cof = tricontourf!(
    ax,
    Γ,
    x_plot,
    u,
    σ,
    mode=:relative,
    levels=0:0.1:1,)


# plot test points used for convergence study
sc0 = scatter!(
    ax, x_test,
    label="Test Locations", strokewidth=1, color=:red,
    marker=:star4, strokecolor=:black,
)

# plot locations of point sources
sc1 = scatter!(
    ax, Γ_source.x,
    label="Point Sources",
    strokewidth=1,
    color=data(density_source),
    # marker=:star8,
    markersize=15,
    colormap=:bwr,
)

tks = LinearTicks(7)

# add colorbars and legend
Colorbar(
    fig[1, 2], sc1, label="Charge",
    ticks=tks,
    tellheight=false,
    # vertical=false,
)
Colorbar(
    fig[1, 3], cof, label="Potential",
    ticks=tks,
    tellheight=false,
    # vertical=false,
)

axislegend(ax, position=:lb, patchsize=(15, 10), labelsize=15)


###########
# Save and display plot
###########
save_and_display!(basename(@__FILE__), fig)
