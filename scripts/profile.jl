

using JLD2
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using ProfileView
using GLMakie

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

n_domain = 200
n_bdry = 400
laplace = Laplace()
Γ = DiscreteClosedCurve(n_bdry, starfish)
xmin, xmax, ymin, ymax = extrema(Γ)
xs = range(xmin, xmax, length=n_domain)
ys = range(ymin, ymax, length=n_domain)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)
# x_dense = Fixtures.test_locations()

Γ_source, density_source, u_exact = manufactured_solution(laplace, x_dense)

# bc, correction = Dirichlet(SingleLayer(laplace, Γ_source, Γ.x; populate_matrix=true) * density_source), Zeta(32)
bc, correction = Neumann(AdjointDoubleLayer(laplace, Γ_source, Γ.x, Γ.n; populate_matrix=true) * density_source), KapurRokhlin(32)


pb = BoundaryValueProblem(laplace, bc, Interior(), Γ)

approach = Indirect()

cutoff_vals = [
    0.0,
    # 0.01,
    # 0.05,
    # 0.1,
    # 0.25,
]

#force compilation
solve_and_evaluate(
    pb,
    approach,
    correction,
    x_dense[:, 1:4],
    # 0.0,
)


for c in cutoff_vals
    @show c
    # b = @benchmark begin
    @profview begin
        # begin
        u, cauchy_data = solve_and_evaluate(
            pb,
            approach,
            correction,
            x_dense,
            # $c
        )

        # val = (u - u_exact) .|> abs .|> ((x)->x + eps(eltype(u))) .|> log10
        # @show extrema(val[.!isnan.(val)])
        #
        # fig, ax, im = scatter(x_dense; color=val, colormap=:viridis)
        # Colorbar(fig[1, 2], im, ticks=LinearTicks(10))
        #
        # fig |> display |> wait

    end
    # display(b)
end
