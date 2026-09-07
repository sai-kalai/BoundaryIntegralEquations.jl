

using JLD2
using StaticArrays
using LinearAlgebra
using ProfileView
using GLMakie

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

n = 200
laplace = Laplace()
correction = Zeta(32)
Γ_dense = DiscreteClosedCurve(n, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n)
ys = range(ymin, ymax, length=n)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

Γ_source, density_source, u_exact = manufactured_solution(laplace, x_dense)

Γ = DiscreteClosedCurve(n, starfish)

bc = Dirichlet(SingleLayer(laplace, Γ_source, Γ.x; populate_matrix=true) * density_source)

bvp = BoundaryValueProblem(laplace, bc, Interior(), Γ)

cutoff_vals = [0.0, 0.01, 0.05, 0.1, 0.25]

indirect = Indirect()
# force compilation
solve_and_evaluate(
    bvp,
    indirect,
    correction,
    x_dense[:, 1:2],
    0.0,
)

for c in cutoff_vals
    # @profview begin
    u, cauchy_data = solve_and_evaluate(
        bvp,
        Indirect(),
        correction,
        x_dense,
        c
    )

    # val = (u - u_exact) .|> abs .|> log10
    #
    # fig, ax, im = image(reshape(val, n, n); colormap=:viridis)
    # Colorbar(fig[1, 2], im)
    #
    # fig |> display |> wait
    # end
end
