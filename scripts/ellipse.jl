
using Revise
using CairoMakie
using LinearAlgebra
using StaticArrays

using BimDiff



function main()


    γ = deg2rad(30.)
    x0 = 2.
    y0 = 3.
    a = 5.
    b = 2.

    function rotated_ellipse(θ)

        mat = SA[
            cos(θ)*cos(γ) sin(θ)*sin(γ)
            cos(θ)*sin(γ) sin(θ)*cos(γ)
        ]
        SA[x0, y0] + mat * SA[a, b]
    end

    function exact_solution(x, y)
        xi = (x - x0) * cos(γ) - (y - y0) * sin(γ)
        eta = (x - x0) * sin(γ) + (y - y0) * cos(γ)
        return SA[xi/a, eta/b]
    end

    x_test = [x0 y0]

    n_vals = 20:20:400

    ord = 32

    laplace = Laplace()
    interior = Interior()
    zeta = Zeta(ord)
    sidi = Sidi()
    indirect = Indirect()

    for (i, n) in enumerate(n_vals)
        # parameter of curve
        θ = range(0, 2π; length=n + 1)[1:end-1]
        Γ = DiscreteClosedCurve(θ, rotated_ellipse)

        fig, ax = visualize(Γ)


        D = DoubleLayer(laplace, Γ)
        H_zeta = Hypersingular(laplace, Γ, zeta)
        H_sidi = Hypersingular(laplace, Γ, sidi)

        D_target = DoubleLayer(laplace, x_test, Γ,)

        xi, _ = solve(
            laplace,
            interior,
            Dirichlet(cos.(θ)),
            indirect,
            D,
            H_zeta,
            D_target
        )

        eta, _ = solve(
            laplace,
            interior,
            Dirichlet(sin.(θ)),
            indirect,
            D,
            H_zeta,
            D_target
        )

        @show xi, eta

        wait(display(fig))
        break
    end

end
