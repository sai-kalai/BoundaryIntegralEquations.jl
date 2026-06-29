
using Revise
using LinearAlgebra
using StaticArrays
using GLMakie
using Statistics
using PolygonOps

GLMakie.activate!()


using BimDiff


"""
    main()

[TODO:description]

"""
function main()

    γ = deg2rad(30.)
    x0 = 2.
    y0 = 3.
    a = 5.
    b = 2.

    function rotated_ellipse(θ, x0, y0, a, b, γ)

        mat = SA[
            cos(θ)*cos(γ) -sin(θ)*sin(γ)
            cos(θ)*sin(γ) sin(θ)*cos(γ)
        ]
        SA[x0, y0] + mat * SA[a, b]
    end

    """
        exact_solution(xy)

    compute the xi, eta values for given pair of xy values

    # Arguments
    - `xy`: two-element array containing the x and y coordinates
    """
    function exact_solution(xy, x0, y0, a, b, γ)
        x, y = xy[1], xy[2]
        xi = (x - x0) * cos(γ) + (y - y0) * sin(γ)
        eta = (x - x0) * -sin(γ) + (y - y0) * cos(γ)
        return SA[xi/a, eta/b]
    end

    n_grid = 200
    θ_grid = range(0, 2π; length=n_grid + 1)[1:(end-1)]
    boundary_grid = Matrix(stack((t) -> rotated_ellipse(t, x0, y0, a, b, γ), θ_grid)')

    xmin, ymin = minimum(boundary_grid, dims=1) |> vec
    xmax, ymax = maximum(boundary_grid, dims=1) |> vec

    xs = range(xmin, xmax, length=n_grid)
    ys = range(ymin, ymax, length=n_grid)

    x_test_all = reduce(vcat, [[x y] for y in ys, x in xs])
    xi_eta_exact_all = stack(
        (t) -> exact_solution(t, x0, y0, a, b, γ),
        eachrow(x_test_all);
        dims=1
    )


    xi_eta_exact_boundary = stack(
        (t) -> exact_solution(t, x0, y0, a, b, γ),
        eachrow(boundary_grid);
        dims=1
    )

    n_vals = 20:80:200
    errs = zeros(Float64, size(n_vals, 1))

    ord = 32

    laplace = Laplace()
    interior = Interior()
    zeta = Zeta(ord)
    sidi = Sidi()
    indirect = Indirect()


    # TODO: combine both scripts

    # regular mesh for evaluating exact solution


    for (i, n) in enumerate(n_vals)

        # Define boundary
        θ = range(0, 2π; length=n + 1)[1:(end-1)]
        Γ = DiscreteClosedCurve(θ, (t) -> rotated_ellipse(t, x0, y0, a, b, γ))


        # check which points are inside
        poly = [[row[1], row[2]] for row in eachrow(Γ.x)]
        push!(poly, Γ.x[1, :])

        mask = [
            inpolygon((x_test_all[i, 1], x_test_all[i, 2]), poly) == 1
            for i in axes(x_test_all, 1)
        ]

        x_test = x_test_all[mask, :]
        xi_eta_exact = xi_eta_exact_all[mask, :]

        D = DoubleLayer(laplace, Γ)
        S = SingleLayer(laplace, Γ, ord)
        H_zeta = Hypersingular(laplace, Γ, zeta)
        H_sidi = Hypersingular(laplace, Γ, sidi)

        D_target = DoubleLayer(laplace, x_test, Γ,)
        S_target = SingleLayer(laplace, x_test, Γ,)

        # TODO: work with vector valued functions
        bc_xi = Dirichlet(cos.(θ))
        bc_eta = Dirichlet(sin.(θ))


        # xi, _ = solve(
        #     laplace,
        #     interior,
        #     bc_xi,
        #     indirect,
        #     D,
        #     H_zeta,
        #     D_target
        # )
        #
        # eta, _ = solve(
        #     laplace,
        #     interior,
        #     bc_eta,
        #     indirect,
        #     D,
        #     H_zeta,
        #     D_target
        # )

        # close evaluation

        phi_xi = solve_bie(
            laplace,
            interior,
            bc_xi,
            D,
        )

        phi_eta = solve_bie(
            laplace,
            interior,
            bc_eta,
            D,
        )

        v_lim_xi = compute_boundary_limit(
            interior,
            D,
            phi_xi,
            Γ
        )
        v_xi = cauchy_integral(
            x_test,
            Γ,
            v_lim_xi
        )
        xi = real.(v_xi)

        v_lim_eta = compute_boundary_limit(
            interior,
            D,
            phi_eta,
            Γ
        )

        v_eta = cauchy_integral(
            x_test,
            Γ,
            v_lim_eta
        )
        eta = real.(v_eta)

        xi_eta_num = hcat(xi, eta)

        # f = Figure()
        # a = Axis(f[1, 1]; aspect=DataAspect())
        # scatterlines!(a, real.(v_lim_xi), real.(v_lim_eta); markersize=4)
        # scatter!(a, xi, eta; markersize=4)
        # scatter!(a, xi_eta_exact...; markersize=4)
        # return f
        # return scatterlines(real.(v_lim_xi), real.(v_lim_eta); markersize=10)

        # Nx2
        e = xi_eta_num .- xi_eta_exact
        # display(xi)
        # display(eta)
        # display(xi_eta_num)
        # display(e)

        # break

        # Nx1, store euclidean norm of error for each point
        e_norm = norm.(eachrow(e), 2) .+ eps(Float64)

        # 1x1
        errs[i] = mean(e_norm .^ 2)

        @show i, errs[i]

        # fig, ax = visualize(Γ)
        #
        # scatter_kwargs = (; colorscale=log10, color=e_norm, markersize=7, colormap=:viridis)
        #
        # sc1 = scatter!(ax, x_test[:, 1], x_test[:, 2]; scatter_kwargs...)
        #
        # ax2 = Axis(fig[1, 2]; aspect=DataAspect(), title="n = $n")
        #
        # lines!(ax2, xi_eta_exact_boundary[:, 1], xi_eta_exact_boundary[:, 2]; color=:black)
        #
        # sc2 = scatter!(ax2, xi_eta_exact[:, 1], xi_eta_exact[:, 2]; scatter_kwargs...)
        #
        # Colorbar(fig[1, 2][1, 3], sc2; label="log10 error inf norm")
        #
        # wait(display(fig))
    end
    #
    # fig3 = Figure()
    # ax3 = Axis(fig3[1, 1]; xscale=log10, yscale=log10)
    #
    # scatterlines!(ax3, n_vals, errs)
    # lines!(ax3, n_vals, (n_vals ./ n_vals[1]) .^ (-2), label="O(h^-2)")
    # lines!(ax3, n_vals, (n_vals ./ n_vals[1]) .^ (-1), label="O(h^-1)")
    # axislegend(ax3)

    # wait(display(fig3))

end
