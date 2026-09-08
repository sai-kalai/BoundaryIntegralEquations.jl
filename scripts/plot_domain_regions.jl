begin
    n = 200
    Γ = DiscreteClosedCurve(n, starfish)
    xmin, xmax, ymin, ymax = extrema(Γ_dense)
    xs = range(xmin, xmax, length=n)
    ys = range(ymin, ymax, length=n)
    iter = Iterators.product(xs, ys)
    x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

    d=0.05
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    nr, fr, bd = classify(Γ, x_dense, Exterior(), d)
    scatter!(ax, x_dense[:, nr], color=:blue, alpha=0.1)
    scatter!(ax, x_dense[:, fr], color=:red, alpha=0.1)
    scatter!(ax, x_dense[:, bd], color=:green, alpha=0.1)
    visualize!(ax, Γ, false, false)
    fig

end

