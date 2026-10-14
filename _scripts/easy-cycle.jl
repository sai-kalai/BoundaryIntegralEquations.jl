begin
    f=Figure();
    a=Axis(
        f[1, 1],
        palette=(;
            color=Makie.to_colormap(:tab10),
            marker=[:circle, :xcross],
            linestyle=[
                :dash,
                :dot,
                :dashdot,
                :dashdotdot,
            ]
        )
    )
    for i in 1:10
        scatterlines!(
            a,
            [1, 2, 3], rand(3) .+ i/100,
            label=i|>string,
            strokewidth=2,
            markercolor=:transparent,
            cycle=Cycle([
                    :marker,
                    [:color, :strokecolor, :linecolor]=>:color,
                    # :linestyle
                ],
                ;
                covary=true),
        )
    end
    axislegend(a)
    f
end

