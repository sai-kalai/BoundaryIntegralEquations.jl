# investigate the numerical conditioning of operators

using LinearAlgebra
using DataFrames


using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

include("plot/utils.jl")

function make_operators(n_vals=100:20:800)

    ops=[]
    for n in n_vals

        ops2=[]
        c = DiscreteClosedCurve(n, starfish)

        append!(
            ops2,
            [DoubleLayer(Laplace(), c)
                AdjointDoubleLayer(Laplace(), c)
                Hypersingular(Laplace(), c, Sidi())]
        )

        for o in [8, 32]
            append!(
                ops2,
                [SingleLayer(Laplace(), c, KapurRokhlin(o))
                    Hypersingular(Laplace(), c, Zeta(o))]
            )
        end
        populate_matrices!(c, ops2...)
        append!(ops, ops2)
    end

    BoundaryIntegralEquations.panic_if_garbage(ops...)

    return ops
end

function make_row(o)

    if o isa AdjointDoubleLayer
        letter="D^*"
    else
        letter=(o|>typeof|>nameof|>string)[1]
    end

    m=BoundaryIntegralEquations.matrix(o);
    if (o isa DoubleLayer || o isa AdjointDoubleLayer)
        m = 0.5I-m
        opname = L"\pm \left(\frac{1}{2}-\mathcal{%$letter}\right)"
    else
        opname = L"\mathcal{%$letter}"
    end

    n = size(m, 2)

    return (;
        pts=n,
        operator=opname,
        correction=correction(o),
        cond=cond(m),
    )
end

function group_name(operator, correction)
    name = string(operator)

    if isnothing(correction)
        name
    elseif correction isa Sidi
        L"%$name (Richardson)"
    elseif correction isa Zeta
        L"%$name (FD(%$(correction.order))"
    elseif correction isa KapurRokhlin
        L"%$name (KR(%$(correction.order))"
    else
        "$name ($(typeof(correction)))"
    end
end

function make_dataframe(ops)
    df = DataFrame(make_row.(ops))
    df.group = group_name.(df.operator, df.correction)
    df
end


function plot_conditioning(df)
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="N",
        ylabel="condition number",
        yscale=log10,
        xscale=log10,
        xtickformat=values -> [@sprintf("%d", v) for v in values],
        palette=get_palette(2),
    )
    veerapaneni
    e = extrema(df.pts)
    ax.xticks=e[1]:100:e[2]

    for g in groupby(sort(df, :group, rev=true), :group)
        scatterlines!(
            ax,
            g.pts,
            g.cond,
            label=g.group[1],
            markersize=25,
            linewidth=3,
            cycle=Cycle([
                    :marker,
                    :color,
                ]; covary=true),
        )
    end

    convergence_trendline!(ax, e, 6, -8, :poly)
    # convergence_trendline!(ax, e, 8, -32, :poly)
    convergence_trendline!(ax, e, 4, -0.1, :expo)
    convergence_trendline!(ax, e, 8, -0.2, :expo)
    convergence_trendline!(ax, e, 2.5, -1, :poly)

    l = axislegend(
        ax
    )
    ylims!(ax, (1, 1e+20))
    fig[1, 2] = l
    fig

end
