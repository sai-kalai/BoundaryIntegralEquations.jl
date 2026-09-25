# Shared functionality for plotting consistently

using Makie
using LinearAlgebra

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools
using BoundaryIntegralEquations: order


using GLMakie
using CairoMakie

const MARKER_LABELS = [
    :diamond,
    'x',
    '+',
    :cross,
    :dtriangle,
    :utriangle,
    :xcross,
    :rect,
    :circle,
    :hexagon,
    :rtriangle,
    :pentagon,
    :star4,
    :star5,
    :star6,
    :star8,
    :ltriangle,
]


mytheme = Theme(
    fontsize=25,
    # Label=(fontsize=25,),
    Axis=(
        xlabelsize=30,
        ylabelsize=35,
        xticklabelsize=20,
        yticklabelsize=20,
        palette=(;
            color=Makie.to_colormap(:tab10),
            marker=MARKER_LABELS,
            linestyle=[
                :dash,
                :dot,
                :dashdot,
                :dashdotdot,
            ]
        ),
        # width=150,
        # height=150,
    ),
    Colorbar=(ticklabelsize=15,),
    # Legend=(labelsize=10),
    annotation=(fontsize=20),
)
mytheme = merge(mytheme, theme_latexfonts())
set_theme!(mytheme)

function save_and_display!(name, fig)
    # save using Cairo
    plotfile = joinpath("figures", name)
    CairoMakie.activate!()
    CairoMakie.save(plotfile * ".pdf", fig)
    CairoMakie.save(plotfile * ".png", fig)

    @info "saved `fig` to $(plotfile)"

    # display using GL
    GLMakie.activate!()
    screen = display(GLMakie.Screen(), fig)

    if !isinteractive()
        screen |> wait
    end
end


const _N = "N"

conv_style = (;
    alpha=0.6,
    # color=:grey,
    linewidth=3,
)
lstyles = [
    :dash,
    :dot,
    :dashdot,
    :dashdotdot,
]
function convergence_trendline!(ax, xrange, yoffset, order, type)
    dense_n = extrema(xrange) |> x -> range(x..., length=50) |> collect

    if type == :poly
        # polynomial trend
        lines!(ax,
            dense_n,
            10. ^ (yoffset) .* (dense_n ./ dense_n[1]) .^ -order,
            ;
            label=L"\mathcal{O}(%$_N^{%$(-order)})",
            cycle=Cycle([:linestyle,]; covary=true),
            color=:black,
            conv_style...
        )

    elseif type == :expo
        # exponential trend
        lines!(ax,
            dense_n,
            10. ^ (yoffset) .* exp.(-order .* (dense_n .- dense_n[1])),
            # exp.(-order .* (dense_n)),
            ;
            label=L"\mathcal{O} (\exp(%$(-order)%$(_N)))",
            cycle=Cycle([:linestyle,]; covary=true),
            color=:grey,
            conv_style...
        )
    elseif type == :nlogn
        lines!(ax,
            dense_n,
            10. ^ (yoffset) .* dense_n .* log2.(dense_n .- dense_n[1] .+ 1.),
            # exp.(-order .* (dense_n)),
            ;
            label=L"\mathcal{O} (%$_N \log(%$(_N)))",
            cycle=Cycle([:linestyle,]; covary=true),
            color=:grey,
            conv_style...
        )
    else

        error("convergence type $type not supported")
    end


end





# categorical colormaps for correction orders
function get_colormap(res::ConvergenceResult, ::Type{<:Zeta})
    # Reverse(
    cgrad(
        :reds,
        length(res.fd_acc_vals), categorical=true
    )
    # )
end
function get_colormap(res::ConvergenceResult, ::Type{<:KapurRokhlin})
    # Reverse(
    cgrad(
        :blues,
        length(res.kr_acc_vals), categorical=true
    )
    # )
end
get_colormap(::ConvergenceResult, ::Type{<:Sidi}) = :tab10 # dummy
get_colormap(res::ConvergenceResult, c::AbstractSingularCorrection) = get_colormap(res, typeof(c))

get_colorrange(res::ConvergenceResult, ::Type{<:Zeta}) = (0.5, length(res.fd_acc_vals) + 0.5)
get_colorrange(res::ConvergenceResult, ::Type{<:KapurRokhlin}) = (0.5, length(res.kr_acc_vals) + 0.5)
get_colorrange(::ConvergenceResult, ::Type{<:Sidi}) = (0, 1) # dummy
get_colorrange(res::ConvergenceResult, c::AbstractSingularCorrection) = get_colorrange(res, typeof(c))

get_markercolorrange(res::ConvergenceResult, ::Type{<:EvaluationMethod}) = (0.5, length(res.cutoff_vals) + 0.5)
get_markercolorrange(res::ConvergenceResult, m::EvaluationMethod) = get_markercolorrange(res, typeof(m))
function get_markercolormap(res::ConvergenceResult, ::Type{<:EvaluationMethod})
    # Reverse(
    cgrad(
        :greens,
        length(res.cutoff_vals), categorical=true
    )
    # )
end
get_markercolormap(res::ConvergenceResult, m::EvaluationMethod) = get_markercolormap(res, typeof(m))
function get_marker(k::SolverParameters)
    if k.solution_t <: BVPSolution
        if k.bdrycond_t <: Dirichlet
            :rect
        elseif k.bdrycond_t <: Neumann
            :cross
        else
            error("invalid bc type: $(k.bdrycond_t)")
        end
    elseif k.solution_t <: BDPSolution
        if k.bdrycond_t <: Dirichlet
            :diamond
        elseif k.bdrycond_t <: Neumann
            :xcross
        else
            error("invalid bc type: $(k.bdrycond_t)")
        end
    else
        error("invalid solution type: $(k.solution_t)")
    end
end
function get_linestyle(k::SolverParameters)
    # if k.approach_t <: Direct
    #     :dot
    # elseif k.approach_t <: Indirect
    #     :dashdot
    # else
    #     error("invalid approach type: $(k.approach_t)")
    # end
    return :solid
end
function get_color(k::SolverParameters, res::ConvergenceResult)::Union{Int,Symbol}

    if k.correction isa Zeta
        findfirst(==(k.correction.order), res.fd_acc_vals)
    elseif k.correction isa KapurRokhlin
        findfirst(==(k.correction.order), res.kr_acc_vals)
    elseif k.correction isa Sidi
        :purple
    else
        error("invalid correction: $(k.correction)")
    end
end
function get_markercolor(k::SolverParameters, res::ConvergenceResult)
    findfirst(==(cutoff(k.evalmethod)), res.cutoff_vals)
end

get_palette(nummarkers) = (;
    color=Makie.to_colormap(:tab10),
    marker=MARKER_LABELS[1:nummarkers],
    linestyle=[
        :dash,
        :dot,
        :dashdot,
        :dashdotdot,
    ],
)

@doc raw"""
    scatterline_common_kwargs(k::SolverParameters)

Defines shared visualization mappings for convergence and timing plots

# Arguments
- `k::SolverParameters`: Information about a solver run
"""
function scatterlines_common_kwargs(k::SolverParameters, res::ConvergenceResult)
    kwargs = (;
        markersize=20,
        strokewidth=1,
        linewidth=2,
        joinstyle=:miter,
        # alpha=0.7,
        # marker=get_marker(k),
        linestyle=get_linestyle(k),
        strokecolor=:black,
        # cycle=[[:color, :linecolor, :markercolor]=>:color]
        # strokecolor=:black,
        # color=get_color(k, res),
        # colormap=cgrad(:tab10, length(res.solutions)),
        # colorrange=(1, length(result.solutions)),
        # colorrange=get_colorrange(res, k.correction),
        # markercolor=get_markercolor(k, res),
        # strokecolor=get_colormap(res, k.correction)[color isa Int ? color : 0],
        # markercolorrange=get_markercolorrange(res, k.evalmethod),
        # markercolormap=get_markercolormap(res, k.evalmethod),
    )

    # distinguish lines that overlap
    # TODO: move this outside
    # if k.approach_t <: Indirect && k.bdrycond_t <: Dirichlet && k.solution_t <: BVPSolution
    #     kwargs = merge(kwargs, (; alpha=0.7)) # make both transparent
    #     if k.correction isa Sidi
    #         kwargs = merge(kwargs, (; linewidth=7)) # make one of them thicker
    #     end
    # end

    return kwargs
end


function scatterlines_common_legend!(fig, res::ConvergenceResult)
    # collect all uniques
    linestyles = Symbol[]
    linestyle_labels = String[]
    markers = Symbol[]
    marker_labels = String[]
    colors=Symbol[]
    color_labels=String[]

    for k in keys(res.solutions)
        ls = get_linestyle(k)
        mk = get_marker(k)
        cl = get_color(k, res)

        if (cl isa Symbol) && !(cl in colors)

            push!(colors, cl)
            push!(color_labels, "$(typeof(k.correction).name.name) Correction")

        end

        if !(ls in linestyles)
            push!(linestyles, ls)
            push!(linestyle_labels, "$(k.approach_t.name.name) Approach")
        end

        if !(mk in markers)
            s = "$(k.bdrycond_t.name.name) BC / $(begin
                if k.solution_t <: BDPSolution
                    "Cauchy data"
                elseif k.solution_t <: BVPSolution
                    "Solution"
                else
                    "Unknown"
                end
            end)"
            push!(markers, mk)
            push!(marker_labels, s)
        end
    end

    @show linestyles
    @show linestyle_labels
    @show markers
    @show marker_labels
    @show colors
    @show color_labels

    legend = Legend(
        fig,
        [
            [LineElement(linestyle=l) for l in linestyles],
            [MarkerElement(color=:black, marker=m) for m in markers],
            [PolyElement(color=c, strokecolor=:transparent) for c in colors],
        ],
        [linestyle_labels, marker_labels, color_labels],
        ["Linestyle", "Marker", "Color"],
        tellwidth=true,
        tellheight=true,
        # halign=:right,
        # valign=:top,
        # margin=(10, 10, 10, 10),
        orientation=:horizontal,
        # nbanks=2,
    )

    return legend
end

function scatterlines_common_colorbars!(fig, res::ConvergenceResult)
    c1 = Colorbar(
        fig,
        colormap=get_colormap(res, KapurRokhlin),
        limits=get_colorrange(res, KapurRokhlin),
        ticks=(1:length(res.kr_acc_vals), string.(res.kr_acc_vals)),
        label="KR Order (Line Color)"
    )

    c2 = Colorbar(
        fig,
        colormap=get_colormap(res, Zeta),
        limits=get_colorrange(res, Zeta),
        ticks=(1:length(res.fd_acc_vals), string.(res.fd_acc_vals)),
        label="FD Order (Line Color)"
    )
    c3 = Colorbar(
        fig,
        colormap=get_markercolormap(res, DistancePolicy),
        limits=get_markercolorrange(res, DistancePolicy),
        ticks=(1:length(res.cutoff_vals), string.(res.cutoff_vals)),
        label="Cutoff Value (Marker Color)"
    )
    return c1, c2, c3
end

