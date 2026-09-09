# Shared functionality for plotting consistently

using LinearAlgebra
using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

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
        # )
    )
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
    if k.approach_t <: Direct
        :dot
    elseif k.approach_t <: Indirect
        :dashdot
    else
        error("invalid approach type: $(k.approach_t)")
    end
end
function get_color(k::SolverParameters, res::ConvergenceResult)
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


@doc raw"""
    scatterline_common_kwargs(k::SolverParameters)

Defines shared visualization mappings for convergence and timing plots

# Arguments
- `k::SolverParameters`: Information about a solver run
"""
function scatterlines_common_kwargs(k::SolverParameters, res::ConvergenceResult)
    kwargs = (;
        markersize=10,
        strokewidth=1,
        linewidth=2,
        marker=get_marker(k),
        linestyle=get_linestyle(k),
        color=get_color(k, res),
        colormap=get_colormap(res, k.correction),
        colorrange=get_colorrange(res, k.correction),
        markercolor=get_markercolor(k, res),
        markercolorrange=get_markercolorrange(res, k.evalmethod),
        markercolormap=get_markercolormap(res, k.evalmethod),
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

function plot_errors(
    res::ConvergenceResult,
    ;
)
    # Plot
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel="n",
        ylabel="L∞-error",
        yscale=log10,
        xscale=log10,
        xticks=LinearTicks(5),
    )
    ylims!(ax, (1e-17, 1e+1))

    linestyles = []
    markers = []

    for (key::SolverParameters, group::SolutionGroup) in res.solutions


        sort!(group, by=swm -> numpoints(swm[1]))
        sols = solutions(group)
        ns = [numpoints(s) for s in sols]

        # extract error according to solution type
        # errs = if key.solution_t <: BVPSolution
        #     [norm(s.u - res.u_exact, Inf) for s in sols]
        # elseif key.solution_t <: BDPSolution
        #     if key.bdrycond_t <: Dirichlet
        #         [norm(s.u - res.neumann_exact[numpoints(s)], Inf) for s in sols]
        #     elseif key.bdrycond_t <: Neumann
        #         [norm(s.u - res.dirichlet_exact[numpoints(s)], Inf) for s in sols]
        #     else
        #         error("invalid bc type $(key.bdrycond_t)")
        #     end
        # else
        #     error("invalid solution type $(key.solution_t)")
        # end

        errs = errors(key, res, group)

        @show key, last(errs)

        if any(isnan, errs)
            @warn "NaN found in errors"
            @show key
            # @show errs
            # @show res.u_exact
            for s in sols
                if any(isnan, s.u)
                    @show typeof(bvp(s))
                    @show numpoints(s), extrema(s.u)
                end
            end
        end

        kwargs = scatterlines_common_kwargs(key, res)

        scatterlines!(
            ax,
            ns,
            errs,
            ;
            kwargs...,
        )
    end

    c1, c2, c3 = scatterlines_common_colorbars!(fig, res)
    legend = scatterlines_common_legend!(fig, res)
    fig[0, 1] = legend
    fig[1, 2][1, 1] = c1
    fig[1, 2][1, 2] = c2
    fig[1, 2][1, 3] = c3


    # # trendlines
    conv_style = (; linestyle=:dashdotdot, linewidth=3)
    α = 0.1
    lines!(ax,
        res.n_vals, # use last iteration for getting ns
        exp.(-α .* res.n_vals),
        ;
        color=:grey,
        conv_style...
    )


    fig, ax
end
