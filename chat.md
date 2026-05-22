Yes, you're on the right track. Using the type system to encode the variations is exactly the Julia way, and it will make your life much easier for plotting, error analysis, filtering, and consistent styling.
Recommended Design
juliaabstract type Solution end

# === Core parameters as type parameters ===
abstract type AbstractSide end
struct Interior <: AbstractSide end
struct Exterior <: AbstractSide end

abstract type AbstractBC{C} end  # C = correction type for Dirichlet
struct Dirichlet{C} <: AbstractBC{C} end
struct Neumann <: AbstractBC{Nothing} end

abstract type AbstractApproach end
struct Direct <: AbstractApproach end
struct Indirect <: AbstractApproach end

abstract type AbstractProblem end
struct Laplace <: AbstractProblem end
# struct Helmholtz{k} <: AbstractProblem end
# struct Stokes <: AbstractProblem end
Then the solution types become:
juliastruct DirichletSolution{P<:AbstractProblem, S<:AbstractSide, A<:AbstractApproach, C} <: Solution
    n::Int
    u::Vector{Float64}
    τ::Vector{Float64}          # Neumann trace
    bc::Dirichlet{C}
    problem::P
    side::S
    approach::A
    # You can also store the operators if you want, or just the metadata
end

struct NeumannSolution{P<:AbstractProblem, S<:AbstractSide, A<:AbstractApproach} <: Solution
    n::Int
    u::Vector{Float64}
    σ::Vector{Float64}          # Dirichlet trace
    bc::Neumann
    problem::P
    side::S
    approach::A
end

struct ExactSolution{P<:AbstractProblem, S<:AbstractSide} <: Solution
    n::Int
    u::Vector{Float64}
    σ::Vector{Float64}
    τ::Vector{Float64}
    problem::P
    side::S
end
Why this is better

The type itself now carries all the semantic information (Laplace, Interior, Direct, Zeta etc.).
No more u_sidi_external_dirichlet nightmare.
Dispatch, filtering, and styling become trivial and type-safe.

Helper functions for analysis/plotting
julia# Extract metadata
problem_type(s::Solution) = s.problem
side(s::Solution) = s.side
approach(s::Solution) = s.approach
bc_type(s::DirichletSolution) = s.bc
bc_type(s::NeumannSolution) = s.bc

is_dirichlet(s::Solution) = s isa DirichletSolution
is_neumann(s::Solution)   = s isa NeumannSolution
is_interior(s::Solution)  = s.side isa Interior
is_exterior(s::Solution)  = s.side isa Exterior

# For consistent plotting styles
function get_style(s::Solution)
    line = is_neumann(s) ? :dash : :solid
    color = is_exterior(s) ? :red : :blue
    marker = s.approach isa Direct ? :circle : :square
    label = label_for(s)
    return (; linestyle=line, color=color, marker=marker, label=label)
end

function label_for(s::Solution)
    p = problem_type(s)
    side_str = is_interior(s) ? "int" : "ext"
    app_str = s.approach isa Direct ? "dir" : "ind"

    if s isa DirichletSolution
        corr = correction_name(s.bc)
        return "$(p) $(side_str) D-$(corr) $(app_str)"
    else
        return "$(p) $(side_str) N $(app_str)"
    end
end

correction_name(::Dirichlet{C}) where {C} = nameof(C)
correction_name(::Neumann) = "none"
Usage in your loop
juliasolutions = Solution[]

for (i, n) in enumerate(n_vals)
    Γ = DiscreteClosedCurve(n, ρ)

    # ... compute σ, τ_exact ...

    exact = ExactSolution{Laplace,Interior}(n, u_exact, σ, τ_exact, Laplace(), Interior())
    push!(solutions, exact)

    # Dirichlet - Zeta
    u, τ = solve(...)
    push!(solutions, DirichletSolution{Laplace,Interior,Direct,Zeta}(
        n, u, τ, Dirichlet{Zeta}(), Laplace(), Interior(), Direct()
    ))

    # Dirichlet - Sidi
    # ...

    # Neumann direct
    u, σ = solve(...)
    push!(solutions, NeumannSolution{Laplace,Interior,Direct}(n, u, σ, Neumann(), Laplace(), Interior(), Direct()))
end
Filtering & grouping
juliadirichlet_interior = filter(is_dirichlet, filter(is_interior, solutions))
neumann = filter(is_neumann, solutions)

# Group by n for convergence plots
grouped = Dict{Int, Vector{Solution}}()
for sol in solutions
    push!(get!(Vector{Solution}, grouped, sol.n), sol)
end
Or use SplitApplyCombine, DataFrames, or just good old findall + type assertions.
Alternative: Solution with a "tag" type
If you prefer even less boilerplate, you can do:
juliastruct SolutionTag{P,S,A,B}
    problem::P
    side::S
    approach::A
    bc::B
end

struct TaggedSolution{T<:SolutionTag} <: Solution
    n::Int
    u::Vector{Float64}
    # ... traces
    tag::T
end
But the first version with concrete types per combination is usually nicer for dispatch and clarity.

Recommendation: Go with the first design (type parameters on DirichletSolution / NeumannSolution). It gives you maximum power with Julia's type system while staying readable.
You can later add Helmholtz{k} etc. very cleanly.
Would you like me to show you a small complete example with error table generation + plotting recipe (using Makie or Plots)?go on to plotting recipeHere's a clean, reusable plotting recipe that leverages the type-encoded solutions:
1. First, add these helper functions
juliausing Plots

# ==================== Metadata extractors ====================
problem_name(s::Solution) = nameof(typeof(s.problem))
side_name(s::Solution)    = s.side isa Interior ? "Interior" : "Exterior"
approach_name(s::Solution)= s.approach isa Direct ? "Direct" : "Indirect"

function bc_name(s::DirichletSolution)
    C = typeof(s.bc).parameters[1]  # the correction type
    return C === Nothing ? "Dirichlet" : "Dirichlet-$(nameof(C))"
end
bc_name(s::NeumannSolution) = "Neumann"

function full_label(s::Solution)
    p = problem_name(s)
    side = side_name(s)[1:3]  # Int / Ext
    bc = bc_name(s)
    app = approach_name(s)[1:3]  # Dir / Ind
    "$(p) $(side) $(bc) $(app)"
end

# ==================== Style rules ====================
function get_plot_style(s::Solution)
    linestyle = s isa NeumannSolution ? :dash : :solid
    color = s.side isa Exterior ? :red : :blue
    marker = s.approach isa Direct ? :circle : :utriangle
    alpha = 0.9

    return (
        label = full_label(s),
        linestyle = linestyle,
        color = color,
        marker = marker,
        alpha = alpha,
        linewidth = 2.0,
        markersize = 4
    )
end
2. Main plotting functions
juliafunction plot_convergence(solutions::Vector{<:Solution};
                          quantity=:u, norm=Inf,
                          title="Convergence", yscale=:log10)

    plt = plot(; xlabel="n (boundary points)", ylabel="Error (∞-norm)",
               title=title, yscale=yscale, legend=:outerright, size=(800,600))

    # Group by configuration
    groups = Dict{String, Vector{Tuple{Int,Float64}}}()

    for sol in solutions
        if quantity == :u
            err = hasproperty(sol, :u) ? norm(sol.u .- reference_u(sol), norm) : nothing
        elseif quantity == :τ || quantity == :neumann
            err = hasproperty(sol, :τ) ? norm(sol.τ .- reference_τ(sol), norm) : nothing
        elseif quantity == :σ
            err = hasproperty(sol, :σ) ? norm(sol.σ .- reference_σ(sol), norm) : nothing
        else
            error("Unknown quantity")
        end

        isnothing(err) && continue

        key = full_label(sol)
        push!(get!(()->[], groups, key), (sol.n, err))
    end

    for (label, data) in sort(collect(groups))
        data = sort(data)  # sort by n
        ns = [p[1] for p in data]
        errs = [p[2] for p in data]

        # Find a representative solution to get style
        rep = findfirst(s -> full_label(s) == label, solutions)
        style = get_plot_style(solutions[rep])

        plot!(plt, ns, errs; style..., label=label)
    end

    return plt
end

# Helper to get reference solution for a given n
function reference_u(sol::Solution)
    # You can store one ExactSolution and look it up by n, or pass a dict
    exact = first(filter(s -> s isa ExactSolution && s.n == sol.n, solutions_global))  # or better design
    return exact.u
end
3. Better reference handling (Recommended)
Add this to your main() or as a global:
juliaconst reference_solutions = Dict{Int, ExactSolution}()

# Inside the loop, after creating exact:
reference_solutions[n] = exact
Then improve the reference functions:
juliareference_u(sol::Solution) = reference_solutions[sol.n].u
reference_τ(sol::Solution) = reference_solutions[sol.n].τ
reference_σ(sol::Solution) = reference_solutions[sol.n].σ
4. Usage examples
julia# After filling `solutions`

p1 = plot_convergence(solutions, quantity=:u, title="Dirichlet & Neumann - u error")
p2 = plot_convergence(filter(is_dirichlet, solutions), quantity=:τ, title="Hypersingular correction comparison")

plot(p1, p2, layout=(2,1), size=(900,700))
You can also do specific comparisons easily:
julia# Only interior direct methods
interior_direct = filter(s -> is_interior(s) && s.approach isa Direct, solutions)

# Only Dirichlet with different corrections
dirichlet = filter(is_dirichlet, solutions)
Bonus: Automatic multi-panel
juliafunction plot_all_convergence(solutions)
    p_u  = plot_convergence(solutions, quantity=:u,  title="Potential u")
    p_τ  = plot_convergence(filter(is_dirichlet, solutions), quantity=:τ, title="Neumann trace τ")
    p_σ  = plot_convergence(filter(is_neumann, solutions),   quantity=:σ, title="Dirichlet trace σ")

    plot(p_u, p_τ, p_σ, layout=(1,3), size=(1200,400), margin=5Plots.mm)
end

Would you like me to also show:

A DataFrame version (very convenient for filtering + StatsPlots)?
How to add slope/reference lines (O(n⁻ᵏ))?
A more advanced version using Makie with interactive legend filtering?

Just say the word. This structure should already eliminate the variable name hell you had before.
