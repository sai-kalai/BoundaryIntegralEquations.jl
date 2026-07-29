using Revise
using BoundaryIntegralEquations
using Documenter

DocMeta.setdocmeta!(BoundaryIntegralEquations, :DocTestSetup, :(using BoundaryIntegralEquations); recursive=true)

makedocs(;
    modules=[BoundaryIntegralEquations],
    authors="Simón Cadavid",
    sitename="BoundaryIntegralEquations.jl",
    format=Documenter.HTML(;
        canonical="https://sai-kalai.github.io/BoundaryIntegralEquations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sai-kalai/BoundaryIntegralEquations.jl",
    devbranch="main",
)
