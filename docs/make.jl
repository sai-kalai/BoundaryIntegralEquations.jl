using BoundaryIntegralMethods
using Documenter

DocMeta.setdocmeta!(BoundaryIntegralMethods, :DocTestSetup, :(using BoundaryIntegralMethods); recursive=true)

makedocs(;
    modules=[BoundaryIntegralMethods],
    authors="Simón Cadavid",
    sitename="BoundaryIntegralMethods.jl",
    format=Documenter.HTML(;
        canonical="https://sai-kalai.github.io/BoundaryIntegralMethods.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sai-kalai/BoundaryIntegralMethods.jl",
    devbranch="main",
)
