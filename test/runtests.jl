using SafeTestsets


@safetestset "Integral Operators" include("operators.jl")

@safetestset "Close Evaluation" include("close_evaluation.jl")
