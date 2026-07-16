
# Extension package for enzyme automatic differentiation
#
module BimDiffEnzymeExt

using BimDiff
using Enzyme

# NOTE: further derivatives are needed for faster optimization algs

"""
    BimDiff.spatial_gradient()

compute the gradient of the solution of a boundary value problem with respect to
physical space

"""
function BimDiff.spatial_gradient(
    mode::Enzyme.ForwardMode,
    problem::BoundaryValueProblem,
    target::AbstractMatrix,
    approach::Approach,
    correction::Union{SingularCorrection,HypersingularCorrection},
)

    # allocate input shadow memory
    d_problem = Enzyme.make_zero(problem)
    dx = Enzyme.make_zero(target)
    dy = Enzyme.make_zero(target)

    dx[:, 1] .= 1.;
    dy[:, 2] .= 1.;



    fwd1 = autodiff(
        ForwardWithPrimal,
        Const(solve_and_evaluate),
        Duplicated(problem, d_problem),
        Const(approach),
        Const(correction),
        Duplicated(target, dx)
    )

    fwd2 = autodiff(
        ForwardWithPrimal,
        Const(solve_and_evaluate),
        Duplicated(problem, d_problem),
        Const(approach),
        Const(correction),
        Duplicated(target, dy)
    )



    return [collect(fwd1[1][1]);; collect(fwd2[1][1])]


end

function BimDiff.spatial_gradient(
    mode::Enzyme.ReverseMode,
    problem::BoundaryValueProblem,
    target::AbstractMatrix,
    approach::Approach,
    correction::Union{SingularCorrection,HypersingularCorrection},
)


    d_problem = Enzyme.make_zero(problem)
    d_target = Enzyme.make_zero(target)



    forward, reverse = autodiff_thunk(
        ReverseSplitWithPrimal,
        Const{typeof(solve_and_evaluate)},
        Duplicated,
        Duplicated{typeof(problem)},
        Const{typeof(approach)},
        Const{typeof(correction)},
        Duplicated{typeof(target)},
    )


    tape, result, shadow_result = forward(
        Const(solve_and_evaluate),
        Duplicated(problem, d_problem),
        Const(approach),
        Const(correction),
        Duplicated(target, d_target),
    )

    shadow_result[1].=1.

    rev = reverse(
        Const(solve_and_evaluate),
        Duplicated(problem, d_problem),
        Const(approach),
        Const(correction),
        Duplicated(target, d_target),
        tape,
    )

    return d_target

end

end


