#definition of algorithms


struct Direct <: Approach end
struct Indirect <: Approach end


struct CauchyIntegral <: EvaluationMethod end
@doc raw"""
    PotentialTheory

Use layer potentials via Nystrom approximation

"""
struct PotentialTheory <: EvaluationMethod end
@doc raw"""
    Adaptive

Choose method depending on the distance from boundry to target point

TODO: this mixes policy and algorithm...

"""
struct DistancePolicy{T<:Real} <: EvaluationMethod
    cutoff::T
end

cutoff(::CauchyIntegral) = 1.0
cutoff(::PotentialTheory) = 0.0
cutoff(d::DistancePolicy) = d.cutoff



# TODO: try to avoid specific names like DirectBIEAlgo
function BIEAlgorithm{Direct}(
    correction::C,
    ls=RFLUFactorization(),
) where {C<:AbstractSingularCorrection}
    DirectBIEAlgorithm(correction, ls)
end
correction(a::BIEAlgorithm{Direct}) = a.correction
function BIEAlgorithm{Indirect}(ls=RFLUFactorization())
    IndirectBIEAlgorithm(ls)
end
struct DirectBIEAlgorithm{ # private
    C<:AbstractSingularCorrection,
    L<:LinearSolve.SciMLLinearSolveAlgorithm
} <: BIEAlgorithm{Direct}
    correction::C
    linearsolve::L
end
struct IndirectBIEAlgorithm{ # private
    L<:LinearSolve.SciMLLinearSolveAlgorithm
} <: BIEAlgorithm{Indirect}
    linearsolve::L
end

function BVPAlgorithm{Direct}()
    DirectBVPAlgorithm()
end
function BVPAlgorithm{Indirect}(m::EvaluationMethod)
    IndirectBVPAlgorithm(m)
end
struct DirectBVPAlgorithm <: BVPAlgorithm{Direct} # private
end
struct IndirectBVPAlgorithm{M<:EvaluationMethod} <: BVPAlgorithm{Indirect} # private
    evalmethod::M
end

# Boundary Data Problem Algorithms
function BDPAlgorithm{Direct}()
    DirectBDPAlgorithm()
end
function BDPAlgorithm{Indirect}(correction::C) where {C<:AbstractSingularCorrection}
    IndirectBDPAlgorithm(correction)
end
struct DirectBDPAlgorithm <: BDPAlgorithm{Direct} end
struct IndirectBDPAlgorithm{C<:AbstractSingularCorrection} <: BDPAlgorithm{Indirect}
    correction::C
end
# traits
approach(::AbstractAlgorithm{A}) where {A} = A
correction(::AbstractAlgorithm) = nothing
evalmethod(::AbstractAlgorithm) = Type{PotentialTheory}
evalmethod(a::BVPAlgorithm{Indirect}) = a.evalmethod
correction(a::BDPAlgorithm{Indirect}) = a.correction


# remaining: check that the correction specified for BIE and BDP match the respective
# boundary condition when calling solve(prob, alg)
