#definition of algorithms

# BIE approaches
struct Direct <: Approach end
struct Indirect <: Approach end

struct DirectBIEAlgorithm{
    L<:LinearSolve.SciMLLinearSolveAlgorithm,
    D<:AbstractDiscretization,
    C<:AbstractSingularCorrection,
} <: AbstractBIEAlgorithm{Direct}
    linsolver::L
    discretization::D
    correction::C
    # TODO: provide default constructor always
end
function DirectBIEAlgorithm(
    ;
    correction::AbstractSingularCorrection,
    linsolver=RFLUFactorization(),
    discretization=Nystroem(),
)
    return DirectBIEAlgorithm(linsolver, discretization, correction)
end
struct IndirectBIEAlgorithm{
    L<:LinearSolve.SciMLLinearSolveAlgorithm,
    D<:AbstractDiscretization,
} <: AbstractBIEAlgorithm{Indirect}
    linsolver::L
    discretization::D
end
function IndirectBIEAlgorithm(;
    linsolver=RFLUFactorization(),
    discretization=Nystroem()
)
    return IndirectBIEAlgorithm(linsolver, discretization)
end

linsolver(a::AbstractBIEAlgorithm) = a.linsolver
discretization(a::AbstractBIEAlgorithm) = a.discretization
correction(a::AbstractBIEAlgorithm{Direct}) = a.correction
correction(::AbstractBIEAlgorithm{Indirect}) = nothing

### BVP solution evaluation algorithms
#
# BVP evaluation method

struct CauchyIntegral <: EvaluationMethod end
struct PotentialTheory <: EvaluationMethod end
struct DistancePolicy{T<:Real} <: EvaluationMethod
    cutoff::T
end

cutoff(::CauchyIntegral) = 1.0
cutoff(::PotentialTheory) = 0.0
cutoff(d::DistancePolicy) = d.cutoff

struct DirectBVPAlgorithm{
    D<:AbstractDiscretization,
} <: AbstractBVPAlgorithm{Direct} # private
    discretization::D
    evalmethod::PotentialTheory
end
function DirectBVPAlgorithm(;
    discretization=Nystroem(),
    evalmethod=PotentialTheory(),
)
    return DirectBVPAlgorithm(discretization, evalmethod)
end

struct IndirectBVPAlgorithm{
    D<:AbstractDiscretization,
    M<:EvaluationMethod,
} <: AbstractBVPAlgorithm{Indirect} # private
    discretization::D
    evalmethod::M
end
function IndirectBVPAlgorithm(;
    discretization=Nystroem(),
    evalmethod=CauchyIntegral(),
)
    return IndirectBVPAlgorithm(discretization, evalmethod)
end

# Boundary Data Problems

struct DirectBDPAlgorithm{
    D<:AbstractDiscretization
} <: AbstractBDPAlgorithm{Direct}
    discretization::D
end
function DirectBDPAlgorithm(;
    discretization=Nystroem(),
)
    return DirectBDPAlgorithm(discretization)
end
struct IndirectBDPAlgorithm{
    D<:AbstractDiscretization,
    C<:AbstractSingularCorrection
} <: AbstractBDPAlgorithm{Indirect}
    discretization::D
    correction::C
end
function IndirectBDPAlgorithm(;
    discretization=Nystroem(),
    correction,
)
    return IndirectBDPAlgorithm(discretization, correction)
end
# traits
approach(::AbstractAlgorithm{A}) where {A} = A
correction(::AbstractAlgorithm) = nothing
evalmethod(::AbstractAlgorithm) = Type{PotentialTheory}
evalmethod(a::AbstractBVPAlgorithm{Indirect}) = a.evalmethod
correction(a::AbstractBDPAlgorithm{Indirect}) = a.correction


# remaining: check that the correction specified for BIE and BDP match the respective
# boundary condition when calling solve(prob, alg)
