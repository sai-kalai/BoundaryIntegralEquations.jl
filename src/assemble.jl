
# common workflow for assembly of operator matrices
@doc raw"""
    populate_matrices!(source::DiscreteClosedCurve{<:Real}, ops::IntegralOperator..., # variadic)

Compute the coefficients of several requested self-interaction integral operators

# Arguments
- `source::DiscreteClosedCurve{<:Real}`: Manifold where the density is defined
- `ops::IntegralOperator... : List of operators that are required.
"""
function populate_matrices!(
    source::DiscreteClosedCurve{<:Real},
    ops::IntegralOperator..., # variadic
)
    populate_matrices!(source, ops)
end

# so client is responsible for allocating the zeros
# self interaction operators
function populate_matrices!(
    source::DiscreteClosedCurve{<:Real},
    ops, # tuple of operators with already allocated matrices
)
    n = size(source, 2)

    # client provides initialized matrices contained in ops

    # compute required stencils: {KR, FD2}

    stencil_cache = StencilCache{Int32,Vector{Float64}}(Dict(), Dict())

    # if both dlp and dlp adjoint are requested, compute once and transpose before applying weights

    # loop over i
    Threads.@threads for i in n:-1:1 # WARN: reverse iteration is back, from where?
        for j in 1:n
            # always every pair gets a fresh cache
            foreach(ops) do op
                # call appropriate code for each operator kind
                compute_entry!(op, i, j, source)
            end

        end



    end
    # note: move outside of hot loop to avoid data races
    Threads.@threads for i in n:-1:1 # WARN: reverse iteration is back, from where?
        foreach(ops) do op
            apply_correction!(op, stencil_cache, i, source)
        end
    end
end

@doc raw"""
    populate_matrices!(source::DiscreteClosedCurve{<:Real}, ops, # tuple of operators with already allocated matrices)

Compute the coefficients of several requested source-target interaction integral operators

# Arguments
- `source::DiscreteClosedCurve{<:Real}`: Manifold where the density is defined
- `ops::IntegralOperator... : List of operators that are required
- `target_normals`: Outward unitary normal vectors at the target locations, in case the AdjointDoubleLayer operator is requested
"""
function populate_matrices!(
    source::DiscreteClosedCurve{T},
    target::AbstractMatrix{T},
    ops::IntegralOperator...; # only
    target_normals::Union{AbstractMatrix{T},Nothing}=nothing, # the AdjointDoubleLayer operator needs the normal vectors at the target locations
) where {T<:AbstractFloat}
    populate_matrices!(source, target, ops, ; target_normals=target_normals)
end

# target interaction operators
function populate_matrices!(
    source::DiscreteClosedCurve{T},
    target::AbstractMatrix{T},
    ops::Tuple;
    target_normals::Union{AbstractMatrix{T},Nothing}=nothing,
) where {T<:AbstractFloat}


    if any(x -> x isa AdjointDoubleLayer, ops) && isnothing(target_normals)
        error("Requested AdjointDoubleLayer, but provided no unit outward normals at the target points")
    end


    n = size(source, 2)
    m = size(target, 2)

    # TODO: find out about SIMD and threading options in this version
    # investigate autodiff on kernels directly

    # slow loop over second index
    Threads.@threads for j in 1:n
        # fast loop over first index
        for i in 1:m

            # always every pair gets a fresh cache

            # call appropriate code for each operator kind
            foreach(ops) do op
                compute_entry!(op, i, j, source, target, target_normals)
            end
        end
    end
end
