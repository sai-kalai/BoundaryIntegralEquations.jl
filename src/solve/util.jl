
is_garbage(x::Number) = isnan(x) || isinf(x) || abs(x) > 1e10
has_garbage(a::Array) = any(is_garbage, a)
has_garbage(o::IntegralOperator) = has_garbage(matrix(o))
const CHECK_GARBAGE = true # TODO: turn off optionally
function panic_if_garbage(ops...)
    if CHECK_GARBAGE
        foreach(ops) do o
            if has_garbage(o)
                @error "Operator contains garbage"
                display(matrix(o))
                error("Operator contains garbage. Either call constructor with `populate_matrix=true` or call `populate_matrices!(...)`")
            end
        end
    else
        return
    end
end
