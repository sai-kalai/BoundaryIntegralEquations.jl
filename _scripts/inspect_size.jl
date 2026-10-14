@eval Base begin
    function summarysize(obj;
        exclude=Union{DataType,Core.TypeName,Core.MethodInstance},
        chargeall=Union{Core.TypeMapEntry,Method})

        @nospecialize obj exclude chargeall
        println("hello from base")

        ss = SummarySize(IdDict(), Any[], Int[], exclude, chargeall)

        size::Int = ss(obj)

        #if val !== nothing
        #    if isstructtype(typeof(x)) && !isa(x, SimpleVector) && !isa(x, GenericMemory)
        #        f_name = fieldname(typeof(x), i)
        #        println("Field `$(f_name)` of $(typeof(x)) -> Value type: $(typeof(val))")
        #    elseif isa(x, GenericMemory) || isa(x, SimpleVector)
        #        println("Index [$i] of $(typeof(x)) -> Value type: $(typeof(val))")
        #    end
        #end

        #println("$(summary(obj)): size contribution: $(size)")


        while !isempty(ss.frontier_x)
            # DFS heap traversal of everything without a specialization
            # BFS heap traversal of anything with a specialization
            x = ss.frontier_x[end]
            i = ss.frontier_i[end]
            val = nothing
            if isa(x, SimpleVector)
                nf = length(x)
                if isassigned(x, i)
                    val = x[i]
                end
            elseif isa(x, GenericMemory)
                T = eltype(x)
                if Base.allocatedinline(T)
                    np = datatype_npointers(T)
                    nf = length(x) * np
                    idx = (i-1) ÷ np + 1
                    if @inbounds @inline isassigned(x, idx)
                        elt = x[idx]
                        p = (i-1) % np + 1
                        if nth_pointer_isdefined(elt, p)
                            val = get_nth_pointer(elt, p)
                        end
                    end
                else
                    nf = length(x)
                    if @inbounds @inline isassigned(x, i)
                        val = x[i]
                    end
                end
            else
                nf = datatype_npointers(typeof(x))
                if nth_pointer_isdefined(x, i)
                    val = get_nth_pointer(x, i)
                end
            end
            if nf > i
                ss.frontier_i[end] = i + 1
            else
                pop!(ss.frontier_x)
                pop!(ss.frontier_i)
            end
            if val !== nothing && !isa(val, Module) && (!isa(val, ss.exclude) || isa(x, ss.chargeall))

                inc = ss(val)::Int
                size += inc


                if (inc > 1e+4)

                    println(summary(val))
                    println("Memory contribution: $(inc/1e+9)")

                    if isstructtype(typeof(x)) && !isa(x, SimpleVector) && !isa(x, GenericMemory) && !isa(x, DataType)
                        #f_name = fieldname(typeof(x), i)
                        println("Field of $(typeof(x)) -> Value type: $(typeof(val))")
                    elseif isa(x, GenericMemory) || isa(x, SimpleVector)
                        println("Index [$i] of $(typeof(x)) -> Value type: $(typeof(val))")
                    else
                        println("fallback $(summary(x))")
                    end
                end



            end
        end
        return size
    end
end
