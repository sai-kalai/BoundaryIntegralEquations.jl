using LinearAlgebra

using BoundaryIntegralEquations
using BoundaryIntegralEquations: matrix

for op in ops
    @printf("%d\t%s\t\t%.3e\t%s\n",
        op|>BoundaryIntegralEquations.matrix|>m->size(m, 2),
        op|>correction |> c -> try
            return c.order
        catch e
            return nothing
        end,
        op|>o -> begin
            m=BoundaryIntegralEquations.matrix(o);
            if (o isa DoubleLayer || o isa AdjointDoubleLayer)
                m = m-0.5I
            end
            m
        end |> cond,
        op|>typeof|>nameof)

end
