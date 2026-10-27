
using Test

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

@testset "solve" begin
    n = 200
    t = (0:(n-1)) * 2pi / n
    c = DiscreteClosedCurve(t, starfish)
    x_test = Fixtures.test_locations()

    # bc = Dirichlet(sin.(t))
    for (bc, corr) in [
        (Dirichlet(sin.(t)), Zeta(32)),
        (Neumann(sin.(t)), KapurRokhlin(32))
    ]

        @testset "bc $(typeof(bc)), correction $corr" begin

            bvpb = BoundaryValueProblem(Laplace(), bc, Interior(), c)

            @testset "Direct approach" begin

                biepb = BIEProblem{Direct}(bvpb)
                biealg = DirectBIEAlgorithm(correction=corr)
                biesol = solve(biepb, biealg) # NOTE: correction as a kwarg here instead of a member of algo would provide more flexibility and adaptability to bc type

                bvpalg = DirectBVPAlgorithm()
                bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)

                bdpb = BDProblem{Direct}(bvpb)
                bdpalg = DirectBDPAlgorithm()
                bdpsol = solve(bdpb, bdpalg; bie_sol=biesol)

                @testset "Internal BIE computation" begin
                    @test bvpsol.u ≈ solve(bvpb, bvpalg, biealg; target=x_test).u
                    @test bdpsol.u ≈ solve(bdpb, bdpalg, biealg).u
                end
            end

            @testset "Indirect approach" begin

                biepb = BIEProblem{Indirect}(bvpb)
                biealg = IndirectBIEAlgorithm()
                biesol = solve(biepb, biealg)


                if bc isa Dirichlet
                    bvpalg = IndirectBVPAlgorithm()
                    bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                    @test bvpsol.alg.evalmethod isa CauchyIntegral
                end

                @testset "PotentialTheory" begin
                    bvpalg = IndirectBVPAlgorithm(evalmethod=PotentialTheory())
                    bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                end
                if bc isa Dirichlet
                    @testset "CauchyIntegral" begin
                        bvpalg = IndirectBVPAlgorithm(evalmethod=CauchyIntegral())
                        bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                    end
                    @testset "DistancePolicy" begin
                        bvpalg = IndirectBVPAlgorithm(evalmethod=DistancePolicy(0.05))
                        bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                    end
                end

                bdpb = BDProblem{Indirect}(bvpb)
                bdpalg = IndirectBDPAlgorithm(correction=corr)
                bdpsol = solve(bdpb, bdpalg; bie_sol=biesol)

                @testset "Internal BIE computation" begin
                    @test bvpsol.u ≈ solve(bvpb, bvpalg, biealg; target=x_test).u
                    @test bdpsol.u ≈ solve(bdpb, bdpalg, biealg).u
                end
            end

        end

    end
end




