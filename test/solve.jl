
using Test

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools

@testset "solve" begin
    n = 200
    t = (0:(n-1)) * 2pi / n
    b = DiscreteClosedCurve(t, starfish)
    x_test = Fixtures.test_locations()

    laplace = Laplace()

    b_source, density_source, u_exact = manufactured_solution(laplace, x_test)

    S_source = SingleLayer(laplace, b_source, b.x)
    D_star_source = AdjointDoubleLayer(laplace, b_source, b.x)
    populate_matrices!(b_source, b.x, S_source, D_star_source; target_normals=b.n)
    σ_exact = S_source * density_source # Dirichlet BC
    τ_exact = D_star_source * density_source # Neumann BC exact solution

    # bc = Dirichlet(sin.(t))
    for (bc, corr, cau_exact) in [
        (Dirichlet(σ_exact), Zeta(32), τ_exact),
        (Neumann(τ_exact), KapurRokhlin(32), σ_exact)
    ]

        @testset "bc $(typeof(bc)), correction $corr" begin

            bvpb = BoundaryValueProblem(Laplace(), bc, Interior(), b)

            @testset "Direct approach" begin

                biepb = BIEProblem{Direct}(bvpb)
                biealg = DirectBIEAlgorithm(correction=corr)
                biesol = solve(biepb, biealg) # NOTE: correction as a kwarg here instead of a member of algo would provide more flexibility and adaptability to bc type

                bvpalg = DirectBVPAlgorithm()
                bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)

                @test bvpsol.u ≈ u_exact atol = 1e-5

                bdpb = BDProblem{Direct}(bvpb)
                bdpalg = DirectBDPAlgorithm()
                bdpsol = solve(bdpb, bdpalg; bie_sol=biesol)

                @test bdpsol.u ≈ cau_exact atol = 1e-5

                @testset "Internal BIE computation" begin
                    @test solve(bvpb, bvpalg, biealg; target=x_test).u ≈ u_exact atol = 1e-5
                    @test solve(bdpb, bdpalg, biealg).u ≈ cau_exact atol = 1e-5
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
                    @test bvpsol.u ≈ u_exact atol = 1e-5
                end

                @testset "PotentialTheory" begin
                    bvpalg = IndirectBVPAlgorithm(evalmethod=PotentialTheory())
                    bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                    @test bvpsol.u ≈ u_exact atol = 1e-5
                end
                if bc isa Dirichlet
                    @testset "CauchyIntegral" begin
                        bvpalg = IndirectBVPAlgorithm(evalmethod=CauchyIntegral())
                        bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                        @test bvpsol.u ≈ u_exact atol = 1e-5
                    end
                    @testset "DistancePolicy" begin
                        bvpalg = IndirectBVPAlgorithm(evalmethod=DistancePolicy(0.05))
                        bvpsol = solve(bvpb, bvpalg; bie_sol=biesol, target=x_test)
                        @test bvpsol.u ≈ u_exact atol = 1e-5
                    end
                end

                bdpb = BDProblem{Indirect}(bvpb)
                bdpalg = IndirectBDPAlgorithm(correction=corr)
                bdpsol = solve(bdpb, bdpalg; bie_sol=biesol)
                @test bdpsol.u ≈ cau_exact atol = 1e-5

                @testset "Internal BIE computation" begin
                    @test solve(bvpb, bvpalg, biealg; target=x_test).u ≈ u_exact atol = 1e-5
                    @test solve(bdpb, bdpalg, biealg).u ≈ cau_exact atol = 1e-5
                end
            end

        end

    end
end




