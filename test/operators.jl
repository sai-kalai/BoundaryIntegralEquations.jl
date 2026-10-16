using BoundaryIntegralEquations
using Test


include("fixtures.jl")

@testset "Integral operators" begin

    n = 20
    ord = 32

    Γ = DiscreteClosedCurve(n, starfish)

    x_test = test_locations()

    laplace = Laplace()
    kapur_rokhlin = KapurRokhlin(ord)
    zeta = Zeta(ord)
    sidi = Sidi()

    allocator = (_m, _n) -> Matrix{Float64}(undef, _m, _n)

    @testset "Separate computation" begin
        # S = SingleLayer(laplace, Γ, kapur_rokhlin)
        # D = DoubleLayer(laplace, ) # ok
        # D_star = AdjointDoubleLayer(laplace, )  # ok
        # H_zeta = Hypersingular(laplace, zeta, ) # ok
        # H_sidi = Hypersingular(laplace, sidi, ) # ok
        # populate_matrices!(Γ, S, D, D_star, H_sidi, H_zeta)
        #
        # S_target = SingleLayer(laplace, nothing, ) # ok
        # D_target = DoubleLayer(laplace, ) # ok
        # populate_matrices!(Γ, x_test, S_target, D_target)

    end

    @testset "Simultaneous computation" begin


        S = SingleLayer(laplace, Γ, kapur_rokhlin,) # ok
        D = DoubleLayer(laplace, Γ) # ok
        D_star = AdjointDoubleLayer(laplace, Γ)  # ok
        H_zeta = Hypersingular(laplace, Γ, zeta,) # ok
        H_sidi = Hypersingular(laplace, Γ, sidi,) # ok
        populate_matrices!(Γ, S, D, D_star, H_sidi, H_zeta)

        @test S.matrix ≈ reference_operator_matrix(typeof(S)) atol=1e-4
        @test D.matrix ≈ reference_operator_matrix(typeof(D), Val(:self)) atol=1e-4
        @test D_star.matrix ≈ reference_operator_matrix(typeof(D_star)) atol=1e-4
        @test H_zeta.matrix ≈ reference_operator_matrix(typeof(H_zeta)) atol=1e-4
        @test H_sidi.matrix ≈ reference_operator_matrix(typeof(H_sidi)) atol=1e-4

        S_target = SingleLayer(laplace, Γ, x_test) # ok
        D_target = DoubleLayer(laplace, Γ, x_test) # ok
        populate_matrices!(Γ, x_test, S_target, D_target)

        @test S_target.matrix ≈ reference_operator_matrix(typeof(S_target)) atol=1e-4
        @test D_target.matrix ≈ reference_operator_matrix(typeof(D_target), Val(:target)) atol=1e-4


    end

end
