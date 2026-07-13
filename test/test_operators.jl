using Test
using BimDiff

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

    @testset "Operator computation" begin

        S = SingleLayer(laplace, Γ, kapur_rokhlin; matrix_factory=allocator) # ok
        D = DoubleLayer(laplace, Γ; matrix_factory=allocator) # ok
        D_star = AdjointDoubleLayer(laplace, Γ; matrix_factory=allocator)  # ok
        H_zeta = Hypersingular(laplace, Γ, zeta; matrix_factory=allocator) # ok
        H_sidi = Hypersingular(laplace, Γ, sidi; matrix_factory=allocator) # ok

        @test S.matrix ≈ reference_operator_matrix(typeof(S)) atol=1e-4
        @test D.matrix ≈ reference_operator_matrix(typeof(D), Val(:self)) atol=1e-4
        @test D_star.matrix ≈ reference_operator_matrix(typeof(D_star)) atol=1e-4
        @test H_zeta.matrix ≈ reference_operator_matrix(typeof(H_zeta)) atol=1e-4
        @test H_sidi.matrix ≈ reference_operator_matrix(typeof(H_sidi)) atol=1e-4

        S_target = SingleLayer(laplace, Γ, x_test; matrix_factory=allocator) # ok
        D_target = DoubleLayer(laplace, Γ, x_test; matrix_factory=allocator) # ok

        @test S_target.matrix ≈ reference_operator_matrix(typeof(S_target)) atol=1e-4
        @test D_target.matrix ≈ reference_operator_matrix(typeof(D_target), Val(:target)) atol=1e-4


    end

end
