using BoundaryIntegralEquations
using Test


@testset "close evaluation" begin

    @testset "holomorphism_boundary_limit" begin
        # problem = interior_dirichlet_problem(circle(128))
        # n = size(problem.boundary, 2)
        #
        # @testset "constant density" begin
        #     φ = ones(n)
        #     density = BoundaryDensity(φ)
        #
        #     v = holomorphism_boundary_limit(problem, density)
        #
        #     @test length(v) == n
        #     @test eltype(v) == ComplexF64
        #
        #     # φ′ = 0 and all pairwise differences vanish
        #     @test v ≈ ComplexF64.(-ones(n))
        # end
        #
        # @testset "zero density" begin
        #     φ = zeros(n)
        #     density = BoundaryDensity(φ)
        #
        #     v = holomorphism_boundary_limit(problem, density)
        #
        #     @test v ≈ zeros(ComplexF64, n)
        # end
        #
        # @testset "output size and type" begin
        #     φ = randn(n)
        #     density = BoundaryDensity(φ)
        #
        #     v = holomorphism_boundary_limit(problem, density)
        #
        #     @test size(v) == (n,)
        #     @test eltype(v) == ComplexF64
        #     @test all(isfinite, real.(v))
        #     @test all(isfinite, imag.(v))
        # end
    end

    @testset "cauchy_integral" begin

        boundary = DiscreteClosedCurve(200, starfish)

        n = size(boundary, 2)

        @testset "known solution" begin

            x_test = randn(ComplexF64, 10) * 0.3

            function f(z::T) where {T<:Complex{<:Real}}
                return z^2/(z^2+2z+2)
            end

            fig, ax = visualize(boundary)

            scatter!(ax, real.(x_test), imag.(x_test))

            wait(display(fig))

            boundary_data = f.(vec(reinterpret(ComplexF64, boundary.x)))

            inside_evaluations = f.(x_test)

            ci = cauchy_integral(boundary, reinterpret(Float64, x_test'), boundary_data)

            display(inside_evaluations - ci)

            @test ci ≈ inside_evaluations atol=1e-4

            fig2, ax2 = lines(imag.(ci))
            lines!(ax2, imag.(inside_evaluations))
            wait(display(fig2))

        end

    end
end

