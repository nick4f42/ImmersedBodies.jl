module TestBodies

using ImmersedBodies
using ImmersedBodies.Bodies
using Test

@testset "bodies" begin
    @testset "deforming bodies" begin
        points = [
            0.0 0.0
            1.0 0.0
            4.0 0.0
            4.0 1.0
            4.0 6.0
        ]

        lengths = [1.0, 2.0, 2.0, 3.0, 5.0]
        # parameters => [0.0, 0.17, 0.33, 0.58, 1.0]

        segments = Segments(points, lengths)

        bc_expect = [
            ClampBC(BodyPointIndex(1)) => 1,
            ClampBC(BodyPointIndex(2)) => 2,
            ClampBC(BodyPointParam(0.3)) => 3,
            ClampBC(BodyPointParam(0.6)) => 4,
            ClampBC(BodyPointParam(0.9)) => 5,
        ]

        @test_throws "between 0 and 1" BodyPointParam(1.1)

        bcs = map(first, bc_expect)
        body = EulerBernoulliBeam(LinearModel, segments, bcs; m=1.0, kb=1.0)

        for (bc, (_, i_expect)) in zip(body.bcs, bc_expect)
            @test bc_point(bc).i == i_expect
        end
    end
end

end # module
