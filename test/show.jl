module TestShow

using ImmersedBodies

macro testshow(expr)
    return quote
        show(devnull, "text/plain", $(esc(expr)))
    end
end

@testshow flow = FreestreamFlow(t -> (t, t); Re=123)

@testshow grid = UniformGrid(0.01, (-1.0, 1.0), (-2.0, 2.0))

@testshow MultiLevelGrid(grid, 1)
@testshow grids = MultiLevelGrid(grid, 3)

@testshow scheme = CNAB(0.001)

@testshow fluid = PsiOmegaFluidGrid(flow, grids; scheme)

@testshow Curves.LineSegment((0.0, 1.0), (2.0, 3.0))
@testshow Curves.Circle()

@testshow GlobalFrame()
@testshow frame = OffsetFrame(GlobalFrame()) do t
    r = (0.0, 1.0)
    v = (0.0, 0.0)
    θ = 0.0
    Ω = 0.0
    return OffsetFrameInstant(r, v, θ, Ω)
end

curve = Curves.Circle()
segments = partition(curve, fluid)

@testshow body1 = RigidBody(segments)
@testshow body2 = RigidBody(segments, frame)

bcs = map(ClampParameterBC, [0.0, 1.0])
@testshow body3 = EulerBernoulliBeam(LinearModel, segments, bcs; m=1.0, kb=1.0)

@testshow bodies = BodyGroup([body1, body2, body3])

@testshow prob = Problem(fluid, bodies)

end
