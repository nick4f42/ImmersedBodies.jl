using ImmersedBodies
using Test

@testset "ImmersedBodies.jl" verbose = true begin
    include("problem.jl")
    include("solver.jl")
    include("io.jl")
    include("curves.jl")
end
