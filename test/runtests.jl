using ImmersedBodies
using Test

@testset "solve function" begin
    include("solving.jl")
end
@testset "quantities" begin
    include("quantities.jl")
end
