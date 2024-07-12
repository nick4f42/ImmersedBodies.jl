include("tests.jl")

arrays = [Array]
CUDA.functional() && push!(arrays, CUDA.CuArray)
AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)

@info "Functional backends: $arrays"

@testset "ImmersedBodies.jl" verbose = true begin
    @testset "utils" test_utils()
    @testset "problems" test_problems()
    @testset "operators" begin
        @testset "$δ" for δ in (DeltaYang3S(),)
            test_delta_func(δ)
        end
    end
    @testset "operators $(nameof(array))" for array in arrays
        @testset "@loop" test_loop(array)
        @testset "FFT R2R" test_fft_r2r(array)
        @testset "$(replace(string(test), r"^test_" => ""))" for test in [
            test_nonlinear
            test_rot
            test_curl
            test_laplacian_inv
            test_multidomain_coarsen
            test_multidomain_interpolate
            test_regularization
        ]
            @testset "$(nd)D" for nd in (2, 3)
                test(array, Val(nd))
            end
        end
    end
end
