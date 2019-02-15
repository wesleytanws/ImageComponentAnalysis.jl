
@testset "iterative_recursion_original_3" begin

    for T in (Int, Bool, Gray{Bool}, Gray{N0f8}, Gray{N0f16}, Gray{N0f32}, Gray{Float64})
        for i = 1:20
            test_volume = T.(eval(Symbol("test_volume_$(i)")))

            for n = 3:2:max(size(test_volume)...)
                # Call iterative recursion original 3 algorithm and compare the number of components
                labels = ImageComponentAnalysis.label_components(IterativeRecursionOriginal3(), test_volume, (n, n, n))
                num_components = maximum(labels)

                @test num_components == test_volume_results[i]
            end
        end
    end

end
