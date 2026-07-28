using Test
using JSON3
using SafeTensors: serialize
using HuggingFaceTransformers.Models: load_weights

@testset verbose = true "load_weights" begin
    @testset "single-file checkpoint" begin
        mktempdir() do dir
            weights = Dict(
                "model.embed_tokens.weight" =>
                    collect(reshape(Float32(1):Float32(12), 4, 3)),
                "model.layers.0.input_layernorm.weight" => Float32[1, 2, 3, 4],
            )
            file = joinpath(dir, "model.safetensors")
            open(io -> serialize(io, weights), file, "w")

            for entry in (file, dir)
                loaded = load_weights(entry)
                @test loaded isa AbstractDict{String}
                @test valtype(loaded) <: AbstractArray
                @test sort(collect(keys(loaded))) == sort(collect(keys(weights)))
                for (k, v) in weights
                    @test loaded[k] == v
                end
            end
        end
    end

    @testset "sharded checkpoint" begin
        mktempdir() do dir
            shard1 = Dict("model.embed_tokens.weight" => Float32[1 2 3; 4 5 6])
            shard2 = Dict(
                "model.layers.0.self_attn.q_proj.weight" => Float32[7 8; 9 10],
                "lm_head.weight" => Float32[11, 12, 13],
            )
            shard1_name = "model-00001-of-00002.safetensors"
            shard2_name = "model-00002-of-00002.safetensors"
            open(io -> serialize(io, shard1), joinpath(dir, shard1_name), "w")
            open(io -> serialize(io, shard2), joinpath(dir, shard2_name), "w")

            weight_map = Dict{String,String}()
            for k in keys(shard1)
                weight_map[k] = shard1_name
            end
            for k in keys(shard2)
                weight_map[k] = shard2_name
            end
            index = Dict("metadata" => Dict("total_size" => 0), "weight_map" => weight_map)
            index_path = joinpath(dir, "model.safetensors.index.json")
            open(io -> JSON3.write(io, index), index_path, "w")

            for entry in (dir, index_path)
                loaded = load_weights(entry)
                merged = merge(shard1, shard2)
                @test sort(collect(keys(loaded))) == sort(collect(keys(merged)))
                for (k, v) in merged
                    @test loaded[k] == v
                end
            end
        end
    end

    @testset "errors" begin
        @test_throws ArgumentError load_weights("/nonexistent/path.safetensors")
        mktempdir() do dir
            @test_throws ArgumentError load_weights(dir)
            stray = joinpath(dir, "weights.bin")
            write(stray, UInt8[0])
            @test_throws ArgumentError load_weights(stray)
        end
    end
end
