using Test
using Random
using JSON3
using BFloat16s: BFloat16
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Models:
    Float8E4M3, FP8Matrix, NVFP4Matrix, BF16Matrix, Qwen3_5Config, Qwen3_5ForCausalLM,
    qwen3_5_state_dict_map, load_pretrained, _resolve, _save_safetensors
using HuggingFaceTransformers.Layers: Linear

const _NVFP4 = Float32[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
# bfloat16 by truncating the low 16 bits. Broadcasting `BFloat16.(x)` hangs Julia
# 1.12's CPU codegen; building from bits never compiles a conversion.
_bf16(x::AbstractArray{Float32}) = collect(reinterpret(BFloat16, map(v -> UInt16(reinterpret(UInt32, v) >> 16), x)))
_e4m3(b::UInt8) = Float32(reinterpret(Float8E4M3, b))

@testset "Float8E4M3 decodes torch's float8_e4m3fn" begin
    @test _e4m3(0x7e) == 448                       # largest finite
    @test _e4m3(0x38) == 1 && _e4m3(0xb8) == -1
    @test _e4m3(0x08) == 2.0f0^-6                  # smallest normal
    @test _e4m3(0x01) == 2.0f0^-9                  # smallest subnormal
    @test isnan(_e4m3(0x7f)) && isnan(_e4m3(0xff))
    @test _e4m3(0x80) === -0.0f0
end

# Random packed weights with their dense float32 equivalent, `(out, in)`.
function _random_fp8(rng, n_out, n_in)
    bytes = rand(rng, setdiff(0x00:0xff, (0x7f, 0xff)), n_in, n_out)
    data = reinterpret(Float8E4M3, bytes)
    scale = rand(rng, Float32, n_out) .+ 0.5f0
    dense = [_e4m3(bytes[k, r]) * scale[r] for r in 1:n_out, k in 1:n_in]
    return FP8Matrix(data, scale), dense
end

function _random_nvfp4(rng, n_out, n_in)
    codes = rand(rng, 0:15, n_in, n_out)
    packed = UInt8.(codes[1:2:end, :] .| (codes[2:2:end, :] .<< 4))
    scale_bytes = rand(rng, 0x30:0x48, n_in ÷ 16, n_out)
    global_scale = 37.5f0
    dense = [_NVFP4[codes[k, r] + 1] * (_e4m3(scale_bytes[(k - 1) ÷ 16 + 1, r]) / global_scale)
             for r in 1:n_out, k in 1:n_in]
    return NVFP4Matrix(packed, reinterpret(Float8E4M3, scale_bytes), global_scale), dense
end

@testset verbose = true "packed weight matrices" begin
    rng = Xoshiro(21)
    for (name, (w, dense)) in (("FP8", _random_fp8(rng, 9, 48)), ("NVFP4", _random_nvfp4(rng, 9, 48)))
        @testset "$(name)" begin
            @test size(w) == (9, 48)
            @test w == dense                                  # element by element
            # One token, whole groups of four, and groups cut short.
            @test all(n -> (x = randn(rng, Float32, 48, n); w * x ≈ dense * x), (1, 2, 3, 4, 5, 9))
            @test w * randn(Xoshiro(1), Float32, 48) ≈ dense * randn(Xoshiro(1), Float32, 48)
            @test_throws DimensionMismatch w * randn(Float32, 47, 1)
            @test Linear(w, nothing)(ones(Float32, 48, 2, 1)) ≈ reshape(dense * ones(Float32, 48, 2), 9, 2, 1)
        end
    end
    b = _bf16(randn(rng, Float32, 4, 6))
    @test BF16Matrix(b) == Float32.(b)
    @test_throws DimensionMismatch FP8Matrix(reinterpret(Float8E4M3, zeros(UInt8, 4, 3)), ones(Float32, 2))
    @test_throws DimensionMismatch NVFP4Matrix(zeros(UInt8, 8, 3), reinterpret(Float8E4M3, zeros(UInt8, 2, 3)), 1)
end

@testset "a quantized Qwen3.5 checkpoint loads packed and matches its dense weights" begin
    rng = Xoshiro(22)
    cfg = Qwen3_5Config(;
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=64, partial_rotary_factor=0.5,
    )
    dense = Qwen3_5ForCausalLM(cfg)
    tensors = Dict{String,AbstractArray}()
    for (key, (path, transform)) in qwen3_5_state_dict_map(cfg)
        a = _resolve(dense, path)
        tensors[key] = transform === :transpose ? permutedims(a, (2, 1)) : copy(a)
    end
    # Store every MLP as NVFP4 and every other projection as FP8, as the 27B
    # checkpoint does, writing the matching dense values back into `dense`.
    for (i, layer) in enumerate(dense.model.layers), (field, proj) in pairs(Dict(
            :gate_proj => "mlp.gate_proj", :up_proj => "mlp.up_proj", :down_proj => "mlp.down_proj"))
        key = "model.language_model.layers.$(i - 1).$(proj)"
        lin = getfield(layer.mlp, field)
        w, values = _random_nvfp4(rng, size(lin.weight)...)
        lin.weight .= values
        delete!(tensors, key * ".weight")
        tensors[key * ".weight_packed"] = permutedims(w.packed, (2, 1))
        tensors[key * ".weight_scale"] = permutedims(w.scales, (2, 1))
        tensors[key * ".weight_global_scale"] = [w.global_scale]
    end
    for (i, layer) in enumerate(dense.model.layers)
        mixer = layer.self_attn
        projections = hasproperty(mixer, :wq) ?
            ((:wq, "self_attn.q_proj"), (:wk, "self_attn.k_proj"), (:wv, "self_attn.v_proj"), (:wo, "self_attn.o_proj")) :
            ((:in_proj_qkv, "linear_attn.in_proj_qkv"), (:in_proj_z, "linear_attn.in_proj_z"), (:out_proj, "linear_attn.out_proj"))
        for (field, proj) in projections
            key = "model.language_model.layers.$(i - 1).$(proj)"
            lin = getfield(mixer, field)
            w, values = _random_fp8(rng, size(lin.weight)...)
            lin.weight .= values ./ 4
            tensors[key * ".weight"] = permutedims(w.data, (2, 1))
            tensors[key * ".weight_scale"] = reshape(w.scale ./ 4, :, 1)
        end
    end
    tensors["model.language_model.embed_tokens.weight"] = _bf16(tensors["model.language_model.embed_tokens.weight"])
    dense.model.embed_tokens.weight .= Float32.(_bf16(dense.model.embed_tokens.weight))

    mktempdir() do dir
        _save_safetensors(joinpath(dir, "model.safetensors"), tensors)
        write(joinpath(dir, "config.json"), JSON3.write(Dict(
            "architectures" => ["Qwen3_5ForConditionalGeneration"],
            "text_config" => Dict(
                "vocab_size" => 64, "hidden_size" => 32, "intermediate_size" => 48,
                "num_hidden_layers" => 4, "num_attention_heads" => 4, "num_key_value_heads" => 2,
                "head_dim" => 8, "linear_num_key_heads" => 2, "linear_num_value_heads" => 4,
                "linear_key_head_dim" => 8, "linear_value_head_dim" => 8,
                "max_position_embeddings" => 64, "tie_word_embeddings" => false,
                "rope_parameters" => Dict("rope_type" => "default", "rope_theta" => 1.0e7,
                    "partial_rotary_factor" => 0.5),
            ),
        )))
        packed = load_pretrained(dir)
        @test packed.model.layers[1].mlp.gate_proj.weight isa NVFP4Matrix
        @test packed.model.layers[1].self_attn.in_proj_qkv.weight isa FP8Matrix
        @test packed.model.embed_tokens.weight isa BF16Matrix
        ids = reshape(rand(rng, 0:63, 7), :, 1)
        @test packed(ids) ≈ dense(ids) rtol = 1.0e-4
    end
end
