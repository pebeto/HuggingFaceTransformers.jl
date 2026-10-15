using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Models:
    GptOssConfig, GptOssForCausalLM, MXFP4Weights, config_from_json, read_config, load_weights

# The full model needs ~42 GB once HF decodes its experts, so the reference is a
# two-layer cut of the real checkpoint (one sliding, one full-attention layer),
# recorded by test/fixtures/record_gpt_oss_parity.py. Gate:
#   HFT_TEST_PARITY_GPT_OSS=1
const _GPT_OSS_FIXTURE = joinpath(@__DIR__, "fixtures", "gpt_oss_20b_2layer_parity.json")

function _first_layers(cfg::GptOssConfig, n::Int)
    fields = Dict(f => getfield(cfg, f) for f in fieldnames(GptOssConfig))
    fields[:num_hidden_layers] = n
    fields[:layer_types] = cfg.layer_types[1:n]
    return GptOssConfig(; fields...)
end

if !isfile(_GPT_OSS_FIXTURE)
    @info "Skipping gpt-oss parity: run `python3 test/fixtures/record_gpt_oss_parity.py`"
else
    fx = JSON3.read(read(_GPT_OSS_FIXTURE, String))
    dir = snapshot_download(String(fx.repo_id); verbose=false)
    tol = Float32(fx.tolerance)
    ids = Int.(collect(fx.input_token_ids))

    @testset verbose = true "gpt-oss-20b, first $(fx.num_hidden_layers) layers" begin
        @testset "tokenizer matches HF" begin
            @test encode(load_tokenizer(dir), String(fx.prompt)) == ids
        end

        cfg = _first_layers(config_from_json(GptOssConfig, read_config(dir)), Int(fx.num_hidden_layers))
        lm = GptOssForCausalLM(cfg, load_weights(dir))
        @test lm.model.layers[1].mlp.experts.gate_up isa MXFP4Weights
        @test lm.model.layers[1].self_attn.window_size == 128
        @test isnothing(lm.model.layers[2].self_attn.window_size)

        logits = lm(reshape(ids, :, 1))
        @testset "position $(r.position)" for r in fx.positions
            col = logits[:, r.position + 1, 1]
            expected = Float32.(collect(r.top_logits))
            got = [col[i + 1] for i in r.top_indices]
            @test maximum(abs.(got .- expected)) < tol
            # The argmax is only meaningful when the top two are further apart
            # than the tolerance; at the last position they are not.
            if expected[1] - expected[2] > 2tol
                @test argmax(col) - 1 == first(r.top_indices)
            end
        end
    end
end
