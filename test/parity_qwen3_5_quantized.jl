using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Generation: load_chat_template, generate
using HuggingFaceTransformers.Models:
    Qwen3_5Config, Qwen3_5ForCausalLM, NVFP4Matrix, FP8Matrix, BF16Matrix, config_from_json,
    read_config, load_weights

# A compressed-tensors NVFP4/FP8 checkpoint is ~22 GB, so the reference is its first
# four layers (three delta-net, one attention), recorded by
# test/fixtures/record_qwen3_5_quantized_parity.py with the weights decoded exactly
# and float32 activations. Gate:
#   HFT_TEST_PARITY_QWEN3_5_QUANTIZED=1
const _QWEN3_5_QUANTIZED_FIXTURE =
    joinpath(@__DIR__, "fixtures", "qwen3_5_27b_nvfp4_4layer_parity.json")

function _first_layers(cfg::Qwen3_5Config, n::Int)
    fields = Dict(f => getfield(cfg, f) for f in fieldnames(Qwen3_5Config))
    fields[:num_hidden_layers] = n
    fields[:layer_types] = cfg.layer_types[1:n]
    return Qwen3_5Config(; fields...)
end

if !isfile(_QWEN3_5_QUANTIZED_FIXTURE)
    @info "Skipping quantized Qwen3.5 parity: run " *
        "`python3 test/fixtures/record_qwen3_5_quantized_parity.py`"
else
    fx = JSON3.read(read(_QWEN3_5_QUANTIZED_FIXTURE, String))
    dir = snapshot_download(String(fx.repo_id); verbose=false)
    tol = Float32(fx.tolerance)
    ids = Int.(collect(fx.input_token_ids))

    @testset verbose = true "$(fx.repo_id), first $(fx.num_hidden_layers) layers" begin
        tokenizer = load_tokenizer(dir)
        @test encode(tokenizer, String(fx.prompt)) == ids

        cfg = _first_layers(config_from_json(Qwen3_5Config, read_config(dir)), Int(fx.num_hidden_layers))
        lm = Qwen3_5ForCausalLM(cfg, load_weights(dir))
        # Packed as stored: NVFP4 MLPs, FP8 projections and head, bf16 embeddings.
        @test lm.model.layers[1].mlp.gate_proj.weight isa NVFP4Matrix
        @test lm.model.layers[1].self_attn.in_proj_qkv.weight isa FP8Matrix
        @test lm.model.layers[4].self_attn.wq.weight isa FP8Matrix
        @test lm.lm_head.weight isa FP8Matrix
        @test lm.model.embed_tokens.weight isa BF16Matrix

        logits = lm(reshape(ids, :, 1))
        @testset "position $(r.position)" for r in fx.positions
            col = logits[:, r.position + 1, 1]
            expected = Float32.(collect(r.top_logits))
            @test maximum(abs.([col[i + 1] for i in r.top_indices] .- expected)) < tol
            if expected[1] - expected[2] > 2tol
                @test argmax(col) - 1 == first(r.top_indices)
            end
        end

        @testset "chat prompt and greedy decode match" begin
            g = fx.generation
            prompt = encode(tokenizer, load_chat_template(dir)(g.messages))
            @test prompt == collect(g.prompt_ids)
            new = generate(lm, prompt; max_new_tokens=length(g.tokens))[(length(prompt) + 1):end]
            gaps = [t[1] - t[2] for t in g.top2]
            comparable = something(findfirst(<(2tol), gaps), length(gaps) + 1) - 1
            @info "quantized Qwen3.5 greedy decode: $(comparable) of $(length(gaps)) steps comparable"
            @test new[1:comparable] == collect(g.tokens)[1:comparable]
        end
    end
end
