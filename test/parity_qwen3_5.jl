using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Generation: load_chat_template, generate
using HuggingFaceTransformers.Models: Qwen3_5ForCausalLM, load_pretrained
using HuggingFaceTransformers.Layers: GatedDeltaNet, GQA

# Recorded by test/fixtures/record_qwen3_5_parity.py. Gate:
#   HFT_TEST_PARITY_QWEN3_5=1
const _QWEN3_5_FIXTURE = joinpath(@__DIR__, "fixtures", "qwen3_5_0_8b_parity.json")

if !isfile(_QWEN3_5_FIXTURE)
    @info "Skipping Qwen3.5 parity: run `python3 test/fixtures/record_qwen3_5_parity.py`"
else
    fx = JSON3.read(read(_QWEN3_5_FIXTURE, String))
    dir = snapshot_download(String(fx.repo_id); verbose=false)
    tol = Float32(fx.tolerance)
    ids = Int.(collect(fx.input_token_ids))

    @testset verbose = true "Qwen3.5 ($(fx.repo_id))" begin
        tokenizer = load_tokenizer(dir)
        @testset "tokenizer matches HF" begin
            @test encode(tokenizer, String(fx.prompt)) == ids
        end

        lm = load_pretrained(dir)
        @test lm isa Qwen3_5ForCausalLM
        @test lm.model.layers[1].self_attn isa GatedDeltaNet
        @test lm.model.layers[4].self_attn isa GQA

        # Past HF's 64-token chunks, so its chunked delta rule spans several.
        logits = lm(reshape(ids, :, 1))
        @testset "position $(r.position)" for r in fx.positions
            col = logits[:, r.position + 1, 1]
            expected = Float32.(collect(r.top_logits))
            @test maximum(abs.([col[i + 1] for i in r.top_indices] .- expected)) < tol
            if expected[1] - expected[2] > 2tol
                @test argmax(col) - 1 == first(r.top_indices)
            end
        end

        @testset "chat prompt and greedy decode match HF" begin
            g = fx.generation
            prompt = encode(tokenizer, load_chat_template(dir)(g.messages))
            @test prompt == collect(g.prompt_ids)
            new = generate(lm, prompt; max_new_tokens=length(g.tokens))[(length(prompt) + 1):end]
            # Greedy choices are comparable up to the first step where HF's top two
            # logits sit within the tolerance; past a near tie either pick is right.
            gaps = [t[1] - t[2] for t in g.top2]
            comparable = something(findfirst(<(2tol), gaps), length(gaps) + 1) - 1
            @info "Qwen3.5 greedy decode: $(comparable) of $(length(gaps)) steps comparable"
            @test new[1:comparable] == collect(g.tokens)[1:comparable]
        end
    end
end
