using Test
using JSON3
using Allspark.HFHub: snapshot_download
using Allspark.Tokenizers: load_tokenizer, encode
using Allspark.Models: load_weights,
                       LlamaForCausalLM,
                       LlamaConfig,
                       LlamaRopeScaling,
                       load_state_dict!

const FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "llama_3_2_1b_parity.json")

isfile(FIXTURE_PATH) || error("""
    Parity fixture missing: $(FIXTURE_PATH)

    Generate it once with the Python recorder:

        python3 test/fixtures/record_llama_parity.py

    The recorder needs `transformers` + `torch` installed and the
    Llama-3.2-1B-Instruct model accessible (HF login or HF_TOKEN).
""")

# Inline config loader. Duplicated with examples/repl_chat.jl on purpose
# — Phase 2 will generalize this into the Models module once a third
# consumer appears.
function _load_llama_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    rope_scaling = nothing
    if haskey(raw, :rope_scaling) && raw.rope_scaling !== nothing
        rs = raw.rope_scaling
        rope_scaling = LlamaRopeScaling(;
            factor=Float64(rs.factor),
            low_freq_factor=Float64(rs.low_freq_factor),
            high_freq_factor=Float64(rs.high_freq_factor),
            original_max_position_embeddings=Int(rs.original_max_position_embeddings),
        )
    end

    head_dim = if haskey(raw, :head_dim)
        Int(raw.head_dim)
    else
        Int(raw.hidden_size) ÷ Int(raw.num_attention_heads)
    end

    return LlamaConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        num_key_value_heads=Int(get(raw, :num_key_value_heads, raw.num_attention_heads)),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        rope_theta=Float64(get(raw, :rope_theta, 500000.0)),
        rms_norm_eps=Float64(get(raw, :rms_norm_eps, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, false)),
        rope_scaling=rope_scaling,
    )
end

fixture = JSON3.read(read(FIXTURE_PATH, String))
repo_id = String(fixture.repo_id)
prompt = String(fixture.prompt)
expected_ids = Int[Int(x) for x in fixture.input_token_ids]
expected_argmax = Int(fixture.argmax_token_id)
expected_top_indices = Int[Int(x) for x in fixture.top_indices]
expected_top_logits = Float32[Float32(x) for x in fixture.top_logits]
tolerance = Float32(get(fixture, :tolerance, 1.0e-3))

snapshot_dir = snapshot_download(repo_id; verbose=false)

@testset "tokenizer matches HF byte-for-byte (no BOS)" begin
    tk = load_tokenizer(snapshot_dir)
    ids = encode(tk, prompt)
    @test ids == expected_ids
end

@testset "last-position logits match HF fp32 within $(tolerance)" begin
    cfg = _load_llama_config(snapshot_dir)
    lm = LlamaForCausalLM(cfg)
    load_state_dict!(lm, load_weights(snapshot_dir))

    input_mat = reshape(expected_ids, :, 1)
    logits = lm(input_mat; step=1)
    last_logits = collect(logits[:, end, 1])

    # Argmax token must match.
    @test argmax(last_logits) - 1 == expected_argmax

    # Top-K logits at HF's recorded positions must match within tolerance.
    # HF token IDs are 0-indexed; Julia arrays are 1-indexed.
    max_err = Float32(0)
    for (i, hf_idx) in enumerate(expected_top_indices)
        v = Float32(last_logits[hf_idx + 1])
        err = abs(v - expected_top_logits[i])
        max_err = max(max_err, err)
    end
    @test max_err < tolerance

    # Top-K rank order: Allspark's top-K (by index) should overlap heavily
    # with HF's top-K. Exact rank order may differ near the boundary due
    # to fp accumulation differences, so we check set membership of the
    # top-10 only.
    allspark_top10 = sortperm(last_logits; rev=true)[1:10] .- 1
    hf_top10 = expected_top_indices[1:10]
    overlap = length(intersect(Set(allspark_top10), Set(hf_top10)))
    @test overlap >= 9   # one swap tolerated for the lowest-magnitude logit
end
