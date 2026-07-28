using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Models:
    load_weights, MixtralForCausalLM, MixtralConfig, load_state_dict!

# Mixtral variants. Both are server-class in fp32: 8x7B ≈ 90 GB on disk,
# 8x22B ≈ 280 GB. The parity infrastructure works the same regardless of
# size; it just may not be runnable on your laptop.
const VARIANTS = (
    ("8x7b", "mixtral_8x7b_instruct_v01_parity.json"),
    ("8x22b", "mixtral_8x22b_instruct_v01_parity.json"),
)

const FIXTURES_DIR = joinpath(@__DIR__, "fixtures")

function _selected_variants(raw::AbstractString)
    s = strip(raw)
    isempty(s) && return String[]
    s == "all" && return String[v[1] for v in VARIANTS]
    parts = String[]
    for token in split(s, ',')
        t = strip(token)
        isempty(t) || push!(parts, String(t))
    end
    return parts
end

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_MIXTRAL", ""))

function _load_mixtral_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    head_dim = if haskey(raw, :head_dim)
        Int(raw.head_dim)
    else
        Int(raw.hidden_size) ÷ Int(raw.num_attention_heads)
    end

    sliding_window = if haskey(raw, :sliding_window) && raw.sliding_window !== nothing
        Int(raw.sliding_window)
    else
        nothing
    end

    return MixtralConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        num_key_value_heads=Int(get(raw, :num_key_value_heads, raw.num_attention_heads)),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        num_local_experts=Int(raw.num_local_experts),
        num_experts_per_tok=Int(raw.num_experts_per_tok),
        rope_theta=Float64(get(raw, :rope_theta, 1_000_000.0)),
        rms_norm_eps=Float64(get(raw, :rms_norm_eps, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, false)),
        sliding_window=sliding_window,
    )
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping Mixtral-$(name) parity: fixture $(fixture_path) not " *
            "present. Generate it with " *
            "`python3 test/fixtures/record_mixtral_parity.py $(name)` on a " *
            "machine with enough RAM."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    prompt = String(fixture.prompt)
    expected_ids = Int[Int(x) for x in fixture.input_token_ids]
    expected_argmax = Int(fixture.argmax_token_id)
    expected_top_indices = Int[Int(x) for x in fixture.top_indices]
    expected_top_logits = Float32[Float32(x) for x in fixture.top_logits]
    tolerance = Float32(get(fixture, :tolerance, 1.0e-3))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "Mixtral-$(name) ($(repo_id))" begin
        @testset "tokenizer matches HF byte-for-byte (no BOS)" begin
            tk = load_tokenizer(snapshot_dir)
            ids = encode(tk, prompt)
            @test ids == expected_ids
        end

        @testset "last-position logits within $(tolerance)" begin
            cfg = _load_mixtral_config(snapshot_dir)
            lm = MixtralForCausalLM(cfg)
            load_state_dict!(lm, load_weights(snapshot_dir))

            input_mat = reshape(expected_ids, :, 1)
            logits = lm(input_mat; step=1)
            last_logits = collect(logits[:, end, 1])

            @test argmax(last_logits) - 1 == expected_argmax

            max_err = Float32(0)
            for (i, hf_idx) in enumerate(expected_top_indices)
                v = Float32(last_logits[hf_idx + 1])
                err = abs(v - expected_top_logits[i])
                max_err = max(max_err, err)
            end
            @test max_err < tolerance

            jl_top10 = sortperm(last_logits; rev=true)[1:10] .- 1
            hf_top10 = expected_top_indices[1:10]
            overlap = length(intersect(Set(jl_top10), Set(hf_top10)))
            @test overlap >= 9
        end
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown HFT_TEST_PARITY_MIXTRAL variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
