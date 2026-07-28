using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Models:
    load_weights, GPT2ForCausalLM, GPT2Config, load_state_dict!

# OpenAI's original GPT-2 weights, redistributed on HF. Sizes span from
# 124M to 1.5B; all share the same architecture and state-dict layout.
const VARIANTS = (
    ("124M", "gpt2_124m_parity.json"),
    ("355M", "gpt2_355m_parity.json"),
    ("774M", "gpt2_774m_parity.json"),
    ("1558M", "gpt2_1558m_parity.json"),
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

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_GPT2", ""))

function _load_gpt2_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    hidden = Int(raw.n_embd)
    return GPT2Config(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=hidden,
        intermediate_size=Int(
            get(raw, :n_inner, nothing) === nothing ? 4 * hidden : raw.n_inner
        ),
        num_hidden_layers=Int(raw.n_layer),
        num_attention_heads=Int(raw.n_head),
        max_position_embeddings=Int(raw.n_positions),
        layer_norm_eps=Float64(get(raw, :layer_norm_epsilon, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, true)),
    )
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping GPT-2-$(name) parity: fixture $(fixture_path) not " *
            "present. Generate it with " *
            "`python3 test/fixtures/record_gpt2_parity.py $(name)`."
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

    @testset verbose = true "GPT-2-$(name) ($(repo_id))" begin
        @testset "tokenizer matches HF byte-for-byte (no BOS)" begin
            tk = load_tokenizer(snapshot_dir)
            ids = encode(tk, prompt)
            @test ids == expected_ids
        end

        @testset "last-position logits within $(tolerance)" begin
            cfg = _load_gpt2_config(snapshot_dir)
            lm = GPT2ForCausalLM(cfg)
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
    "Unknown HFT_TEST_PARITY_GPT2 variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
