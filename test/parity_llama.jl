using Test
using JSON3
using Allspark.HFHub: snapshot_download
using Allspark.Tokenizers: load_tokenizer, encode
using Allspark.Models:
    load_weights, LlamaForCausalLM, LlamaConfig, LlamaRopeScaling, load_state_dict!

# Variant label → fixture filename. Each fixture is recorded once via
# test/fixtures/record_llama_parity.py against an authoritative HF run.
# Variants are gated individually so a laptop can run 1B / 3B and a
# server can opt into 8B / 70B.
const VARIANTS = (
    ("1B", "llama_3_2_1b_parity.json"),
    ("3B", "llama_3_2_3b_parity.json"),
    ("8B", "llama_3_1_8b_parity.json"),
    ("70B", "llama_3_1_70b_parity.json"),
)

const FIXTURES_DIR = joinpath(@__DIR__, "fixtures")

# Parse the gate. Accepts a comma-separated list ("1B,3B"), the literal
# "all" (every variant with a fixture present), or the legacy "1" alias
# (kept so older CI configs still run the 1B test).
function _selected_variants(raw::AbstractString)
    s = strip(raw)
    isempty(s) && return String[]
    s == "all" && return String[v[1] for v in VARIANTS]
    s == "1" && return ["1B"]
    parts = String[]
    for token in split(s, ',')
        t = strip(token)
        isempty(t) || push!(parts, String(t))
    end
    return parts
end

const SELECTED = _selected_variants(get(ENV, "ALLSPARK_TEST_PARITY", ""))

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

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping Llama-$(name) parity: fixture $(fixture_path) not present. " *
            "Generate it with `python3 test/fixtures/record_llama_parity.py $(name)`."
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

    @testset verbose = true "Llama-$(name) ($(repo_id))" begin
        @testset "tokenizer matches HF byte-for-byte (no BOS)" begin
            tk = load_tokenizer(snapshot_dir)
            ids = encode(tk, prompt)
            @test ids == expected_ids
        end

        @testset "last-position logits within $(tolerance)" begin
            cfg = _load_llama_config(snapshot_dir)
            lm = LlamaForCausalLM(cfg)
            load_state_dict!(lm, load_weights(snapshot_dir))

            input_mat = reshape(expected_ids, :, 1)
            logits = lm(input_mat; step=1)
            last_logits = collect(logits[:, end, 1])

            # Argmax token must match HF.
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

            # Top-10 set membership. Exact rank order may differ near the
            # boundary due to fp accumulation differences; one swap is fine.
            allspark_top10 = sortperm(last_logits; rev=true)[1:10] .- 1
            hf_top10 = expected_top_indices[1:10]
            overlap = length(intersect(Set(allspark_top10), Set(hf_top10)))
            @test overlap >= 9
        end
    end
end

# Surface a clear error if someone selected a variant that isn't known.
unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown ALLSPARK_TEST_PARITY variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]), \"all\", or \"1\" (legacy → 1B).",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
