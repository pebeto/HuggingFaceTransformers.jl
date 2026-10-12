using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Models: QwenForCausalLM, load_pretrained

# Variant label → fixture filename. Each fixture is recorded once via
# test/fixtures/record_qwen_parity.py. The Qwen2.5 variants span both branches
# of the tied-embeddings flag (small variants tie, larger don't); Qwen3 adds the
# bias-free attention with per-head query/key norms.
const VARIANTS = (
    ("0.5B", "qwen2_5_0_5b_instruct_parity.json"),
    ("1.5B", "qwen2_5_1_5b_instruct_parity.json"),
    ("3B", "qwen2_5_3b_instruct_parity.json"),
    ("7B", "qwen2_5_7b_instruct_parity.json"),
    ("Qwen3-0.6B", "qwen3_0_6b_parity.json"),
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

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_QWEN", ""))

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping Qwen $(name) parity: fixture $(fixture_path) " *
            "not present. Generate it with " *
            "`python3 test/fixtures/record_qwen_parity.py $(name)`."
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

    @testset verbose = true "Qwen $(name) ($(repo_id))" begin
        @testset "tokenizer matches HF byte-for-byte (no BOS)" begin
            tk = load_tokenizer(snapshot_dir)
            ids = encode(tk, prompt)
            @test ids == expected_ids
        end

        @testset "last-position logits within $(tolerance)" begin
            # Through the auto path, so config parsing is part of what is checked.
            lm = load_pretrained(snapshot_dir)
            @test lm isa QwenForCausalLM

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
    "Unknown HFT_TEST_PARITY_QWEN variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
