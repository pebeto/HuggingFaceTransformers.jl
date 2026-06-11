using Test
using JSON3
using Allspark.HFHub: snapshot_download
using Allspark.Tokenizers: load_tokenizer, encode, decode

# Tokenizer parity harness. Each variant ties an Allspark-side label to a
# fixture file recorded by `test/fixtures/record_tokenizer_parity.py`. The
# fixture itself names the HF repo and the family (informational only —
# `load_tokenizer` dispatches on the JSON contents), so adding a variant
# here doesn't require new model-side code.
#
# Coverage spans every tokenizer family in the tree:
#   - byte-level BPE      : gpt2, roberta, qwen2.5, pythia, phi-3, llama3.2, mistral-v0.2
#   - Unigram + byte fb   : gemma-2
#   - WordPiece           : bert-base-uncased, bert-base-cased
const VARIANTS = (
    ("gpt2", "tokenizer_parity_gpt2.json"),
    ("roberta", "tokenizer_parity_roberta.json"),
    ("qwen2.5", "tokenizer_parity_qwen2_5.json"),
    ("pythia", "tokenizer_parity_pythia.json"),
    ("phi-3", "tokenizer_parity_phi_3.json"),
    ("llama3.2", "tokenizer_parity_llama_3_2.json"),
    ("mistral-v0.2", "tokenizer_parity_mistral_v02.json"),
    ("gemma-2", "tokenizer_parity_gemma_2.json"),
    ("bert-uncased", "tokenizer_parity_bert_uncased.json"),
    ("bert-cased", "tokenizer_parity_bert_cased.json"),
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

const SELECTED = _selected_variants(get(ENV, "ALLSPARK_TEST_PARITY_TOKENIZER", ""))

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping tokenizer parity for $(name): fixture " *
              "$(fixture_path) not present. Generate it with " *
              "`python3 test/fixtures/record_tokenizer_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    family = String(get(fixture, :tokenizer_family, "unknown"))
    samples = fixture.samples::JSON3.Array

    snapshot_dir = snapshot_download(repo_id; verbose=false)
    tk = load_tokenizer(snapshot_dir)

    @testset verbose = true "$(name) ($(family) — $(repo_id))" begin
        encode_pass = Ref(0)
        decode_pass = Ref(0)
        for (i, sample) in enumerate(samples)
            text = String(sample.text::AbstractString)
            expected_ids = Int[Int(x) for x in sample.ids::JSON3.Array]
            expected_decoded = String(get(sample, :decoded, text))

            label = "[$(i)] " * (length(text) > 40 ? text[1:40] * "…" : text)

            @testset "$(label) — encode" begin
                ids = encode(tk, text)
                @test ids == expected_ids
                ids == expected_ids && (encode_pass[] += 1)
            end

            @testset "$(label) — decode round-trip" begin
                decoded = decode(tk, expected_ids)
                @test decoded == expected_decoded
                decoded == expected_decoded && (decode_pass[] += 1)
            end
        end
        @info "$(name): $(encode_pass[])/$(length(samples)) encode OK, " *
              "$(decode_pass[])/$(length(samples)) decode OK"
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown ALLSPARK_TEST_PARITY_TOKENIZER variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
