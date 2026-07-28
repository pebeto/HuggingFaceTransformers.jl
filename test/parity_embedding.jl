using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Models:
    load_weights, BertConfig, BertEmbeddingModel, embed, load_state_dict!

# Sentence-embedding parity against HF (sentence-transformers pooling).
# Variants are BERT-family checkpoints; each fixture records the pooling
# strategy, the prefix, the special-token-bearing input IDs, and the unit-norm
# reference vector. Record with `test/fixtures/record_embedding_parity.py`.
const VARIANTS = (
    ("bge-small", "bge_small_en_parity.json"), ("e5-small", "e5_small_v2_parity.json")
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

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_EMBEDDING", ""))

function _load_embedding_config(snapshot_dir::AbstractString, fixture)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    offset = Int(get(fixture, :position_embedding_offset, 0))
    return BertConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        max_position_embeddings=Int(raw.max_position_embeddings) - offset,
        type_vocab_size=Int(get(raw, :type_vocab_size, 2)),
        layer_norm_eps=Float64(get(raw, :layer_norm_eps, 1.0e-12)),
        pad_token_id=Int(get(raw, :pad_token_id, 0)),
        position_embedding_offset=offset,
        hf_prefix=String(get(fixture, :hf_prefix, "")),
        head_prefix="cls.predictions",   # unused: the head is never loaded
        tie_word_embeddings=true,
    )
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping embedding-$(name) parity: fixture $(fixture_path) not " *
            "present. Generate it with " *
            "`python3 test/fixtures/record_embedding_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    expected_ids = Int[Int(x) for x in fixture.input_token_ids]
    expected = Float32[Float32(x) for x in fixture.embedding]
    pooling = Symbol(String(fixture.pooling))
    normalize = Bool(get(fixture, :normalize, true))
    tolerance = Float32(get(fixture, :tolerance, 1.0e-3))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "embedding-$(name) ($(repo_id))" begin
        cfg = _load_embedding_config(snapshot_dir, fixture)
        m = BertEmbeddingModel(cfg; pooling=pooling, normalize=normalize)
        load_state_dict!(m, load_weights(snapshot_dir))

        v = embed(m, expected_ids)
        @test length(v) == length(expected)
        @test maximum(abs.(v .- expected)) < tolerance
        # Cosine similarity to the reference (both unit-norm when normalize).
        @test sum(v .* expected) > 1 - tolerance
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown HFT_TEST_PARITY_EMBEDDING variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
