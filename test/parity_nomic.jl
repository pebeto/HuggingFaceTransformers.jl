using Test
using JSON3
using Allspark.HFHub: snapshot_download
using Allspark.Models: load_weights, NomicBertConfig, BertEmbeddingModel, embed, load_state_dict!

# Parity for nomic-embed against HF (trust_remote_code NomicBertModel) with mean
# pooling + L2 normalization. Record with `record_nomic_parity.py`.
const VARIANTS = (("v1.5", "nomic_embed_v1_5_parity.json"),)

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

const SELECTED = _selected_variants(get(ENV, "ALLSPARK_TEST_PARITY_NOMIC", ""))

function _load_nomic_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    hidden = Int(get(raw, :n_embd, get(raw, :hidden_size, 768)))
    n_head = Int(get(raw, :n_head, get(raw, :num_attention_heads, 12)))
    rope_theta = if haskey(raw, :rotary_emb_base)
        Float64(raw.rotary_emb_base)
    elseif haskey(raw, :rope_parameters)
        Float64(get(raw.rope_parameters, :rope_theta, 1000.0))
    else
        1000.0
    end
    return NomicBertConfig(;
        vocab_size=Int(get(raw, :vocab_size, 30528)),
        hidden_size=hidden,
        intermediate_size=Int(get(raw, :n_inner, get(raw, :intermediate_size, 4 * hidden))),
        num_hidden_layers=Int(get(raw, :n_layer, get(raw, :num_hidden_layers, 12))),
        num_attention_heads=n_head,
        head_dim=Int(get(raw, :head_dim, hidden ÷ n_head)),
        type_vocab_size=Int(get(raw, :type_vocab_size, 2)),
        layer_norm_eps=Float64(
            get(raw, :layer_norm_epsilon, get(raw, :layer_norm_eps, 1.0e-12))
        ),
        rope_theta=rope_theta,
    )
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping nomic-$(name) parity: fixture $(fixture_path) not present. " *
            "Generate it with `python3 test/fixtures/record_nomic_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    expected_ids = Int[Int(x) for x in fixture.input_token_ids]
    expected = Float32[Float32(x) for x in fixture.embedding]
    tolerance = Float32(get(fixture, :tolerance, 1.0e-3))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "nomic-$(name) ($(repo_id))" begin
        cfg = _load_nomic_config(snapshot_dir)
        m = BertEmbeddingModel(cfg; pooling=:mean, normalize=true)
        load_state_dict!(m, load_weights(snapshot_dir))

        v = embed(m, expected_ids)
        @test length(v) == length(expected)
        @test maximum(abs.(v .- expected)) < tolerance
        @test sum(v .* expected) > 1 - tolerance       # cosine similarity
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown ALLSPARK_TEST_PARITY_NOMIC variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
