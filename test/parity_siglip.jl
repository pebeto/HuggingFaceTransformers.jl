using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Models:
    load_weights,
    SiglipModel,
    SiglipConfig,
    SiglipVisionConfig,
    SiglipTextConfig,
    load_state_dict!

# SigLIP parity: the full image + text → logits path. The fixture stores the
# preprocessed pixel tensor and the tokenized (padded to 64) text ids, so this
# verifies both towers and the contrastive head without an image or SentencePiece
# dependency. Record with `record_siglip_parity.py`.
const VARIANTS = (("base", "siglip_base_patch16_224_parity.json"),)

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

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_SIGLIP", ""))

function _load_siglip_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    v, t = raw.vision_config, raw.text_config
    return SiglipConfig(;
        vision=SiglipVisionConfig(;
            hidden_size=Int(get(v, :hidden_size, 768)),
            intermediate_size=Int(get(v, :intermediate_size, 3072)),
            num_hidden_layers=Int(get(v, :num_hidden_layers, 12)),
            num_attention_heads=Int(get(v, :num_attention_heads, 12)),
            num_channels=Int(get(v, :num_channels, 3)),
            image_size=Int(get(v, :image_size, 224)),
            patch_size=Int(get(v, :patch_size, 16)),
            layer_norm_eps=Float64(get(v, :layer_norm_eps, 1.0e-6)),
        ),
        text=SiglipTextConfig(;
            vocab_size=Int(get(t, :vocab_size, 32000)),
            hidden_size=Int(get(t, :hidden_size, 768)),
            intermediate_size=Int(get(t, :intermediate_size, 3072)),
            num_hidden_layers=Int(get(t, :num_hidden_layers, 12)),
            num_attention_heads=Int(get(t, :num_attention_heads, 12)),
            max_position_embeddings=Int(get(t, :max_position_embeddings, 64)),
            projection_size=Int(get(t, :projection_size, get(t, :hidden_size, 768))),
            layer_norm_eps=Float64(get(t, :layer_norm_eps, 1.0e-6)),
        ),
    )
end

function _pixels_from_fixture(flat::Vector{Float32}, shape)
    n, c, hh, ww = Int.(shape)
    a = reshape(flat, ww, hh, c, n)
    return permutedims(a, (3, 2, 1, 4))
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping SigLIP-$(name) parity: fixture $(fixture_path) not present. " *
            "Generate it with `python3 test/fixtures/record_siglip_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    pixels = _pixels_from_fixture(
        Float32[Float32(x) for x in fixture.pixel_values], fixture.pixel_shape
    )
    # input_ids is a list of per-text id rows; stack as (seq, n_texts).
    ids = reduce(hcat, [Int[Int(x) for x in row] for row in fixture.input_ids])
    expected = Float32[Float32(x) for x in fixture.logits_per_image]
    expected_argmax = Int(fixture.argmax_text)
    tolerance = Float32(get(fixture, :tolerance, 1.0e-2))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "SigLIP-$(name) ($(repo_id))" begin
        cfg = _load_siglip_config(snapshot_dir)
        m = SiglipModel(cfg)
        load_state_dict!(m, load_weights(snapshot_dir))

        logits = collect(m(pixels, ids)[1, :])       # (n_texts,)
        @test length(logits) == length(expected)
        @test argmax(logits) - 1 == expected_argmax
        @test maximum(abs.(logits .- expected)) < tolerance
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown HFT_TEST_PARITY_SIGLIP variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
