using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Models:
    load_weights, ViTConfig, ViTForImageClassification, load_state_dict!

# ViT image-classification parity against HF. The fixture stores the already
# preprocessed `pixel_values` tensor (image decode/resize is out of scope), so
# this verifies the model forward, not the image pipeline. Record with
# `record_vit_parity.py`.
const VARIANTS = (("base", "vit_base_patch16_224_parity.json"),)

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

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_VIT", ""))

function _load_vit_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    num_labels =
        haskey(raw, :id2label) ? length(raw.id2label) : Int(get(raw, :num_labels, 1000))
    return ViTConfig(;
        hidden_size=Int(raw.hidden_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        intermediate_size=Int(raw.intermediate_size),
        image_size=Int(get(raw, :image_size, 224)),
        patch_size=Int(get(raw, :patch_size, 16)),
        num_channels=Int(get(raw, :num_channels, 3)),
        layer_norm_eps=Float64(get(raw, :layer_norm_eps, 1.0e-12)),
        num_labels=num_labels,
        qkv_bias=Bool(get(raw, :qkv_bias, true)),
    )
end

# Rebuild the (C, H, W, N) model input from a numpy (N, C, H, W) C-order flat.
function _pixels_from_fixture(flat::Vector{Float32}, shape)
    n, c, hh, ww = Int.(shape)
    a = reshape(flat, ww, hh, c, n)       # column-major undo of C-order
    return permutedims(a, (3, 2, 1, 4))   # (C, H, W, N)
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping ViT-$(name) parity: fixture $(fixture_path) not present. " *
            "Generate it with `python3 test/fixtures/record_vit_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    flat = Float32[Float32(x) for x in fixture.pixel_values]
    expected_argmax = Int(fixture.argmax_label)
    expected_top_indices = Int[Int(x) for x in fixture.top_indices]
    expected_top_logits = Float32[Float32(x) for x in fixture.top_logits]
    tolerance = Float32(get(fixture, :tolerance, 1.0e-3))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "ViT-$(name) ($(repo_id))" begin
        cfg = _load_vit_config(snapshot_dir)
        m = ViTForImageClassification(cfg)
        load_state_dict!(m, load_weights(snapshot_dir))

        pixels = _pixels_from_fixture(flat, fixture.pixel_shape)
        logits = collect(m(pixels)[:, 1])

        @test argmax(logits) - 1 == expected_argmax

        max_err = Float32(0)
        for (i, hf_idx) in enumerate(expected_top_indices)
            max_err = max(
                max_err, abs(Float32(logits[hf_idx + 1]) - expected_top_logits[i])
            )
        end
        @test max_err < tolerance

        jl_top10 = sortperm(logits; rev=true)[1:10] .- 1
        overlap = length(intersect(Set(jl_top10), Set(expected_top_indices[1:10])))
        @test overlap >= 9
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown HFT_TEST_PARITY_VIT variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
