using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Models:
    load_weights, Dinov2Config, Dinov2Model, load_state_dict!, config_from_json,
    read_config

# DINOv2 parity: the CLS feature (HF `pooler_output`, i.e. the CLS token after
# the final LayerNorm). The fixture stores the preprocessed pixel tensor, so this
# verifies the backbone forward without an image pipeline. Record with
# `record_dinov2_parity.py`.
const VARIANTS = (("base", "dinov2_base_parity.json"),)

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

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_DINOV2", ""))

# The canonical parser lives in Models; the test exercises it rather than keeping
# a second copy that can drift (this one read `intermediate_size`, which DINOv2's
# config.json does not have).
_load_dinov2_config(snapshot_dir::AbstractString) =
    config_from_json(Dinov2Config, read_config(snapshot_dir))

function _pixels_from_fixture(flat::Vector{Float32}, shape)
    n, c, hh, ww = Int.(shape)
    a = reshape(flat, ww, hh, c, n)
    return permutedims(a, (3, 2, 1, 4))
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping DINOv2-$(name) parity: fixture $(fixture_path) not present. " *
            "Generate it with `python3 test/fixtures/record_dinov2_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    pixels = _pixels_from_fixture(
        Float32[Float32(x) for x in fixture.pixel_values], fixture.pixel_shape
    )
    expected = Float32[Float32(x) for x in fixture.pooler_output]
    tolerance = Float32(get(fixture, :tolerance, 1.0e-2))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "DINOv2-$(name) ($(repo_id))" begin
        cfg = _load_dinov2_config(snapshot_dir)
        m = Dinov2Model(cfg)
        load_state_dict!(m, load_weights(snapshot_dir))

        cls = collect(m(pixels)[:, 1, 1])            # CLS after final LayerNorm
        @test length(cls) == length(expected)
        @test maximum(abs.(cls .- expected)) < tolerance
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown HFT_TEST_PARITY_DINOV2 variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
