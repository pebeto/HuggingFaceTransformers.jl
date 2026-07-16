using Test
using JSON3
using Allspark.HFHub: snapshot_download
using Allspark.Models: load_weights, WhisperConfig, WhisperModel, load_state_dict!

# Whisper parity: the teacher-forced forward (input_features + decoder_input_ids
# → logits). The fixture stores the preprocessed log-mel features (audio → mel is
# out of scope), so this verifies the encoder-decoder without an audio frontend.
# Record with `record_whisper_parity.py`.
const VARIANTS = (("base", "whisper_base_parity.json"),)

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

const SELECTED = _selected_variants(get(ENV, "ALLSPARK_TEST_PARITY_WHISPER", ""))

function _load_whisper_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    return WhisperConfig(;
        vocab_size=Int(raw.vocab_size),
        num_mel_bins=Int(raw.num_mel_bins),
        d_model=Int(raw.d_model),
        encoder_layers=Int(raw.encoder_layers),
        encoder_attention_heads=Int(raw.encoder_attention_heads),
        encoder_ffn_dim=Int(raw.encoder_ffn_dim),
        decoder_layers=Int(raw.decoder_layers),
        decoder_attention_heads=Int(raw.decoder_attention_heads),
        decoder_ffn_dim=Int(raw.decoder_ffn_dim),
        max_source_positions=Int(raw.max_source_positions),
        max_target_positions=Int(raw.max_target_positions),
        decoder_start_token_id=Int(get(raw, :decoder_start_token_id, 50258)),
        eos_token_id=Int(get(raw, :eos_token_id, 50257)),
    )
end

# Rebuild (num_mel_bins, frames, N) from a numpy (N, mel, frames) C-order flat.
function _features_from_fixture(flat::Vector{Float32}, shape)
    n, mel, frames = Int.(shape)
    a = reshape(flat, frames, mel, n)
    return permutedims(a, (2, 1, 3))
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping Whisper-$(name) parity: fixture $(fixture_path) not present. " *
            "Generate it with `python3 test/fixtures/record_whisper_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    features = _features_from_fixture(
        Float32[Float32(x) for x in fixture.input_features], fixture.feature_shape
    )
    ids = reshape(Int[Int(x) for x in fixture.decoder_input_ids], :, 1)
    expected_argmax = Int(fixture.argmax_token_id)
    expected_top_indices = Int[Int(x) for x in fixture.top_indices]
    expected_top_logits = Float32[Float32(x) for x in fixture.top_logits]
    tolerance = Float32(get(fixture, :tolerance, 1.0e-2))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "Whisper-$(name) ($(repo_id))" begin
        cfg = _load_whisper_config(snapshot_dir)
        m = WhisperModel(cfg)
        load_state_dict!(m, load_weights(snapshot_dir))

        logits = collect(m(features, ids)[:, end, 1])
        @test argmax(logits) - 1 == expected_argmax

        max_err = Float32(0)
        for (i, hf_idx) in enumerate(expected_top_indices)
            max_err = max(max_err, abs(Float32(logits[hf_idx + 1]) - expected_top_logits[i]))
        end
        @test max_err < tolerance

        allspark_top10 = sortperm(logits; rev=true)[1:10] .- 1
        overlap = length(intersect(Set(allspark_top10), Set(expected_top_indices[1:10])))
        @test overlap >= 9
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown ALLSPARK_TEST_PARITY_WHISPER variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
