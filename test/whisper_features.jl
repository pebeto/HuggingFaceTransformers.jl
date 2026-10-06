using Test
using Random
using JSON3
using HuggingFaceTransformers.Models:
    WhisperFeatureExtractor, mel_filter_bank, load_feature_extractor, WhisperConfig,
    WhisperModel

# Mirrors `signal()` in test/fixtures/record_whisper_features_parity.py exactly, so
# the fixture never has to store the audio.
function _test_signal(n=8_000, sr=16_000)
    t = (0:(n - 1)) ./ sr
    x = @. 0.4 * sin(2π * 440 * t) + 0.25 * sin(2π * 1250 * t) +
        0.15 * sin(2π * 3700 * t) + 0.2 * sin(2π * (200 + 900 * t) * t)
    return Float32.(x)
end

const _WHISPER_FEATURES_FIXTURE =
    joinpath(@__DIR__, "fixtures", "whisper_features_parity.json")

@testset verbose = true "matches HF's WhisperFeatureExtractor" begin
    # No model weights and no download: the extractor is pure arithmetic and the
    # audio is regenerated, so this parity check runs in the default suite.
    if !isfile(_WHISPER_FEATURES_FIXTURE)
        @info "Skipping Whisper feature parity: run " *
            "`python3 test/fixtures/record_whisper_features_parity.py`"
    else
        fx = JSON3.read(read(_WHISPER_FEATURES_FIXTURE, String))
        tol = Float64(fx.tolerance)
        features = WhisperFeatureExtractor()(_test_signal(Int(fx.n_samples)))

        @test size(features) == Tuple(Int.(fx.shape))
        head = permutedims(reduce(hcat, [Float64.(row) for row in fx.head]))
        # The frames carrying signal match value for value.
        @test maximum(abs.(features[:, 1:Int(fx.head_frames)] .- head)) < tol
        # Everything after the audio is silence, floored to one constant.
        @test all(abs.(features[:, (Int(fx.head_frames) + 1):end] .- fx.plateau) .< tol)
    end
end

@testset verbose = true "mel filter bank" begin
    filters = mel_filter_bank(201, 80, 0.0, 8000.0, 16_000)
    @test size(filters) == (201, 80)
    @test all(>=(0), filters)
    @test all(>(0), vec(maximum(filters; dims=1)))      # no filter is empty

    # Without Slaney normalization a triangle never exceeds 1. It can peak well
    # below that when its apex falls between bins, which happens to the narrow
    # low-frequency filters (two of these peak near 0.48).
    peaks = vec(maximum(mel_filter_bank(201, 80, 0.0, 8000.0, 16_000; norm=nothing); dims=1))
    @test all(p -> 0 < p <= 1.0, peaks)

    @test size(mel_filter_bank(201, 40, 0.0, 8000.0, 16_000; mel_scale=:htk)) == (201, 40)
    @test_throws ArgumentError mel_filter_bank(201, 80, 0.0, 8000.0, 16_000; mel_scale=:bark)
    @test_throws ArgumentError mel_filter_bank(201, 80, 0.0, 8000.0, 16_000; norm=:peak)
    @test_throws ArgumentError mel_filter_bank(201, 80, 9000.0, 8000.0, 16_000)
end

@testset verbose = true "WhisperFeatureExtractor" begin
    fe = WhisperFeatureExtractor()

    @testset "always 30 s of frames" begin
        # Short input is padded, long input truncated, so the model always sees
        # the frame count it was trained on.
        @test size(fe(_test_signal(1_600))) == (80, 3000)
        @test size(fe(_test_signal(16_000 * 31))) == (80, 3000)
    end

    @testset "truncation keeps exactly the first chunk" begin
        long = _test_signal(16_000 * 31)
        @test fe(long) == fe(long[1:(16_000 * 30)])
    end

    @testset "silence is a single floor value" begin
        @test length(unique(fe(zeros(Float32, 16_000)))) == 1
    end

    @testset "batching clamps each row against its own maximum" begin
        a, b = _test_signal(4_000), 0.01f0 .* _test_signal(12_000)
        batch = fe([a, b])
        @test size(batch) == (80, 3000, 2)
        @test batch[:, :, 1] == fe(a)
        @test batch[:, :, 2] == fe(b)
        @test_throws ArgumentError fe(Vector{Float32}[])
    end

    @testset "integer and Float64 input see the same samples" begin
        # HF casts to Float32 before anything else; so do we.
        x = _test_signal(4_000)
        @test fe(Float64.(x)) == fe(x)
    end

    @testset "features fit the model" begin
        # The extractor's (mel, frames) layout is what WhisperModel takes.
        Random.seed!(0xE1)
        cfg = WhisperConfig(;
            vocab_size=20, num_mel_bins=80, d_model=8, encoder_layers=1,
            encoder_attention_heads=2, encoder_ffn_dim=16, decoder_layers=1,
            decoder_attention_heads=2, decoder_ffn_dim=16, max_source_positions=1500,
            max_target_positions=8,
        )
        logits = WhisperModel(cfg)(reshape(fe(_test_signal()), 80, 3000, 1), reshape([1, 2], :, 1))
        @test size(logits) == (20, 2, 1)
        @test all(isfinite, logits)
    end
end

@testset verbose = true "load_feature_extractor" begin
    mktempdir() do dir
        cfg = Dict(
            "feature_extractor_type" => "WhisperFeatureExtractor",
            "feature_size" => 80, "sampling_rate" => 16_000, "hop_length" => 160,
            "chunk_length" => 30, "n_fft" => 400, "padding_value" => 0.0,
            # Real configs carry the filters inline; they are recomputed instead.
            "mel_filters" => [[0.0]],
        )
        write(joinpath(dir, "preprocessor_config.json"), JSON3.write(cfg))
        fe = load_feature_extractor(dir)
        @test fe isa WhisperFeatureExtractor
        @test fe(_test_signal()) == WhisperFeatureExtractor()(_test_signal())

        write(
            joinpath(dir, "preprocessor_config.json"),
            JSON3.write(Dict("feature_extractor_type" => "Wav2Vec2FeatureExtractor")),
        )
        @test_throws ArgumentError load_feature_extractor(dir)
        @test_throws ArgumentError load_feature_extractor(mktempdir())
    end
end
