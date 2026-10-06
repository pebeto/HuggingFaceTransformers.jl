# Slaney's mel scale is linear below 1 kHz and logarithmic above; HTK is
# logarithmic throughout. Whisper uses Slaney.
function _hz_to_mel(f::Real, scale::Symbol)
    scale === :htk && return 2595.0 * log10(1.0 + f / 700.0)
    f < 1000.0 && return 3.0 * f / 200.0
    return 15.0 + log(f / 1000.0) * (27.0 / log(6.4))
end

function _mel_to_hz(m::Real, scale::Symbol)
    scale === :htk && return 700.0 * (10.0^(m / 2595.0) - 1.0)
    m < 15.0 && return 200.0 * m / 3.0
    return 1000.0 * exp((log(6.4) / 27.0) * (m - 15.0))
end

"""
    mel_filter_bank(num_frequency_bins, num_mel_filters, min_frequency,
                    max_frequency, sampling_rate; norm = :slaney, mel_scale = :slaney)
        -> Matrix{Float64}

Triangular mel filters of shape `(num_frequency_bins, num_mel_filters)`, matching
HF's `audio_utils.mel_filter_bank`. Filters are evenly spaced on the mel scale and
triangular in hertz; `norm = :slaney` scales each to unit area, `nothing` leaves
them peak-normalized.
"""
function mel_filter_bank(
    num_frequency_bins::Integer,
    num_mel_filters::Integer,
    min_frequency::Real,
    max_frequency::Real,
    sampling_rate::Integer;
    norm::Union{Nothing,Symbol}=:slaney,
    mel_scale::Symbol=:slaney,
)
    mel_scale in (:slaney, :htk) ||
        throw(ArgumentError("mel_scale must be :slaney or :htk, got :$(mel_scale)"))
    norm in (nothing, :slaney) ||
        throw(ArgumentError("norm must be nothing or :slaney, got :$(norm)"))
    min_frequency <= max_frequency ||
        throw(ArgumentError("min_frequency must not exceed max_frequency"))

    mels = range(
        _hz_to_mel(min_frequency, mel_scale), _hz_to_mel(max_frequency, mel_scale);
        length=num_mel_filters + 2,
    )
    edges = [_mel_to_hz(m, mel_scale) for m in mels]          # filter corner frequencies
    bins = range(0.0, Float64(sampling_rate ÷ 2); length=num_frequency_bins)

    widths = diff(edges)
    filters = zeros(Float64, num_frequency_bins, num_mel_filters)
    for j in 1:num_mel_filters, (i, f) in enumerate(bins)
        rising = (f - edges[j]) / widths[j]
        falling = (edges[j + 2] - f) / widths[j + 1]
        filters[i, j] = max(0.0, min(rising, falling))
    end

    if norm === :slaney
        for j in 1:num_mel_filters
            filters[:, j] .*= 2.0 / (edges[j + 2] - edges[j])
        end
    end
    return filters
end

"""
    WhisperFeatureExtractor(; feature_size = 80, sampling_rate = 16_000,
                            hop_length = 160, chunk_length = 30, n_fft = 400,
                            padding_value = 0.0)

Turns a raw waveform into the log-mel features [`WhisperModel`](@ref) and
[`transcribe`](@ref) take, matching HF's `WhisperFeatureExtractor`.

Call it on a vector of samples (16 kHz mono) to get a `(feature_size, frames)`
`Float32` matrix, or on a vector of waveforms for a `(feature_size, frames, batch)`
array. Audio is padded with `padding_value` or truncated to `chunk_length`
seconds first, so the frame count is always `chunk_length * sampling_rate ÷
hop_length` (3000 for the defaults).

The pipeline is HF's: a periodic Hann window, a centred STFT with reflect padding,
the last frame dropped, power spectrum, Slaney mel projection, `log10` floored at
`1e-10`, a clamp to within 8 of the maximum, then `(x + 4) / 4`. The 400-point
transform is small enough to apply as a precomputed DFT matrix, so no FFT library
is needed.
"""
struct WhisperFeatureExtractor
    feature_size::Int
    sampling_rate::Int
    hop_length::Int
    chunk_length::Int
    n_fft::Int
    padding_value::Float64
    n_samples::Int
    window::Vector{Float64}
    mel_filters::Matrix{Float64}      # (n_fft ÷ 2 + 1, feature_size)
    dft_real::Matrix{Float64}         # (n_fft ÷ 2 + 1, n_fft)
    dft_imag::Matrix{Float64}
end

function WhisperFeatureExtractor(;
    feature_size::Integer=80,
    sampling_rate::Integer=16_000,
    hop_length::Integer=160,
    chunk_length::Integer=30,
    n_fft::Integer=400,
    padding_value::Real=0.0,
)
    n_freq = n_fft ÷ 2 + 1
    # Periodic Hann, which is what `torch.hann_window` returns by default.
    window = [0.5 - 0.5 * cos(2π * n / n_fft) for n in 0:(n_fft - 1)]
    dft_real = [cos(2π * k * n / n_fft) for k in 0:(n_freq - 1), n in 0:(n_fft - 1)]
    dft_imag = [-sin(2π * k * n / n_fft) for k in 0:(n_freq - 1), n in 0:(n_fft - 1)]
    filters = mel_filter_bank(n_freq, feature_size, 0.0, 8000.0, sampling_rate)
    return WhisperFeatureExtractor(
        feature_size, sampling_rate, hop_length, chunk_length, n_fft,
        Float64(padding_value), chunk_length * sampling_rate, window, filters,
        dft_real, dft_imag,
    )
end

function (fe::WhisperFeatureExtractor)(waveform::AbstractVector{<:Real})
    # HF casts the input to Float32 before anything else, so do the same to see
    # identical samples; the arithmetic then runs in Float64.
    samples = Float64.(Float32.(waveform))
    if length(samples) >= fe.n_samples
        samples = samples[1:(fe.n_samples)]
    else
        samples = vcat(samples, fill(fe.padding_value, fe.n_samples - length(samples)))
    end

    # Centred frames: reflect-pad by half a window, excluding the edge sample.
    half = fe.n_fft ÷ 2
    n = length(samples)
    padded = vcat(samples[(half + 1):-1:2], samples, samples[(n - 1):-1:(n - half)])
    n_frames = 1 + (length(padded) - fe.n_fft) ÷ fe.hop_length

    frames = Matrix{Float64}(undef, fe.n_fft, n_frames)
    for t in 1:n_frames
        start = (t - 1) * fe.hop_length
        @views frames[:, t] .= padded[(start + 1):(start + fe.n_fft)] .* fe.window
    end

    re = fe.dft_real * frames
    im = fe.dft_imag * frames
    # HF drops the final frame: the centred STFT yields one more than the model
    # was trained on.
    power = @views re[:, 1:(end - 1)] .^ 2 .+ im[:, 1:(end - 1)] .^ 2

    logspec = log10.(max.(fe.mel_filters' * power, 1.0e-10))
    logspec = max.(logspec, maximum(logspec) - 8.0)
    return Float32.((logspec .+ 4.0) ./ 4.0)
end

function (fe::WhisperFeatureExtractor)(waveforms::AbstractVector{<:AbstractVector{<:Real}})
    isempty(waveforms) && throw(ArgumentError("waveforms must be non-empty"))
    # Each waveform is clamped against its own maximum, as HF does per batch row.
    return cat((fe(w) for w in waveforms)...; dims=3)
end

"""
    load_feature_extractor(path) -> WhisperFeatureExtractor

Build a feature extractor from a `preprocessor_config.json`, given the file or a
snapshot directory containing it. Only Whisper's extractor exists so far; any
other `feature_extractor_type` raises. Any `mel_filters` stored in the file are
ignored in favour of recomputing them, which is what HF does at runtime too.
"""
function load_feature_extractor(path::AbstractString)
    file = isdir(path) ? joinpath(path, "preprocessor_config.json") : path
    isfile(file) || throw(ArgumentError("preprocessor_config.json not found at $(file)"))
    raw = JSON3.read(read(file, String))

    kind = _hf_str(raw, :feature_extractor_type, "")
    kind == "WhisperFeatureExtractor" || throw(
        ArgumentError("unsupported feature extractor `$(kind)`; only Whisper's exists"),
    )
    return WhisperFeatureExtractor(;
        feature_size=_hf_int(raw, :feature_size, 80),
        sampling_rate=_hf_int(raw, :sampling_rate, 16_000),
        hop_length=_hf_int(raw, :hop_length, 160),
        chunk_length=_hf_int(raw, :chunk_length, 30),
        n_fft=_hf_int(raw, :n_fft, 400),
        padding_value=_hf_float(raw, :padding_value, 0.0),
    )
end
