# Deterministic model inputs, mirroring test/fixtures/parity_inputs.py exactly.
# Fixtures store only the model's outputs; both languages regenerate the input.
# Computed in Float64 and cast to Float32, as the Python side does.

"""
    pixel_pattern(channels, height, width) -> Array{Float32,4}

`(C, H, W, 1)` in this package's layout, where `px[c+1, h+1, w+1, 1]` equals the
PyTorch tensor's `[0, c, h, w]`.
"""
function pixel_pattern(channels::Integer, height::Integer, width::Integer)
    px = Array{Float32,4}(undef, channels, height, width, 1)
    for c in 0:(channels - 1), h in 0:(height - 1), w in 0:(width - 1)
        v = sin(0.31 * h + 0.17 * w + 1.3 * c) + 0.5 * cos(0.13 * h - 0.29 * w + 0.7 * c)
        px[c + 1, h + 1, w + 1, 1] = Float32(v)
    end
    return px
end

"""
    feature_pattern(mel_bins, frames) -> Array{Float32,3}

`(mel, frames, 1)`, where `f[m+1, t+1, 1]` equals the PyTorch tensor's `[0, m, t]`.
"""
function feature_pattern(mel_bins::Integer, frames::Integer)
    f = Array{Float32,3}(undef, mel_bins, frames, 1)
    for m in 0:(mel_bins - 1), t in 0:(frames - 1)
        f[m + 1, t + 1, 1] = Float32(0.6 * sin(0.19 * m + 0.031 * t) + 0.4 * cos(0.07 * m - 0.013 * t))
    end
    return f
end
