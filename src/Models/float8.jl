"""
    Float8E4M3

The 8-bit float of FP8-quantized checkpoints (`F8_E4M3` in safetensors, torch's
`float8_e4m3fn`): one sign bit, four exponent bits with bias 7, three mantissa
bits, no infinities, and NaN at `0x7f` / `0xff`. It only converts to `Float32`;
arithmetic happens after decoding.
"""
primitive type Float8E4M3 <: AbstractFloat 8 end

function _decode_e4m3(b::UInt8)
    sign = b & 0x80 == 0 ? 1.0f0 : -1.0f0
    exponent, mantissa = Int((b >> 3) & 0x0f), Int(b & 0x07)
    exponent == 15 && mantissa == 7 && return NaN32
    exponent == 0 && return sign * ldexp(Float32(mantissa) / 8, -6)
    return sign * ldexp(1.0f0 + Float32(mantissa) / 8, exponent - 7)
end

const _E4M3 = [_decode_e4m3(UInt8(b)) for b in 0:255]

Base.Float32(x::Float8E4M3) = _E4M3[reinterpret(UInt8, x) + 1]
Base.Float64(x::Float8E4M3) = Float64(Float32(x))
Base.convert(::Type{Float32}, x::Float8E4M3) = Float32(x)
Base.promote_rule(::Type{Float32}, ::Type{Float8E4M3}) = Float32
Base.show(io::IO, x::Float8E4M3) = print(io, "Float8E4M3(", Float32(x), ")")
