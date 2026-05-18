function _build_bytes_to_unicode()
    bs = UInt8[]
    for b in 0x21:0x7e
        push!(bs, b)
    end
    for b in 0xa1:0xac
        push!(bs, b)
    end
    for b in 0xae:0xff
        push!(bs, b)
    end
    cs = Char[Char(Int(b)) for b in bs]
    n = 0
    for b in 0x00:0xff
        if !(b in bs)
            push!(bs, b)
            push!(cs, Char(256 + n))
            n += 1
        end
    end
    return Dict{UInt8,Char}(zip(bs, cs))
end

const BYTES_TO_UNICODE = _build_bytes_to_unicode()
const UNICODE_TO_BYTES = Dict{Char,UInt8}(c => b for (b, c) in BYTES_TO_UNICODE)

"""
    bytes_to_string(s) -> String

GPT-2 byte-level encoding. Treats the UTF-8 bytes of `s` as opaque values
and maps each through `BYTES_TO_UNICODE` to a printable Unicode character.
The output is what BPE merges operate on.
"""
function bytes_to_string(s::AbstractString)
    io = IOBuffer()
    for b in codeunits(s)
        print(io, BYTES_TO_UNICODE[b])
    end
    return String(take!(io))
end

"""
    string_to_bytes(s) -> String

Inverse of [`bytes_to_string`](@ref). Iterates `s` per character, looks each
up in `UNICODE_TO_BYTES`, and reassembles the original UTF-8 string.
"""
function string_to_bytes(s::AbstractString)
    out = UInt8[]
    for c in s
        push!(out, UNICODE_TO_BYTES[c])
    end
    return String(out)
end
