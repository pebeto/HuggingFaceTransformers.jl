const GPT2_PRETOKENIZE_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

abstract type PreTokenizer end

struct IdentityPreTokenizer <: PreTokenizer end

struct SequencePreTokenizer <: PreTokenizer
    steps::Vector{PreTokenizer}
end

struct ByteLevelPreTokenizer <: PreTokenizer
    add_prefix_space::Bool
    use_regex::Bool
end

struct SplitPreTokenizer <: PreTokenizer
    pattern::Regex
    behavior::Symbol
end

apply_pre(::IdentityPreTokenizer, ts::Vector{String}) = ts

function apply_pre(p::SequencePreTokenizer, ts::Vector{String})
    for step in p.steps
        ts = apply_pre(step, ts)
    end
    return ts
end

function apply_pre(p::SplitPreTokenizer, ts::Vector{String})
    out = String[]
    for t in ts
        last_end = 0
        for m in eachmatch(p.pattern, t)
            start = m.offset
            stop = m.offset + ncodeunits(m.match) - 1
            if start > last_end + 1
                push!(out, String(SubString(t, last_end + 1, prevind(t, start))))
            end
            if p.behavior === :isolated
                push!(out, String(m.match))
            end
            last_end = stop
        end
        if last_end < ncodeunits(t)
            push!(out, String(SubString(t, last_end + 1, ncodeunits(t))))
        end
    end
    return filter(!isempty, out)
end

function apply_pre(p::ByteLevelPreTokenizer, ts::Vector{String})
    out = String[]
    for t in ts
        s = t
        if p.add_prefix_space && !startswith(s, ' ')
            s = " " * s
        end
        if p.use_regex
            for m in eachmatch(GPT2_PRETOKENIZE_REGEX, s)
                push!(out, bytes_to_string(m.match))
            end
        else
            push!(out, bytes_to_string(s))
        end
    end
    return out
end

abstract type Decoder end

struct ByteLevelDecoder <: Decoder end

struct SequenceDecoder <: Decoder
    steps::Vector{Decoder}
end

apply_dec(::ByteLevelDecoder, s::AbstractString) = string_to_bytes(s)

function apply_dec(d::SequenceDecoder, s::AbstractString)
    for step in d.steps
        s = apply_dec(step, s)
    end
    return s
end
