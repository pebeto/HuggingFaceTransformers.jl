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

"""
    MetaspacePreTokenizer

SentencePiece-style pretokenizer. Replaces every ASCII space in the
input with `replacement` (typically `▁`, U+2581). With
`prepend_scheme == :always`, a `replacement` is also prepended to the
input; `:first` prepends only when the input is the first chunk in a
sequence; `:never` skips prepending entirely. This implementation
treats all chunks the same way (`:first` and `:always` behave
identically) since we don't track inter-chunk context — that matches
the HF default for the Gemma / T5 family.
"""
struct MetaspacePreTokenizer <: PreTokenizer
    replacement::String
    prepend_scheme::Symbol     # :always | :first | :never
end

"""
    BertPreTokenizer

BERT's `BasicTokenizer`-style splitter: break on whitespace, and
isolate every punctuation character into its own token. Matches the
HF `BertPreTokenizer` (no parameters in the JSON schema; behavior is
fixed).
"""
struct BertPreTokenizer <: PreTokenizer end

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

function apply_pre(p::MetaspacePreTokenizer, ts::Vector{String})
    out = String[]
    for t in ts
        s = replace(t, ' ' => p.replacement)
        if p.prepend_scheme !== :never && !startswith(s, p.replacement)
            s = p.replacement * s
        end
        push!(out, s)
    end
    return out
end

# BERT's punctuation set: every ASCII punctuation plus any Unicode
# punctuation. Matches the HF reference; ASCII non-letter non-digit
# non-space characters in the ranges 33..47, 58..64, 91..96, 123..126
# always count as punctuation.
@inline _is_bert_punct(c::Char) =
    let cp = UInt32(c)
        (33 <= cp <= 47) ||
            (58 <= cp <= 64) ||
            (91 <= cp <= 96) ||
            (123 <= cp <= 126) ||
            ispunct(c)
    end

function apply_pre(::BertPreTokenizer, ts::Vector{String})
    out = String[]
    for t in ts
        buf = IOBuffer()
        for c in t
            if isspace(c)
                s = String(take!(buf))
                isempty(s) || push!(out, s)
            elseif _is_bert_punct(c)
                s = String(take!(buf))
                isempty(s) || push!(out, s)
                push!(out, string(c))
            else
                print(buf, c)
            end
        end
        s = String(take!(buf))
        isempty(s) || push!(out, s)
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

"""
    ReplaceDecoder

Literal string replacement, e.g. `▁` → " " for SentencePiece decoders.
"""
struct ReplaceDecoder <: Decoder
    pattern::String
    content::String
end

apply_dec(d::ReplaceDecoder, s::AbstractString) = replace(s, d.pattern => d.content)

"""
    ByteFallbackDecoder

Collapses runs of SentencePiece byte-fallback tokens (`<0xHH>`) back into
the bytes they represent. Tokens that aren't byte-fallback markers pass
through verbatim.
"""
struct ByteFallbackDecoder <: Decoder end

const _BYTE_FALLBACK_REGEX = r"<0x([0-9A-Fa-f]{2})>"

function apply_dec(::ByteFallbackDecoder, s::AbstractString)
    # Coalesce a contiguous run of `<0xHH>` markers and decode the
    # collected bytes as UTF-8; isolated markers (the common case) just
    # decode that single byte.
    out = IOBuffer()
    buf = UInt8[]
    function flush_buf!()
        isempty(buf) && return
        try
            print(out, String(copy(buf)))
        catch
            # Invalid UTF-8 run — fall back to emitting bytes verbatim
            # as Latin-1 so we don't lose data.
            for b in buf
                write(out, b)
            end
        end
        empty!(buf)
    end

    i = firstindex(s)
    n = lastindex(s)
    while i <= n
        m = match(_BYTE_FALLBACK_REGEX, s, i)
        if m === nothing || m.offset != i
            flush_buf!()
            if m === nothing
                print(out, SubString(s, i))
                break
            end
            print(out, SubString(s, i, prevind(s, m.offset)))
            i = m.offset
        end
        # Match begins exactly at i — consume one byte.
        push!(buf, parse(UInt8, m.captures[1]::AbstractString; base=16))
        i = m.offset + ncodeunits(m.match)
    end
    flush_buf!()
    return String(take!(out))
end

"""
    FuseDecoder

No-op concatenation marker. HF's tokenizer pipeline uses `Fuse` to
indicate that prior steps' outputs should be joined into a single
string; in Allspark every decoder already operates on the joined
string, so this is the identity.
"""
struct FuseDecoder <: Decoder end

apply_dec(::FuseDecoder, s::AbstractString) = String(s)

"""
    WordPieceDecoder

Reverse a WordPiece tokenization. Tokens come into `apply_dec` already
joined with single spaces (the `decode` driver concatenates everything
through the model's `id_to_token`). This decoder removes the
`continuing_subword_prefix` along with the space that precedes it (so
`"un ##able"` becomes `"unable"`), and when `cleanup=true` collapses
the space-before-punctuation that BERT's pre-tokenizer introduced.
"""
struct WordPieceDecoder <: Decoder
    prefix::String
    cleanup::Bool
end

function apply_dec(d::WordPieceDecoder, s::AbstractString)
    str = replace(String(s), " " * d.prefix => "")
    if d.cleanup
        # Drop the space that BertPreTokenizer left before ASCII punctuation,
        # plus the textbook contractions ("don ' t" → "don't").
        str = replace(str, r" ([.,!?;:)\]}])" => s"\1")
        str = replace(str, r"([([{]) " => s"\1")
        str = replace(str, " ' " => "'")
        str = replace(str, " n't" => "n't")
    end
    return str
end

"""
    StripDecoder

Removes `start` characters from the beginning and `stop` from the end.
Used by SentencePiece decoders to undo the `▁` prepended by the
Metaspace pretokenizer.
"""
struct StripDecoder <: Decoder
    content::String
    start::Int
    stop::Int
end

function apply_dec(d::StripDecoder, s::AbstractString)
    str = String(s)
    pat = d.content
    pat_bytes = ncodeunits(pat)
    pat_bytes > 0 || return str

    # Left strip: pat's end always falls on a char boundary, so SubString
    # from `pat_bytes + 1` is safe.
    for _ in 1:d.start
        startswith(str, pat) || break
        str = String(SubString(str, pat_bytes + 1))
    end

    # Right strip: walk to the char boundary just before where pat starts.
    # `prevind` skips over continuation bytes for multi-byte chars.
    for _ in 1:d.stop
        endswith(str, pat) || break
        pat_start = ncodeunits(str) - pat_bytes + 1
        cut_end = prevind(str, pat_start)
        str = cut_end < 1 ? "" : String(SubString(str, 1, cut_end))
    end

    return str
end
