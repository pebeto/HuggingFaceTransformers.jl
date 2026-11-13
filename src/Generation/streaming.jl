"""
    TextStreamer(tokenizer, out=stdout; skip_prompt=false, skip_special_tokens=false)

Prints text as [`generate`](@ref) produces it, as `transformers.TextStreamer`
does. Pass it as `streamer`:

```julia
generate(lm, ids; max_new_tokens=128, streamer=TextStreamer(tokenizer; skip_prompt=true))
```

`out` is an `IO`, or a function called with each new piece of text. A piece is
released once its characters are complete, so a character split across tokens
(byte-level BPE, or SentencePiece byte fallback) arrives whole. Each piece is
decoded with a few tokens of context before it, which keeps the leading space a
SentencePiece decoder strips from the first token of a string. When generation
ends, an `IO` gets a final newline, as HF's streamer prints one.

Any callable works as a `streamer`: it receives the prompt ids once, then a
one-element vector per generated token, and returning `false` from a token call
stops generation after that token.
"""
mutable struct TextStreamer{O}
    tokenizer::Tokenizer
    out::O
    skip_prompt::Bool
    skip_special_tokens::Bool
    ids::Vector{Int}
    prefix::Int
    read::Int
    started::Bool
end

function TextStreamer(
    tokenizer::Tokenizer, out=stdout; skip_prompt::Bool=false, skip_special_tokens::Bool=false
)
    return TextStreamer(tokenizer, out, skip_prompt, skip_special_tokens, Int[], 0, 0, false)
end

_emit(s::TextStreamer{<:IO}, text) = (print(s.out, text); flush(s.out))
_emit(s::TextStreamer, text) = s.out(text)

# Tokens of context decoded ahead of a piece, which is what HF's streamers and
# vLLM's detokenizer use to recover the boundary between pieces.
const _STREAM_CONTEXT = 5

function (s::TextStreamer)(ids::AbstractVector{<:Integer})
    if !s.started
        s.started = true
        if s.skip_prompt
            append!(s.ids, ids)
            s.read = length(s.ids)
            s.prefix = max(s.read - _STREAM_CONTEXT, 0)
            return nothing
        end
    end
    append!(s.ids, ids)
    _release!(s, false)
    return nothing
end

function _release!(s::TextStreamer, final::Bool)
    done = decode(s.tokenizer, view(s.ids, (s.prefix + 1):s.read); s.skip_special_tokens)
    text = decode(s.tokenizer, view(s.ids, (s.prefix + 1):length(s.ids)); s.skip_special_tokens)
    ncodeunits(text) > ncodeunits(done) || return nothing
    final || !_ends_mid_character(text) || return nothing
    _emit(s, String(codeunits(text)[(ncodeunits(done) + 1):end]))
    s.prefix = max(length(s.ids) - _STREAM_CONTEXT, s.read)
    s.read = length(s.ids)
    return nothing
end

function _end_stream(s::TextStreamer)
    _release!(s, true)
    s.out isa IO && _emit(s, "\n")
    return nothing
end
_end_stream(_) = nothing

# A character split across tokens decodes to an unfinished UTF-8 sequence, from
# byte-level BPE and from SentencePiece byte fallback alike. Invalid bytes that no
# later token can complete are released as they are.
function _ends_mid_character(text::AbstractString)
    bytes = codeunits(text)
    n = length(bytes)
    for i in n:-1:max(1, n - 2)
        b = bytes[i]
        b & 0xc0 == 0x80 && continue
        need = b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 1
        return need > n - i + 1
    end
    return false
end

function _check_stop_strings(stop_strings, tokenizer)
    isempty(stop_strings) && return String[]
    isnothing(tokenizer) && throw(
        ArgumentError("stop_strings needs the tokenizer: pass `tokenizer=` to generate"),
    )
    any(isempty, stop_strings) && throw(ArgumentError("stop_strings cannot be empty"))
    return String[String(s) for s in stop_strings]
end

# HF's rule: stop when a stop string overlaps the last token, so one that only
# occurs further back does not stop generation again. The tail decoded here
# reaches back far enough to hold the longest stop string before the last token.
function _hits_stop_string(tokenizer, ids::AbstractVector{Int}, stop_strings, longest::Int)
    isempty(stop_strings) && return false
    n = length(ids)
    k = 1
    while true
        k = min(2k, n)
        before = decode(tokenizer, view(ids, (n - k + 1):(n - 1)))
        (ncodeunits(before) >= longest || k == n) || continue
        text = decode(tokenizer, view(ids, (n - k + 1):n))
        cut = ncodeunits(before)
        return any(stop_strings) do stop
            r = findlast(stop, text)
            !isnothing(r) && last(r) > cut
        end
    end
end
