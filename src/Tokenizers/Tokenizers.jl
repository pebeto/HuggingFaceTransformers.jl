"""
    Allspark.Tokenizers

Generic tokenizers driven by HuggingFace's `tokenizer.json` schema
(byte-level BPE first; SentencePiece and WordPiece later). No per-model
tokenizer subclasses — `load_tokenizer("org/repo")` should be the only
public entry point for the vast majority of models.
"""
module Tokenizers

using JSON3

export Tokenizer, AddedToken, load_tokenizer, encode, decode

include("byte_level.jl")
include("bpe.jl")
include("unigram.jl")
include("pretokenizer.jl")

struct AddedToken
    id::Int
    content::String
    special::Bool
end

"""
    Tokenizer{M}

A loaded HuggingFace tokenizer. `M` is the model type — currently
either `BPEModel` (GPT-2 / Llama-3 / Mistral / Qwen / Phi-3 / NeoX /
RoBERTa) or `UnigramModel` (SentencePiece — Gemma, T5).
"""
struct Tokenizer{M}
    model::M
    pre_tokenizer::PreTokenizer
    decoder::Decoder
    added_tokens::Vector{AddedToken}
    added_token_lookup::Dict{String,Int}
    id_lookup::Dict{Int,String}
end

function _parse_merges(raw::AbstractVector)
    out = Tuple{String,String}[]
    for entry in raw
        if entry isa AbstractString
            i = findfirst(' ', entry)
            i === nothing && throw(ArgumentError("malformed BPE merge: $(entry)"))
            push!(out, (entry[1:prevind(entry, i)], entry[nextind(entry, i):end]))
        elseif entry isa AbstractVector && length(entry) == 2
            a, b = entry[1], entry[2]
            (a isa AbstractString && b isa AbstractString) ||
                throw(ArgumentError("BPE merge entries must be strings: $(entry)"))
            push!(out, (String(a), String(b)))
        else
            throw(ArgumentError("unrecognized BPE merge entry: $(entry)"))
        end
    end
    return out
end

function _parse_model(m::JSON3.Object)
    typ = String(m[:type]::AbstractString)
    if typ == "BPE"
        vocab_raw = m[:vocab]::JSON3.Object
        vocab = Dict{String,Int}()
        for (k, v) in pairs(vocab_raw)
            vocab[String(k)] = Int(v::Integer)
        end
        merges = _parse_merges(m[:merges]::JSON3.Array)
        unk_raw = get(m, :unk_token, nothing)
        unk = unk_raw isa AbstractString ? String(unk_raw) : nothing
        return BPEModel(vocab, merges, unk)
    elseif typ == "Unigram"
        vocab_raw = m[:vocab]::JSON3.Array
        vocab = Tuple{String,Float32}[]
        for entry in vocab_raw
            arr = entry::JSON3.Array
            length(arr) == 2 || throw(
                ArgumentError("Unigram vocab entry must be [token, score]: $(arr)"),
            )
            push!(vocab, (String(arr[1]::AbstractString), Float32(arr[2]::Real)))
        end
        unk_id_raw = get(m, :unk_id, nothing)
        unk_id = unk_id_raw isa Integer ? Int(unk_id_raw) : nothing
        byte_fallback = Bool(get(m, :byte_fallback, false)::Bool)
        return UnigramModel(vocab; unk_id=unk_id, byte_fallback=byte_fallback)
    else
        throw(
            ArgumentError(
                "unsupported tokenizer model: $(typ); supported: BPE, Unigram"
            ),
        )
    end
end

function _parse_pre_tokenizer(p::JSON3.Object)
    typ = String(p[:type]::AbstractString)
    if typ == "Sequence"
        children = p[:pretokenizers]::JSON3.Array
        steps = PreTokenizer[]
        for child in children
            push!(steps, _parse_pre_tokenizer(child::JSON3.Object))
        end
        return SequencePreTokenizer(steps)
    elseif typ == "ByteLevel"
        return ByteLevelPreTokenizer(
            Bool(get(p, :add_prefix_space, false)::Bool),
            Bool(get(p, :use_regex, true)::Bool),
        )
    elseif typ == "Split"
        pat = p[:pattern]::JSON3.Object
        haskey(pat, :Regex) ||
            throw(ArgumentError("Split pretokenizer requires a Regex pattern: $(pat)"))
        regex_str = String(pat[:Regex]::AbstractString)
        behavior_str = String(p[:behavior]::AbstractString)
        behavior = if behavior_str == "Isolated"
            :isolated
        elseif behavior_str == "Removed"
            :removed
        else
            throw(
                ArgumentError(
                    "Split pretokenizer with unsupported behavior: $(behavior_str)"
                ),
            )
        end
        return SplitPreTokenizer(Regex(regex_str), behavior)
    elseif typ == "Metaspace"
        replacement = String(get(p, :replacement, "▁")::AbstractString)
        scheme_raw = get(p, :prepend_scheme, "always")
        # HF tokenizers historically also used the bool flag `add_prefix_space`;
        # honor either form.
        scheme = if scheme_raw isa AbstractString
            s = String(scheme_raw)
            s == "always" ? :always : s == "first" ? :first : s == "never" ? :never :
            throw(ArgumentError("unknown Metaspace prepend_scheme: $(s)"))
        else
            Bool(get(p, :add_prefix_space, true)::Bool) ? :always : :never
        end
        return MetaspacePreTokenizer(replacement, scheme)
    else
        throw(ArgumentError("unsupported pre_tokenizer type: $(typ)"))
    end
end

function _parse_decoder(d::JSON3.Object)
    typ = String(d[:type]::AbstractString)
    if typ == "ByteLevel"
        return ByteLevelDecoder()
    elseif typ == "Sequence"
        children = d[:decoders]::JSON3.Array
        steps = Decoder[]
        for child in children
            push!(steps, _parse_decoder(child::JSON3.Object))
        end
        return SequenceDecoder(steps)
    elseif typ == "Replace"
        pat = d[:pattern]::JSON3.Object
        haskey(pat, :String) || throw(
            ArgumentError("Replace decoder only supports literal-string patterns"),
        )
        return ReplaceDecoder(
            String(pat[:String]::AbstractString),
            String(d[:content]::AbstractString),
        )
    elseif typ == "ByteFallback"
        return ByteFallbackDecoder()
    elseif typ == "Fuse"
        return FuseDecoder()
    elseif typ == "Strip"
        return StripDecoder(
            String(d[:content]::AbstractString),
            Int(get(d, :start, 0)::Integer),
            Int(get(d, :stop, 0)::Integer),
        )
    elseif typ == "Metaspace"
        # Some HF tokenizer JSONs also list Metaspace as a decoder; treat it
        # as Replace(replacement → " ").
        replacement = String(get(d, :replacement, "▁")::AbstractString)
        return ReplaceDecoder(replacement, " ")
    else
        throw(ArgumentError("unsupported decoder type: $(typ)"))
    end
end

"""
    load_tokenizer(path) -> Tokenizer

Load a HuggingFace `tokenizer.json`. `path` may be the JSON file itself or
a directory containing one.

Supports BPE models with ByteLevel pre-tokenization (the GPT-2 / Llama-3 /
Qwen2 / RoBERTa family) and Unigram models with Metaspace pre-tokenization
(SentencePiece — Gemma, T5, …). WordPiece is not yet implemented; classic
BERT checkpoints still need it.
"""
function load_tokenizer(path::AbstractString)
    file = isdir(path) ? joinpath(path, "tokenizer.json") : path
    isfile(file) || throw(ArgumentError("tokenizer.json not found at $(file)"))

    parsed = JSON3.read(read(file, String))::JSON3.Object

    model = _parse_model(parsed[:model]::JSON3.Object)

    pre_raw = get(parsed, :pre_tokenizer, nothing)
    pre = pre_raw isa JSON3.Object ? _parse_pre_tokenizer(pre_raw) : IdentityPreTokenizer()

    dec_raw = get(parsed, :decoder, nothing)
    dec = dec_raw isa JSON3.Object ? _parse_decoder(dec_raw) : ByteLevelDecoder()

    added = AddedToken[]
    added_raw = get(parsed, :added_tokens, nothing)
    if added_raw isa AbstractVector
        for at in added_raw
            obj = at::JSON3.Object
            push!(
                added,
                AddedToken(
                    Int(obj[:id]::Integer),
                    String(obj[:content]::AbstractString),
                    Bool(get(obj, :special, false)::Bool),
                ),
            )
        end
    end

    added_lookup = Dict{String,Int}()
    id_lookup = Dict{Int,String}()
    for at in added
        added_lookup[at.content] = at.id
        id_lookup[at.id] = at.content
        if !haskey(model.vocab, at.content)
            model.vocab[at.content] = at.id
            model.id_to_token[at.id] = at.content
        end
    end

    return Tokenizer(model, pre, dec, added, added_lookup, id_lookup)
end

function _split_on_added(tk::Tokenizer, text::AbstractString)
    isempty(tk.added_tokens) &&
        return Tuple{String,Union{Nothing,Int}}[(String(text), nothing)]
    sorted = sort(collect(tk.added_tokens); by=at -> ncodeunits(at.content), rev=true)

    chunks = Tuple{String,Union{Nothing,Int}}[]
    pos = 1
    n = ncodeunits(text)
    while pos <= n
        best_start = n + 1
        best_end = 0
        best_at::Union{Nothing,AddedToken} = nothing
        for at in sorted
            r = findnext(at.content, text, pos)
            if r !== nothing && first(r) < best_start
                best_start = first(r)
                best_end = last(r)
                best_at = at
            end
        end
        if best_at === nothing
            push!(chunks, (String(SubString(text, pos, n)), nothing))
            break
        end
        if best_start > pos
            push!(
                chunks, (String(SubString(text, pos, prevind(text, best_start))), nothing)
            )
        end
        push!(chunks, (best_at.content, best_at.id))
        pos = nextind(text, best_end)
    end
    return chunks
end

"""
    encode(tk, text) -> Vector{Int}

Tokenize `text` to a vector of 0-indexed token IDs (matching HF
convention). Added tokens (e.g. `<|endoftext|>`) appearing verbatim in
`text` are emitted as their dedicated IDs; everything else is run through
the pre-tokenizer and BPE model.
"""
function encode(tk::Tokenizer, text::AbstractString)
    ids = Int[]
    for (chunk, special_id) in _split_on_added(tk, text)
        if special_id !== nothing
            push!(ids, special_id)
        else
            isempty(chunk) && continue
            for pt in apply_pre(tk.pre_tokenizer, [chunk])
                append!(ids, encode_word(tk.model, pt))
            end
        end
    end
    return ids
end

"""
    decode(tk, ids; skip_special_tokens=false) -> String

Reverse [`encode`](@ref). Consecutive BPE tokens are concatenated and
byte-decoded together; added tokens are rendered verbatim. With
`skip_special_tokens=true`, added tokens marked `special` are omitted.
"""
function decode(
    tk::Tokenizer, ids::AbstractVector{<:Integer}; skip_special_tokens::Bool=false
)
    special_ids = Set{Int}(at.id for at in tk.added_tokens if at.special)
    out = IOBuffer()
    buffer = IOBuffer()

    function flush!()
        s = String(take!(buffer))
        isempty(s) || print(out, apply_dec(tk.decoder, s))
    end

    for id in ids
        if haskey(tk.id_lookup, id)
            flush!()
            (skip_special_tokens && id in special_ids) && continue
            print(out, tk.id_lookup[id])
        else
            tok = get(tk.model.id_to_token, id, nothing)
            tok === nothing && throw(KeyError(id))
            print(buffer, tok)
        end
    end
    flush!()
    return String(take!(out))
end

end # module Tokenizers
