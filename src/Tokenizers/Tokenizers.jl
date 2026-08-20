"""
    HuggingFaceTransformers.Tokenizers

Generic tokenizers driven by HuggingFace's `tokenizer.json` schema, covering
byte-level BPE, SentencePiece Unigram, and WordPiece. There are no per-model
tokenizer subclasses: [`load_tokenizer`](@ref) is the single public entry point
for the vast majority of models.
"""
module Tokenizers

using JSON3

export Tokenizer, AddedToken, load_tokenizer, load_wordpiece_from_vocab_txt, encode, decode

include("byte_level.jl")
include("bpe.jl")
include("unigram.jl")
include("wordpiece.jl")
include("normalizer.jl")
include("pretokenizer.jl")

"""
    AddedToken

An entry from a tokenizer's `added_tokens` table: the literal `content` string,
the `id` it maps to, and whether it is a `special` token. Added tokens are
matched before the model's own segmentation runs, and
[`decode`](@ref) can skip the special ones.
"""
struct AddedToken
    id::Int
    content::String
    special::Bool
end

"""
    Tokenizer{M}

A loaded HuggingFace tokenizer. `M` is the segmentation model: `BPEModel`
(GPT-2, Llama-3, Mistral, Qwen, Phi-3, NeoX, RoBERTa), `UnigramModel`
(SentencePiece checkpoints such as Gemma), or `WordPieceModel` (BERT and its
descendants).
"""
struct Tokenizer{M}
    model::M
    normalizer::Normalizer
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

# `type` was added to `tokenizer.json` after the first checkpoints shipped, and
# files without it are still on the Hub (`gpt2`'s among them). HF infers the model
# from which fields are present, so do the same: a `merges` list means BPE, a
# vocab of `[token, score]` pairs means Unigram, and a plain vocab object without
# merges means WordPiece.
function _infer_model_type(m::JSON3.Object)
    haskey(m, :merges) && return "BPE"
    vocab = get(m, :vocab, nothing)
    vocab isa JSON3.Array && return "Unigram"
    vocab isa JSON3.Object && return "WordPiece"
    throw(
        ArgumentError(
            "tokenizer.json `model` has no `type` and no recognizable vocab to " *
            "infer one from; keys present: $(collect(keys(m)))",
        ),
    )
end

function _parse_model(m::JSON3.Object)
    typ = haskey(m, :type) ? String(m[:type]::AbstractString) : _infer_model_type(m)
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
            length(arr) == 2 ||
                throw(ArgumentError("Unigram vocab entry must be [token, score]: $(arr)"))
            push!(vocab, (String(arr[1]::AbstractString), Float32(arr[2]::Real)))
        end
        unk_id_raw = get(m, :unk_id, nothing)
        unk_id = unk_id_raw isa Integer ? Int(unk_id_raw) : nothing
        byte_fallback = Bool(get(m, :byte_fallback, false)::Bool)
        return UnigramModel(vocab; unk_id=unk_id, byte_fallback=byte_fallback)
    elseif typ == "WordPiece"
        vocab_raw = m[:vocab]::JSON3.Object
        vocab = Dict{String,Int}()
        for (k, v) in pairs(vocab_raw)
            vocab[String(k)] = Int(v::Integer)
        end
        unk = String(get(m, :unk_token, "[UNK]")::AbstractString)
        prefix = String(get(m, :continuing_subword_prefix, "##")::AbstractString)
        max_chars = Int(get(m, :max_input_chars_per_word, 100)::Integer)
        return WordPieceModel(
            vocab;
            unk_token=unk,
            continuing_subword_prefix=prefix,
            max_input_chars_per_word=max_chars,
        )
    else
        throw(
            ArgumentError(
                "unsupported tokenizer model: $(typ); supported: BPE, Unigram, WordPiece"
            ),
        )
    end
end

function _parse_normalizer(n::Union{Nothing,JSON3.Object})
    n === nothing && return IdentityNormalizer()
    typ = String(n[:type]::AbstractString)
    if typ == "BertNormalizer"
        return BertNormalizer(;
            clean_text=Bool(get(n, :clean_text, true)::Bool),
            handle_chinese_chars=Bool(get(n, :handle_chinese_chars, true)::Bool),
            strip_accents=Bool(get(n, :strip_accents, true)::Bool),
            lowercase=Bool(get(n, :lowercase, true)::Bool),
        )
    elseif typ == "Sequence"
        # Tokenizers JSON allows a stack of normalizers; for our purposes a
        # single Bert-shaped normalizer is enough — pick the first
        # `BertNormalizer` in the sequence if present, else identity.
        children = n[:normalizers]::JSON3.Array
        for child in children
            obj = child::JSON3.Object
            if String(obj[:type]::AbstractString) == "BertNormalizer"
                return _parse_normalizer(obj)
            end
        end
        return IdentityNormalizer()
    else
        # Unknown normalizers (NFC, NFD, Lowercase, Replace, …) fall back
        # to identity. Inputs that need them won't get byte-perfect parity
        # but the model still runs.
        return IdentityNormalizer()
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
            if s == "always"
                :always
            elseif s == "first"
                :first
            elseif s == "never"
                :never
            else
                throw(ArgumentError("unknown Metaspace prepend_scheme: $(s)"))
            end
        else
            Bool(get(p, :add_prefix_space, true)::Bool) ? :always : :never
        end
        return MetaspacePreTokenizer(replacement, scheme)
    elseif typ == "BertPreTokenizer"
        return BertPreTokenizer()
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
        haskey(pat, :String) ||
            throw(ArgumentError("Replace decoder only supports literal-string patterns"))
        return ReplaceDecoder(
            String(pat[:String]::AbstractString), String(d[:content]::AbstractString)
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
    elseif typ == "WordPiece"
        return WordPieceDecoder(
            String(get(d, :prefix, "##")::AbstractString),
            Bool(get(d, :cleanup, true)::Bool),
        )
    else
        throw(ArgumentError("unsupported decoder type: $(typ)"))
    end
end

"""
    load_tokenizer(path) -> Tokenizer

Load a HuggingFace `tokenizer.json`. `path` may be the JSON file itself or
a directory containing one.

Three model types are supported:

- BPE with ByteLevel pre-tokenization, for the GPT-2, Llama-3, Qwen2, and
  RoBERTa family.
- Unigram with Metaspace pre-tokenization, for SentencePiece checkpoints such
  as Gemma and T5.
- WordPiece with `BertNormalizer`, for BERT and its descendants.

Checkpoints that ship only a `vocab.txt` load through
[`load_wordpiece_from_vocab_txt`](@ref) instead.
"""
function load_tokenizer(path::AbstractString)
    file = isdir(path) ? joinpath(path, "tokenizer.json") : path
    isfile(file) || throw(ArgumentError("tokenizer.json not found at $(file)"))

    parsed = JSON3.read(read(file, String))::JSON3.Object

    model = _parse_model(parsed[:model]::JSON3.Object)

    norm_raw = get(parsed, :normalizer, nothing)
    norm = norm_raw isa JSON3.Object ? _parse_normalizer(norm_raw) : IdentityNormalizer()

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

    return Tokenizer(model, norm, pre, dec, added, added_lookup, id_lookup)
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
            normalized = apply_norm(tk.normalizer, chunk)
            for pt in apply_pre(tk.pre_tokenizer, [normalized])
                append!(ids, encode_word(tk.model, pt))
            end
        end
    end
    return ids
end

"""
    load_wordpiece_from_vocab_txt(path; lowercase=true, strip_accents=true, unk_token="[UNK]") -> Tokenizer

Legacy loader for BERT checkpoints that ship a `vocab.txt` instead of a
`tokenizer.json`. Each non-empty line of the file becomes a vocab entry
keyed on its 0-indexed line number. Constructs a `WordPieceModel` +
`BertNormalizer` (lowercase + accent strip by default) +
`BertPreTokenizer` + `WordPieceDecoder` matching the BERT-base convention.
`path` may be the `vocab.txt` file itself or a directory containing one.
"""
function load_wordpiece_from_vocab_txt(
    path::AbstractString;
    lowercase::Bool=true,
    strip_accents::Bool=true,
    unk_token::AbstractString="[UNK]",
)
    file = isdir(path) ? joinpath(path, "vocab.txt") : path
    isfile(file) || throw(ArgumentError("vocab.txt not found at $(file)"))

    vocab = Dict{String,Int}()
    open(file) do io
        for (i, line) in enumerate(eachline(io))
            tok = strip(line, '\n')
            vocab[String(tok)] = i - 1
        end
    end

    model = WordPieceModel(vocab; unk_token=unk_token)
    normalizer = BertNormalizer(;
        clean_text=true,
        handle_chinese_chars=true,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )
    pre = BertPreTokenizer()
    decoder = WordPieceDecoder("##", true)

    return Tokenizer(
        model,
        normalizer,
        pre,
        decoder,
        AddedToken[],
        Dict{String,Int}(),
        Dict{Int,String}(),
    )
end

# Separator inserted between regular tokens before the decoder sees the
# concatenated stream. ByteLevel decoders expect tokens jammed together
# (the `Ġ` prefix already marks word boundaries); WordPiece decoders
# expect them space-separated so they can strip the `" ##"` join.
_token_separator(::Decoder) = ""
_token_separator(::WordPieceDecoder) = " "
function _token_separator(d::SequenceDecoder)
    return isempty(d.steps) ? "" : _token_separator(d.steps[1])
end

"""
    decode(tk, ids; skip_special_tokens=false) -> String

Reverse [`encode`](@ref). Regular tokens are accumulated and joined
with the decoder-specific separator before being passed through the
decoder pipeline; added tokens are rendered verbatim. With
`skip_special_tokens=true`, added tokens marked `special` are omitted.
"""
function decode(
    tk::Tokenizer, ids::AbstractVector{<:Integer}; skip_special_tokens::Bool=false
)
    special_ids = Set{Int}(at.id for at in tk.added_tokens if at.special)
    out = IOBuffer()
    buffer = String[]
    sep = _token_separator(tk.decoder)

    function flush!()
        isempty(buffer) && return nothing
        joined = join(buffer, sep)
        print(out, apply_dec(tk.decoder, joined))
        return empty!(buffer)
    end

    for id in ids
        if haskey(tk.id_lookup, id)
            flush!()
            (skip_special_tokens && id in special_ids) && continue
            print(out, tk.id_lookup[id])
        else
            tok = get(tk.model.id_to_token, id, nothing)
            tok === nothing && throw(KeyError(id))
            push!(buffer, tok)
        end
    end
    flush!()
    return String(take!(out))
end

end # module Tokenizers
