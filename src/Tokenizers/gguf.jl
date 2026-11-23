# llama.cpp's token types, `tokenizer.ggml.token_type`.
const _GGUF_NORMAL, _GGUF_UNKNOWN, _GGUF_CONTROL, _GGUF_USER_DEFINED = 1, 2, 3, 4

# The split regexes behind `tokenizer.ggml.pre`, as the original checkpoints'
# `tokenizer.json` spell them. Llama 3 groups digits in threes, Qwen2 splits them
# one by one.
const _LLAMA3_SPLIT = raw"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
const _QWEN2_SPLIT = raw"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

function _gguf_byte_level_pre(pre::AbstractString)
    pre in ("llama-bpe", "llama3") && return SequencePreTokenizer(
        PreTokenizer[SplitPreTokenizer(Regex(_LLAMA3_SPLIT), :isolated), ByteLevelPreTokenizer(false, false)],
    )
    pre == "qwen2" && return SequencePreTokenizer(
        PreTokenizer[SplitPreTokenizer(Regex(_QWEN2_SPLIT), :isolated), ByteLevelPreTokenizer(false, false)],
    )
    pre == "smollm" && return SequencePreTokenizer(
        PreTokenizer[SplitPreTokenizer(r"\p{N}", :isolated), ByteLevelPreTokenizer(false, true)],
    )
    pre in ("gpt-2", "default") && return ByteLevelPreTokenizer(false, true)
    throw(
        ArgumentError(
            "GGUF pre-tokenizer `$(pre)` is not supported; supported are llama-bpe, " *
            "qwen2, smollm, gpt-2 and default",
        ),
    )
end

# SentencePiece BPE merges whichever adjacent pair forms the highest-scoring
# piece. GGUF keeps the scores but no merge list, so derive one: every split of a
# normal piece into two normal pieces, ranked by the whole piece's score, then by
# its id (SentencePiece orders its vocabulary by score), then by the parts' scores.
function _merges_from_scores(tokens, scores, types)
    normal = Dict{String,Int}(tokens[i] => i for i in eachindex(tokens) if types[i] == _GGUF_NORMAL)
    ranked = Tuple{Float32,Int,Float32,Float32,Int,String,String}[]
    for (piece, i) in normal
        length(piece) < 2 && continue
        for (k, cut) in enumerate(Iterators.drop(eachindex(piece), 1))
            left, right = piece[1:prevind(piece, cut)], piece[cut:end]
            l, r = get(normal, left, 0), get(normal, right, 0)
            (l > 0 && r > 0) || continue
            push!(ranked, (-scores[i], i, -scores[l], -scores[r], k, left, right))
        end
    end
    sort!(ranked)
    return Tuple{String,String}[(l, r) for (_, _, _, _, _, l, r) in ranked]
end

function _gguf_tokenizer(path::AbstractString)
    meta = GGUF._open_gguf(path).metadata
    key(name, default=nothing) = get(meta, "tokenizer.ggml.$(name)", default)
    kind = key("model")
    isnothing(kind) && throw(ArgumentError("$(path) carries no tokenizer"))
    tokens = String.(key("tokens"))
    types = Int.(something(key("token_type"), fill(_GGUF_NORMAL, length(tokens))))
    id(name) = (v = key("$(name)_token_id"); isnothing(v) ? nothing : Int(v))
    bos, eos = id("bos"), id("eos")

    if kind == "gpt2"
        vocab = Dict(t => i - 1 for (i, t) in enumerate(tokens))
        merges = Tuple{String,String}[Tuple(split(m, ' '; limit=2)) for m in key("merges")]
        model = BPEModel(vocab, merges)
        pre_name = String(key("pre", "default"))
        pre = _gguf_byte_level_pre(pre_name)
        norm = pre_name == "qwen2" ? UnicodeNormalizer(:NFC) : IdentityNormalizer()
        dec = ByteLevelDecoder()
        # llama.cpp adds a BOS by default for Llama 3's pre-tokenizer only.
        add_bos = Bool(key("add_bos_token", pre_name in ("llama-bpe", "llama3")))
    elseif kind == "llama"
        vocab = Dict(t => i - 1 for (i, t) in enumerate(tokens))
        unk = id("unknown")
        model = BPEModel(
            vocab, _merges_from_scores(tokens, Float32.(key("scores")), types),
            isnothing(unk) ? nothing : tokens[unk + 1]; byte_fallback=true,
        )
        # llama.cpp prefixes a space to every run of text, also after a control
        # token mid-prompt, as HF's legacy Llama tokenizer does.
        prefix = Bool(key("add_space_prefix", true))
        replace_spaces = ReplaceNormalizer(" ", "▁")
        norm = prefix ? SequenceNormalizer(Normalizer[PrependNormalizer("▁"), replace_spaces]) : replace_spaces
        pre = IdentityPreTokenizer()
        steps = Decoder[ReplaceDecoder("▁", " "), ByteFallbackDecoder(), FuseDecoder()]
        prefix && push!(steps, StripDecoder(" ", 1, 0))
        dec = SequenceDecoder(steps)
        add_bos = Bool(key("add_bos_token", true))
    else
        throw(ArgumentError("GGUF tokenizer model `$(kind)` is not supported; supported are gpt2 and llama"))
    end

    added = AddedToken[
        AddedToken(i - 1, tokens[i], types[i] != _GGUF_USER_DEFINED)
        for i in eachindex(tokens) if types[i] in (_GGUF_UNKNOWN, _GGUF_CONTROL, _GGUF_USER_DEFINED)
    ]
    prefix_ids = add_bos && !isnothing(bos) ? [bos] : Int[]
    suffix_ids = Bool(key("add_eos_token", false)) && !isnothing(eos) ? [eos] : Int[]
    return _assemble_tokenizer(model, norm, pre, dec, added, PostProcessor(prefix_ids, suffix_ids))
end
