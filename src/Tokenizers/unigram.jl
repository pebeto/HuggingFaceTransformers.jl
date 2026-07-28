# SentencePiece Unigram model: best-segmentation Viterbi over a vocab of
# (token, log-probability) pairs, optionally with byte-fallback for
# out-of-vocabulary characters (Gemma, T5, etc.).

"""
    TrieNode

Byte-keyed trie node. `token_id` is non-`nothing` when the path from the
root to this node spells a vocab token (storing that token's 0-indexed
HF ID).
"""
mutable struct TrieNode
    children::Dict{UInt8,TrieNode}
    token_id::Union{Nothing,Int}
end

TrieNode() = TrieNode(Dict{UInt8,TrieNode}(), nothing)

function _trie_insert!(root::TrieNode, token::AbstractString, token_id::Integer)
    node = root
    for byte in codeunits(token)
        child = get(node.children, byte, nothing)
        if child === nothing
            child = TrieNode()
            node.children[byte] = child
        end
        node = child
    end
    node.token_id = Int(token_id)
    return root
end

"""
    UnigramModel

SentencePiece-Unigram tokenizer. `vocab` and `id_to_token` provide the
usual Dict-shaped views (so the added-token grafting in
`load_tokenizer` works the same way as for BPE). `scores` is indexed by
HF token ID (i.e. `scores[id + 1]` gives the log-probability of the
token with HF ID `id`). `byte_fallback` enables Gemma/T5-style fallback
where any out-of-vocabulary byte gets encoded as a `<0xHH>` token, with
`byte_to_id[byte + 1]` giving the HF ID of that fallback token.

The `trie` is built once at construction and indexes only the original
vocab, not added tokens, which are matched upstream in
`_split_on_added`.
"""
mutable struct UnigramModel
    vocab::Dict{String,Int}
    id_to_token::Dict{Int,String}
    scores::Vector{Float32}
    unk_id::Union{Nothing,Int}
    unk_token::Union{Nothing,String}
    byte_fallback::Bool
    byte_to_id::Vector{Union{Nothing,Int}}   # length 256; nothing if missing
    trie::TrieNode
end

"""
    UnigramModel(vocab::Vector{Tuple{String,Float32}}; unk_id=nothing, byte_fallback=false)

Construct a `UnigramModel`. `vocab` is the HF vocab list (assigning
0-indexed IDs in order). When `byte_fallback` is true, the constructor
scans for `<0xHH>` tokens and records their IDs into the `byte_to_id`
lookup; missing byte tokens leave `nothing` in the slot.
"""
function UnigramModel(
    vocab::Vector{Tuple{String,Float32}};
    unk_id::Union{Nothing,Integer}=nothing,
    byte_fallback::Bool=false,
)
    vocab_dict = Dict{String,Int}()
    id_to_token = Dict{Int,String}()
    scores = Vector{Float32}(undef, length(vocab))
    for (i, (tok, score)) in enumerate(vocab)
        id = i - 1
        vocab_dict[tok] = id
        id_to_token[id] = tok
        scores[i] = score
    end

    unk_id_int = unk_id === nothing ? nothing : Int(unk_id)
    unk_tok = unk_id_int === nothing ? nothing : get(id_to_token, unk_id_int, nothing)

    byte_to_id = Vector{Union{Nothing,Int}}(nothing, 256)
    if byte_fallback
        for b in 0x00:0xff
            tok = "<0x" * uppercase(string(b; base=16, pad=2)) * ">"
            id = get(vocab_dict, tok, nothing)
            byte_to_id[Int(b) + 1] = id
        end
    end

    trie = TrieNode()
    for (i, (tok, _)) in enumerate(vocab)
        _trie_insert!(trie, tok, i - 1)
    end

    return UnigramModel(
        vocab_dict,
        id_to_token,
        scores,
        unk_id_int,
        unk_tok,
        byte_fallback,
        byte_to_id,
        trie,
    )
end

# Find every vocab token that prefix-matches `text` starting at byte
# offset `start`. Returns `(token_id, length_in_bytes)` pairs.
function _trie_matches(root::TrieNode, text::AbstractString, start::Int)
    matches = Tuple{Int,Int}[]
    node = root
    n = sizeof(text)
    j = start
    while j <= n
        byte = codeunit(text, j)
        child = get(node.children, byte, nothing)
        child === nothing && break
        node = child
        j += 1
        if node.token_id !== nothing
            push!(matches, (node.token_id, j - start))
        end
    end
    return matches
end

"""
    encode_word(model::UnigramModel, text) -> Vector{Int}

Best-segmentation (Viterbi) encode `text` into 0-indexed token IDs.
Walks every byte position, gathers vocab-token matches via the trie,
and (when `byte_fallback` is true) adds a single-byte fallback
transition using the `<0xHH>` token. Picks the highest-log-probability
backtrack path.

If no path reaches the end of `text`, which is pathological and needs
neither matches nor byte fallback, this returns the `unk_id` token alone (or
an empty vector when no `unk` is configured).
"""
function encode_word(model::UnigramModel, text::AbstractString)
    isempty(text) && return Int[]
    n = sizeof(text)

    best_score = fill(typemin(Float32), n + 1)
    best_score[1] = 0.0f0
    best_back = Vector{Tuple{Int,Int}}(undef, n + 1)

    @inbounds for i in 1:n
        best_score[i] > typemin(Float32) || continue

        # Vocab transitions.
        for (tok_id, len) in _trie_matches(model.trie, text, i)
            j = i + len
            new_score = best_score[i] + model.scores[tok_id + 1]
            if new_score > best_score[j]
                best_score[j] = new_score
                best_back[j] = (i, tok_id)
            end
        end

        # Byte fallback: single-byte transition.
        if model.byte_fallback
            byte = codeunit(text, i)
            byte_tok_id = model.byte_to_id[Int(byte) + 1]
            if byte_tok_id !== nothing
                j = i + 1
                new_score = best_score[i] + model.scores[byte_tok_id + 1]
                if new_score > best_score[j]
                    best_score[j] = new_score
                    best_back[j] = (i, byte_tok_id)
                end
            end
        end
    end

    if best_score[n + 1] == typemin(Float32)
        return model.unk_id === nothing ? Int[] : [model.unk_id]
    end

    ids = Int[]
    pos = n + 1
    while pos > 1
        prev, tok_id = best_back[pos]
        push!(ids, tok_id)
        pos = prev
    end
    reverse!(ids)
    return ids
end
