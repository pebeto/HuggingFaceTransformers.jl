# WordPiece tokenizer. Used by BERT, DistilBERT, and the
# `bert-base-*` family of checkpoints. Algorithm:
#
# Each pre-tokenized word is greedy-longest-match scanned against the
# vocab from the left. Non-initial matches are looked up with the
# `continuing_subword_prefix` (default `##`) prepended; if no prefix
# of the remaining text matches at any length, the word collapses to
# the configured unknown token.

"""
    WordPieceModel

WordPiece tokenizer model. `vocab` and `id_to_token` provide the
standard Dict-shaped views (mirroring `BPEModel` / `UnigramModel` so
the added-token grafting in `load_tokenizer` works the same way).

`unk_token` is the string emitted for unmatchable words.
`continuing_subword_prefix` is what's prepended to the second-and-later
piece of a word (default `##`). `max_input_chars_per_word` caps the
length of a single word; longer words fall back to `unk_token`.
"""
mutable struct WordPieceModel
    vocab::Dict{String,Int}
    id_to_token::Dict{Int,String}
    unk_token::String
    continuing_subword_prefix::String
    max_input_chars_per_word::Int
end

function WordPieceModel(
    vocab::Dict{String,Int};
    unk_token::AbstractString="[UNK]",
    continuing_subword_prefix::AbstractString="##",
    max_input_chars_per_word::Integer=100,
)
    id_to_token = Dict{Int,String}(id => tok for (tok, id) in vocab)
    return WordPieceModel(
        vocab,
        id_to_token,
        String(unk_token),
        String(continuing_subword_prefix),
        Int(max_input_chars_per_word),
    )
end

"""
    encode_word(model::WordPieceModel, word) -> Vector{Int}

Greedy longest-prefix-match encode. The first piece is looked up
verbatim; subsequent pieces get `continuing_subword_prefix` prepended
before lookup. If no prefix matches at some position, the whole word
collapses to `unk_token`.
"""
function encode_word(model::WordPieceModel, word::AbstractString)
    isempty(word) && return Int[]

    # Length cap is over character count, matching the HF reference.
    if length(word) > model.max_input_chars_per_word
        return [model.vocab[model.unk_token]]
    end

    # Work with character boundaries via `nextind`/`prevind`. Convert to
    # a Vector{Char} once so slicing is O(1).
    chars = collect(word)
    n = length(chars)

    pieces = Int[]
    start = 1
    while start <= n
        endi = n
        matched_id = nothing
        matched_end = 0
        while endi >= start
            substr = String(chars[start:endi])
            candidate = start > 1 ? model.continuing_subword_prefix * substr : substr
            id = get(model.vocab, candidate, nothing)
            if id !== nothing
                matched_id = id
                matched_end = endi
                break
            end
            endi -= 1
        end
        if matched_id === nothing
            return [model.vocab[model.unk_token]]
        end
        push!(pieces, matched_id)
        start = matched_end + 1
    end
    return pieces
end
