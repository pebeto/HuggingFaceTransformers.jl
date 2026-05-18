struct BPEModel
    vocab::Dict{String,Int}
    id_to_token::Dict{Int,String}
    merge_ranks::Dict{Tuple{String,String},Int}
    unk_token::Union{Nothing,String}
end

function BPEModel(
    vocab::Dict{String,Int},
    merges::Vector{Tuple{String,String}},
    unk_token::Union{Nothing,AbstractString}=nothing,
)
    id_to_token = Dict{Int,String}(id => tok for (tok, id) in vocab)
    merge_ranks =
        Dict{Tuple{String,String},Int}(p => i - 1 for (i, p) in enumerate(merges))
    unk = unk_token === nothing ? nothing : String(unk_token)
    return BPEModel(vocab, id_to_token, merge_ranks, unk)
end

"""
    bpe_encode_word(model, word) -> Vector{String}

Apply BPE to `word`, merging all occurrences of the lowest-rank bigram in
each pass until no further merge is possible. Mirrors the reference
implementation in `transformers.GPT2Tokenizer.bpe`.
"""
function bpe_encode_word(model::BPEModel, word::AbstractString)
    isempty(word) && return String[]
    tokens = String[string(c) for c in word]
    while length(tokens) >= 2
        best_rank = typemax(Int)
        best_pair = ("", "")
        for i in 1:(length(tokens) - 1)
            pair = (tokens[i], tokens[i + 1])
            rank = get(model.merge_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end
        best_rank == typemax(Int) && break

        merged = String[]
        i = 1
        n = length(tokens)
        while i <= n
            if i < n && tokens[i] == best_pair[1] && tokens[i + 1] == best_pair[2]
                push!(merged, best_pair[1] * best_pair[2])
                i += 2
            else
                push!(merged, tokens[i])
                i += 1
            end
        end
        tokens = merged
    end
    return tokens
end

"""
    token_ids(model, tokens) -> Vector{Int}

Resolve a vector of BPE token strings to integer IDs (0-indexed). Unknown
tokens fall back to `model.unk_token` if set; otherwise a `KeyError` is
thrown.
"""
function token_ids(model::BPEModel, tokens::Vector{String})
    out = Int[]
    for t in tokens
        id = get(model.vocab, t, nothing)
        if id === nothing
            model.unk_token === nothing && throw(KeyError(t))
            id = model.vocab[model.unk_token]
        end
        push!(out, id)
    end
    return out
end
