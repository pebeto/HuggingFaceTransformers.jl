#!/usr/bin/env julia
# HuggingFaceTransformers.jl masked-token completion REPL backed by RoBERTa.
#
# Usage:
#   julia --project=. examples/mask_fill_roberta.jl
#   julia --project=. examples/mask_fill_roberta.jl FacebookAI/roberta-large
#
# Type a sentence containing `<mask>`; the model returns the top-K
# tokens it predicts for that position. This is the natural way to
# exercise an encoder-only MaskedLM checkpoint — RoBERTa has no
# autoregressive generation.
#
# Resource cost (fp32):
#   FacebookAI/roberta-base:  ~0.5 GB download, ~0.6 GB RAM
#   FacebookAI/roberta-large: ~1.5 GB download, ~1.7 GB RAM
#
# BERT proper isn't in the example set yet. WordPiece tokenization is
# supported, so adding it is mostly a matter of writing the example.

using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode, decode
using HuggingFaceTransformers.Models:
    load_weights, BertForMaskedLM, BertConfig, load_state_dict!
using JSON3

const DEFAULT_MODEL = "FacebookAI/roberta-base"
const TOP_K = 5

function load_roberta_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    return BertConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        max_position_embeddings=Int(raw.max_position_embeddings) - 2,
        type_vocab_size=Int(get(raw, :type_vocab_size, 1)),
        layer_norm_eps=Float64(get(raw, :layer_norm_eps, 1.0e-5)),
        pad_token_id=Int(get(raw, :pad_token_id, 1)),
        position_embedding_offset=2,
        hf_prefix="roberta",
        head_prefix="lm_head",
        tie_word_embeddings=true,
    )
end

function load_mask_token_id(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "tokenizer_config.json"), String))
    haskey(raw, :mask_token) || error("no mask_token in tokenizer_config.json")
    # The mask token id usually lives in special_tokens_map / added_tokens.
    # Easier: re-encode the mask token's string form via the tokenizer.
    return String(raw.mask_token)
end

function main(repo_id::AbstractString=DEFAULT_MODEL)
    println("Resolving $(repo_id)...")
    snapshot_dir = snapshot_download(repo_id; verbose=true)
    println("Snapshot at $(snapshot_dir)")

    println("Parsing config and tokenizer...")
    cfg = load_roberta_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    mask_token = load_mask_token_id(snapshot_dir)

    # Look up the integer ID for the mask token string.
    mask_ids = encode(tokenizer, mask_token)
    length(mask_ids) == 1 ||
        error("mask token $(mask_token) didn't tokenize to a single ID; got $(mask_ids)")
    mask_id = mask_ids[1]

    println(
        "Materializing model ($(cfg.num_hidden_layers) layers, " *
        "$(cfg.hidden_size) hidden)...",
    )
    lm = BertForMaskedLM(cfg)

    println("Loading weights...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    println()
    println("HuggingFaceTransformers.jl mask-fill REPL (RoBERTa). Ctrl-D to exit.")
    println("Type a sentence containing `$(mask_token)`; I'll suggest top-$(TOP_K) tokens.")
    println()

    while true
        print("> ")
        eof(stdin) && break
        prompt = readline(stdin)
        isempty(prompt) && continue
        if !occursin(mask_token, prompt)
            println("(prompt has no $(mask_token); add one and retry)")
            continue
        end

        ids = encode(tokenizer, prompt)
        # Locate the mask position. With byte-level BPE, the mask should
        # encode as a single token; scan the ID list for it.
        mask_pos = findfirst(==(mask_id), ids)
        if isnothing(mask_pos)
            println("(couldn't find $(mask_token) in encoded IDs; tokenization edge case)")
            continue
        end

        input_mat = reshape(ids, :, 1)
        logits = lm(input_mat)
        scores = collect(logits[:, mask_pos, 1])
        topk = sortperm(scores; rev=true)[1:TOP_K]
        println("Top-$(TOP_K) predictions for $(mask_token):")
        for (rank, idx) in enumerate(topk)
            # idx is 1-indexed Julia; the token ID is idx - 1.
            tok = decode(tokenizer, [idx - 1])
            println("  $(rank). $(repr(tok))  (logit $(round(scores[idx]; digits=3)))")
        end
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    repo = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_MODEL
    main(repo)
end
