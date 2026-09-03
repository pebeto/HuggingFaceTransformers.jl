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
using JSON3

const DEFAULT_MODEL = "FacebookAI/roberta-base"
const TOP_K = 5

# Where the mask token is declared varies by checkpoint. `roberta-base` ships a
# `tokenizer_config.json` holding nothing but `model_max_length`, so fall back to
# the added-tokens table in `tokenizer.json`, which every fast tokenizer carries.
function load_mask_token(snapshot_dir::AbstractString)
    config_path = joinpath(snapshot_dir, "tokenizer_config.json")
    if isfile(config_path)
        raw = JSON3.read(read(config_path, String))
        haskey(raw, :mask_token) && return String(raw.mask_token)
    end

    tokenizer_json = JSON3.read(read(joinpath(snapshot_dir, "tokenizer.json"), String))
    for added in get(tokenizer_json, :added_tokens, ())
        content = String(added.content)
        content in ("<mask>", "[MASK]") && return content
    end
    return error("no mask token in tokenizer_config.json or tokenizer.json")
end

function main(repo_id::AbstractString=DEFAULT_MODEL)
    println("Resolving $(repo_id)...")
    snapshot_dir = snapshot_download(repo_id; verbose=true)
    println("Snapshot at $(snapshot_dir)")

    println("Parsing tokenizer...")
    tokenizer = load_tokenizer(snapshot_dir)
    mask_token = load_mask_token(snapshot_dir)

    # Look up the integer ID for the mask token string.
    mask_ids = encode(tokenizer, mask_token)
    length(mask_ids) == 1 ||
        error("mask token $(mask_token) didn't tokenize to a single ID; got $(mask_ids)")
    mask_id = mask_ids[1]

    println("Loading model...")
    lm = load(snapshot_dir)
    println(
        "Loaded $(nameof(typeof(lm))): $(lm.config.num_hidden_layers) layers, " *
        "$(lm.config.hidden_size) hidden",
    )

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
