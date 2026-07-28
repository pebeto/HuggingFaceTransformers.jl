#!/usr/bin/env julia
# HuggingFaceTransformers.jl REPL chat against a Mixtral (sparse Mixture-of-Experts) model.
#
# Usage:
#   julia --project=. examples/repl_chat_mixtral.jl
#   julia --project=. examples/repl_chat_mixtral.jl mistralai/Mixtral-8x22B-Instruct-v0.1
#
# Mixtral checkpoints are large in fp32 because all experts are
# materialized even though only top-K of them run per token at
# inference time:
#   Mixtral-8x7B-Instruct-v0.1:   ~90 GB download, ~95 GB peak RAM
#   Mixtral-8x22B-Instruct-v0.1: ~280 GB download, ~290 GB peak RAM
#
# The forward pass is correctness-first: each expert is called once
# with its assigned slice of tokens (the standard scatter-gather), not
# fused. Fusing the expert matmuls is an open optimization.

using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode, decode
using HuggingFaceTransformers.Models:
    load_weights, MixtralForCausalLM, MixtralConfig, load_state_dict!
using HuggingFaceTransformers.Generation: generate, ChatTemplate
using JSON3

const DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Fallback Mistral-style template (Mixtral uses the same [INST] markers).
const FALLBACK_MIXTRAL_TEMPLATE = raw"""
{{- bos_token -}}
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
 [INST] {{ message['content'] }} [/INST]
{%- elif message['role'] == 'assistant' -%}
 {{ message['content'] }}{{ eos_token }}
{%- endif -%}
{%- endfor -%}
"""

function load_mixtral_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    head_dim = if haskey(raw, :head_dim)
        Int(raw.head_dim)
    else
        Int(raw.hidden_size) ÷ Int(raw.num_attention_heads)
    end

    sliding_window = if haskey(raw, :sliding_window) && raw.sliding_window !== nothing
        Int(raw.sliding_window)
    else
        nothing
    end

    return MixtralConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        num_key_value_heads=Int(get(raw, :num_key_value_heads, raw.num_attention_heads)),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        num_local_experts=Int(raw.num_local_experts),
        num_experts_per_tok=Int(raw.num_experts_per_tok),
        rope_theta=Float64(get(raw, :rope_theta, 1_000_000.0)),
        rms_norm_eps=Float64(get(raw, :rms_norm_eps, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, false)),
        sliding_window=sliding_window,
    )
end

function load_chat_template(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "tokenizer_config.json"), String))

    bos = String(get(raw, :bos_token, ""))
    eos = String(get(raw, :eos_token, ""))
    template_source = get(raw, :chat_template, nothing)

    if template_source !== nothing
        try
            return ChatTemplate(String(template_source)), bos, eos
        catch err
            @warn """Bundled chat_template uses Jinja features HuggingFaceTransformers
                  doesn't support yet. Falling back to a plain Mistral
                  [INST] template; tool calls won't work.""" err
        end
    end

    return ChatTemplate(FALLBACK_MIXTRAL_TEMPLATE), bos, eos
end

function load_eos_ids(snapshot_dir::AbstractString)
    gen_path = joinpath(snapshot_dir, "generation_config.json")
    if isfile(gen_path)
        gen = JSON3.read(read(gen_path, String))
        if haskey(gen, :eos_token_id)
            ids = gen.eos_token_id
            return ids isa Integer ? [Int(ids)] : Int[Int(x) for x in ids]
        end
    end
    tk_path = joinpath(snapshot_dir, "tokenizer_config.json")
    tk = JSON3.read(read(tk_path, String))
    if haskey(tk, :eos_token_id)
        ids = tk.eos_token_id
        return ids isa Integer ? [Int(ids)] : Int[Int(x) for x in ids]
    end
    return Int[]
end

function main(repo_id::AbstractString=DEFAULT_MODEL)
    println("Resolving $(repo_id)...")
    snapshot_dir = snapshot_download(repo_id; verbose=true)
    println("Snapshot at $(snapshot_dir)")

    println("Parsing config and tokenizer...")
    cfg = load_mixtral_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    template, bos_token, eos_token = load_chat_template(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println(
        "Materializing model ($(cfg.num_hidden_layers) layers, " *
        "$(cfg.hidden_size) hidden, " *
        "$(cfg.num_local_experts) experts × top-$(cfg.num_experts_per_tok))...",
    )
    lm = MixtralForCausalLM(cfg)

    println("Loading weights (8 experts per layer — this takes a while)...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    messages = Dict{String,String}[]
    println()
    println(
        "HuggingFaceTransformers.jl REPL chat (Mixtral). Ctrl-D to exit, /reset to clear history.",
    )
    println()

    while true
        print("> ")
        eof(stdin) && break
        line = strip(readline(stdin))
        isempty(line) && continue
        if line == "/reset"
            empty!(messages)
            println("(history cleared)")
            continue
        end

        push!(messages, Dict("role" => "user", "content" => String(line)))
        prompt = template(
            messages; add_generation_prompt=true, bos_token=bos_token, eos_token=eos_token
        )
        prompt_ids = encode(tokenizer, prompt)

        out_ids = generate(
            lm,
            prompt_ids;
            max_new_tokens=256,
            eos_token_id=isempty(eos_ids) ? nothing : eos_ids,
        )

        new_ids = out_ids[(length(prompt_ids) + 1):end]
        while !isempty(new_ids) && new_ids[end] in eos_ids
            pop!(new_ids)
        end
        reply = decode(tokenizer, new_ids)

        println(reply)
        println()
        push!(messages, Dict("role" => "assistant", "content" => reply))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    repo = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_MODEL
    main(repo)
end
