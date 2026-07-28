#!/usr/bin/env julia
# HuggingFaceTransformers.jl REPL chat against a Gemma2 model.
#
# Usage:
#   julia --project=. examples/repl_chat_gemma.jl
#   julia --project=. examples/repl_chat_gemma.jl google/gemma-2-9b-it
#
# Authentication: Gemma weights are gated. Accept Google's terms on the
# model page first, then:
#   export HF_TOKEN=hf_...        # or `huggingface-cli login`
#
# Resource cost (fp32):
#   gemma-2-2b-it:   ~10 GB download, ~12 GB RAM
#   gemma-2-9b-it:   ~37 GB download, ~40 GB RAM   (server-class)
#   gemma-2-27b-it: ~108 GB download, ~120 GB RAM  (workstation only)

using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode, decode
using HuggingFaceTransformers.Models:
    load_weights, GemmaForCausalLM, GemmaConfig, load_state_dict!
using HuggingFaceTransformers.Generation: generate, ChatTemplate
using JSON3

const DEFAULT_MODEL = "google/gemma-2-2b-it"

# Fallback Gemma chat template (used when the bundled chat_template
# exercises Jinja features outside our scope). Gemma uses <start_of_turn>
# / <end_of_turn> markers with "user" and "model" roles.
const FALLBACK_GEMMA_TEMPLATE = raw"""
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'assistant' -%}
<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<start_of_turn>model
{% endif -%}
"""

function load_gemma_config(snapshot_dir::AbstractString)
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

    attn_softcap =
        if haskey(raw, :attn_logit_softcapping) && raw.attn_logit_softcapping !== nothing
            Float64(raw.attn_logit_softcapping)
        else
            nothing
        end

    final_softcap =
        if haskey(raw, :final_logit_softcapping) && raw.final_logit_softcapping !== nothing
            Float64(raw.final_logit_softcapping)
        else
            nothing
        end

    qpas = if haskey(raw, :query_pre_attn_scalar) && raw.query_pre_attn_scalar !== nothing
        Int(raw.query_pre_attn_scalar)
    else
        nothing
    end

    return GemmaConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        num_key_value_heads=Int(get(raw, :num_key_value_heads, raw.num_attention_heads)),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        rope_theta=Float64(get(raw, :rope_theta, 10_000.0)),
        rms_norm_eps=Float64(get(raw, :rms_norm_eps, 1.0e-6)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, true)),
        sliding_window=sliding_window,
        attn_logit_softcapping=attn_softcap,
        final_logit_softcapping=final_softcap,
        query_pre_attn_scalar=qpas,
    )
end

function load_chat_template(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "tokenizer_config.json"), String))

    template_source = get(raw, :chat_template, nothing)
    if template_source !== nothing
        try
            return ChatTemplate(String(template_source))
        catch err
            @warn """Bundled chat_template uses Jinja features HuggingFaceTransformers
                  doesn't support yet. Falling back to a plain Gemma
                  <start_of_turn> template; tool calls and other advanced
                  features won't work.""" err
        end
    end

    return ChatTemplate(FALLBACK_GEMMA_TEMPLATE)
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
    cfg = load_gemma_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    template = load_chat_template(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    softcap_note =
        if cfg.attn_logit_softcapping !== nothing && cfg.final_logit_softcapping !== nothing
            ", softcap=$(cfg.attn_logit_softcapping)/$(cfg.final_logit_softcapping)"
        else
            ""
        end
    println(
        "Materializing model ($(cfg.num_hidden_layers) layers, " *
        "$(cfg.hidden_size) hidden$(softcap_note))...",
    )
    lm = GemmaForCausalLM(cfg)

    println("Loading weights...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    messages = Dict{String,String}[]
    println()
    println(
        "HuggingFaceTransformers.jl REPL chat (Gemma2). Ctrl-D to exit, /reset to clear history.",
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
        prompt = template(messages; add_generation_prompt=true)
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
