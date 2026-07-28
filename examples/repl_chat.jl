#!/usr/bin/env julia
# HuggingFaceTransformers.jl REPL chat: download a Llama, then talk to it.
#
# Usage:
#   julia --project=. examples/repl_chat.jl
#   julia --project=. examples/repl_chat.jl meta-llama/Llama-3.2-1B-Instruct
#
# Gated models (Llama-3.x in particular) need an HF access token:
#   export HF_TOKEN=hf_...
# or run `huggingface-cli login` once. See https://huggingface.co/docs/hub/security-tokens.
#
# First-run cost: a ~2.5 GB download for Llama-3.2-1B, cached to
# ~/.cache/huggingface/hub. Subsequent runs are instant.

using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode, decode
using HuggingFaceTransformers.Models:
    load_weights, LlamaForCausalLM, LlamaConfig, LlamaRopeScaling, load_state_dict!
using HuggingFaceTransformers.Generation: generate, ChatTemplate
using JSON3

const DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Fallback used when the model's bundled `chat_template` exercises Jinja
# features we don't implement (tool calls, date filters, tuple literals).
# Matches the structure of Llama-3.x's actual template for plain chat.
const FALLBACK_LLAMA3_TEMPLATE = raw"""
{%- for message in messages -%}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif -%}
"""

function load_llama_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    rope_scaling = nothing
    if haskey(raw, :rope_scaling) && raw.rope_scaling !== nothing
        rs = raw.rope_scaling
        rope_scaling = LlamaRopeScaling(;
            factor=Float64(rs.factor),
            low_freq_factor=Float64(rs.low_freq_factor),
            high_freq_factor=Float64(rs.high_freq_factor),
            original_max_position_embeddings=Int(rs.original_max_position_embeddings),
        )
    end

    head_dim = if haskey(raw, :head_dim)
        Int(raw.head_dim)
    else
        Int(raw.hidden_size) ÷ Int(raw.num_attention_heads)
    end

    return LlamaConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        num_key_value_heads=Int(get(raw, :num_key_value_heads, raw.num_attention_heads)),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        rope_theta=Float64(get(raw, :rope_theta, 500000.0)),
        rms_norm_eps=Float64(get(raw, :rms_norm_eps, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, false)),
        rope_scaling=rope_scaling,
    )
end

function load_chat_template(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "tokenizer_config.json"), String))

    template_source = get(raw, :chat_template, nothing)
    if template_source !== nothing
        try
            return ChatTemplate(String(template_source)), String(get(raw, :bos_token, ""))
        catch err
            @warn """Model's bundled chat_template uses Jinja features HuggingFaceTransformers
                  doesn't support yet. Falling back to a plain Llama-3
                  template; chat works, but tool calls and other advanced
                  template features won't.""" err
        end
    end

    return ChatTemplate(FALLBACK_LLAMA3_TEMPLATE), String(get(raw, :bos_token, ""))
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
    cfg = load_llama_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    template, bos_token = load_chat_template(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println(
        "Materializing model ($(cfg.num_hidden_layers) layers, " *
        "$(cfg.hidden_size) hidden)...",
    )
    lm = LlamaForCausalLM(cfg)

    println("Loading weights...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    messages = Dict{String,String}[]
    println()
    println(
        "HuggingFaceTransformers.jl REPL chat. Ctrl-D to exit, /reset to clear history."
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
        prompt = template(messages; add_generation_prompt=true, bos_token=bos_token)
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
