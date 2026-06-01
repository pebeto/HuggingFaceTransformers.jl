#!/usr/bin/env julia
# Allspark.jl REPL chat against a Qwen2 / Qwen2.5 model.
#
# Usage:
#   julia --project=. examples/repl_chat_qwen.jl
#   julia --project=. examples/repl_chat_qwen.jl Qwen/Qwen2.5-1.5B-Instruct
#
# Authentication: most Qwen2.5 models are open weights, no HF token needed.
#
# Resource cost (fp32):
#   Qwen2.5-0.5B-Instruct: ~2 GB download, ~2.5 GB RAM   (tied embeddings)
#   Qwen2.5-1.5B-Instruct: ~6 GB download, ~7 GB RAM     (tied embeddings)
#   Qwen2.5-3B-Instruct:   ~12 GB download, ~14 GB RAM   (untied)
#   Qwen2.5-7B-Instruct:   ~30 GB download, ~32 GB RAM   (untied)

using Allspark
using Allspark.HFHub: snapshot_download
using Allspark.Tokenizers: load_tokenizer, encode, decode
using Allspark.Models: load_weights, QwenForCausalLM, QwenConfig, load_state_dict!
using Allspark.Generation: generate, ChatTemplate
using JSON3

const DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Fallback ChatML-style template used if the bundled chat_template trips
# on Jinja features we don't support. Matches Qwen2.5's <|im_start|> /
# <|im_end|> framing for plain chat (no tool calls).
const FALLBACK_QWEN_TEMPLATE = raw"""
{%- for message in messages -%}
{%- if loop.first and messages[0]['role'] != 'system' -%}
<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
"""

function load_qwen_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    head_dim = if haskey(raw, :head_dim)
        Int(raw.head_dim)
    else
        Int(raw.hidden_size) ÷ Int(raw.num_attention_heads)
    end

    use_sw = Bool(get(raw, :use_sliding_window, false))
    sliding_window =
        if use_sw && haskey(raw, :sliding_window) && raw.sliding_window !== nothing
            Int(raw.sliding_window)
        else
            nothing
        end

    return QwenConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        num_key_value_heads=Int(get(raw, :num_key_value_heads, raw.num_attention_heads)),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        rope_theta=Float64(get(raw, :rope_theta, 1_000_000.0)),
        rms_norm_eps=Float64(get(raw, :rms_norm_eps, 1.0e-6)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, false)),
        sliding_window=sliding_window,
    )
end

function load_chat_template(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "tokenizer_config.json"), String))

    template_source = get(raw, :chat_template, nothing)
    if template_source !== nothing
        try
            return ChatTemplate(String(template_source))
        catch err
            @warn """Bundled chat_template uses Jinja features Allspark
                  doesn't support yet. Falling back to a plain ChatML
                  template; tool calls and other advanced features
                  won't work.""" err
        end
    end

    return ChatTemplate(FALLBACK_QWEN_TEMPLATE)
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
    cfg = load_qwen_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    template = load_chat_template(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println(
        "Materializing model ($(cfg.num_hidden_layers) layers, " *
        "$(cfg.hidden_size) hidden, " *
        "tied=$(cfg.tie_word_embeddings))...",
    )
    lm = QwenForCausalLM(cfg)

    println("Loading weights...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    messages = Dict{String,String}[]
    println()
    println("Allspark.jl REPL chat (Qwen). Ctrl-D to exit, /reset to clear history.")
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
