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

    println("Parsing tokenizer...")
    tokenizer = load_tokenizer(snapshot_dir)
    template, bos_token = load_chat_template(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println("Loading model...")
    lm = load(snapshot_dir)
    println(
        "Loaded $(nameof(typeof(lm))): $(lm.config.num_hidden_layers) layers, " *
        "$(lm.config.hidden_size) hidden",
    )

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
        # The chat template already emits its special tokens as text, so the
        # post-processor must not add them a second time.
        prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)

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
