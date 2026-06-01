#!/usr/bin/env julia
# Allspark.jl text-completion REPL backed by GPT-2.
#
# Usage:
#   julia --project=. examples/completion_gpt2.jl
#   julia --project=. examples/completion_gpt2.jl gpt2-xl
#
# GPT-2 is a base language model, not chat-tuned — there are no role
# markers or system prompts. You type a prefix, the model continues it.
#
# Resource cost (fp32):
#   gpt2:        ~0.5 GB download, ~0.6 GB RAM   (124M params)
#   gpt2-medium: ~1.5 GB download, ~1.7 GB RAM   (355M)
#   gpt2-large:  ~3 GB download,   ~3.5 GB RAM   (774M)
#   gpt2-xl:     ~6 GB download,   ~7 GB RAM     (1558M)

using Allspark
using Allspark.HFHub: snapshot_download
using Allspark.Tokenizers: load_tokenizer, encode, decode
using Allspark.Models: load_weights, GPT2ForCausalLM, GPT2Config, load_state_dict!
using Allspark.Generation: generate
using JSON3

const DEFAULT_MODEL = "gpt2"

function load_gpt2_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    hidden = Int(raw.n_embd)
    return GPT2Config(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=hidden,
        intermediate_size=Int(
            get(raw, :n_inner, nothing) === nothing ? 4 * hidden : raw.n_inner
        ),
        num_hidden_layers=Int(raw.n_layer),
        num_attention_heads=Int(raw.n_head),
        max_position_embeddings=Int(raw.n_positions),
        layer_norm_eps=Float64(get(raw, :layer_norm_epsilon, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, true)),
    )
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
    return Int[]
end

function main(repo_id::AbstractString=DEFAULT_MODEL)
    println("Resolving $(repo_id)...")
    snapshot_dir = snapshot_download(repo_id; verbose=true)
    println("Snapshot at $(snapshot_dir)")

    println("Parsing config and tokenizer...")
    cfg = load_gpt2_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println(
        "Materializing model ($(cfg.num_hidden_layers) layers, " *
        "$(cfg.hidden_size) hidden)...",
    )
    lm = GPT2ForCausalLM(cfg)

    println("Loading weights (transposing Conv1D + slicing fused QKV)...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    println()
    println("Allspark.jl completion REPL (GPT-2). Ctrl-D to exit.")
    println("Type a prefix; the model continues it for up to 64 tokens.")
    println()

    while true
        print("> ")
        eof(stdin) && break
        prefix = readline(stdin)
        isempty(prefix) && continue

        prompt_ids = encode(tokenizer, prefix)
        out_ids = generate(
            lm,
            prompt_ids;
            max_new_tokens=64,
            eos_token_id=isempty(eos_ids) ? nothing : eos_ids,
        )
        completion = decode(tokenizer, out_ids[(length(prompt_ids) + 1):end])
        println(completion)
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    repo = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_MODEL
    main(repo)
end
