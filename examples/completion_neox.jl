#!/usr/bin/env julia
# Allspark.jl text-completion REPL backed by GPT-NeoX (Pythia).
#
# Usage:
#   julia --project=. examples/completion_neox.jl
#   julia --project=. examples/completion_neox.jl EleutherAI/pythia-1b
#
# Like GPT-2, Pythia is a base language model — type a prefix, the
# model continues it. No chat tuning.
#
# Resource cost (fp32):
#   pythia-70m:   ~0.3 GB download, ~0.4 GB RAM
#   pythia-410m:  ~1.6 GB download, ~1.8 GB RAM
#   pythia-1b:    ~4 GB download,   ~4.5 GB RAM
#   pythia-6.9b:  ~28 GB download,  ~30 GB RAM
# The 12B sibling needs more headroom than most laptops have.

using Allspark
using Allspark.HFHub: snapshot_download
using Allspark.Tokenizers: load_tokenizer, encode, decode
using Allspark.Models: load_weights, NeoXForCausalLM, NeoXConfig, load_state_dict!
using Allspark.Generation: generate
using JSON3

const DEFAULT_MODEL = "EleutherAI/pythia-70m"

function load_neox_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))
    head_dim = if haskey(raw, :head_dim)
        Int(raw.head_dim)
    else
        Int(raw.hidden_size) ÷ Int(raw.num_attention_heads)
    end
    rotary_pct = Float64(get(raw, :rotary_pct, 0.25))

    return NeoXConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        head_dim=head_dim,
        max_position_embeddings=Int(raw.max_position_embeddings),
        rope_theta=Float64(get(raw, :rotary_emb_base, 10_000.0)),
        partial_rotary_factor=rotary_pct,
        layer_norm_eps=Float64(get(raw, :layer_norm_eps, 1.0e-5)),
        tie_word_embeddings=Bool(get(raw, :tie_word_embeddings, false)),
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
    cfg = load_neox_config(snapshot_dir)
    tokenizer = load_tokenizer(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println("Materializing model ($(cfg.num_hidden_layers) layers, " *
            "$(cfg.hidden_size) hidden, " *
            "rotary_pct=$(cfg.partial_rotary_factor))...")
    lm = NeoXForCausalLM(cfg)

    println("Loading weights (slicing per-head interleaved QKV)...")
    load_state_dict!(lm, load_weights(snapshot_dir))

    println()
    println("Allspark.jl completion REPL (GPT-NeoX / Pythia). Ctrl-D to exit.")
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
