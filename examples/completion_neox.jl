#!/usr/bin/env julia
# HuggingFaceTransformers.jl text-completion REPL backed by GPT-NeoX (Pythia).
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

using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode, decode
using HuggingFaceTransformers.Generation: generate
using JSON3

const DEFAULT_MODEL = "EleutherAI/pythia-70m"

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

    println("Parsing tokenizer...")
    tokenizer = load_tokenizer(snapshot_dir)
    eos_ids = load_eos_ids(snapshot_dir)

    println("Loading model...")
    lm = load(snapshot_dir)
    println(
        "Loaded $(nameof(typeof(lm))): $(lm.config.num_hidden_layers) layers, " *
        "$(lm.config.hidden_size) hidden",
    )

    println()
    println(
        "HuggingFaceTransformers.jl completion REPL (GPT-NeoX / Pythia). Ctrl-D to exit."
    )
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
