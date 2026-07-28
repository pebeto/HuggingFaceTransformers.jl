"""
    HuggingFaceTransformers

Load and run HuggingFace transformer checkpoints in Julia. Weights are read
from safetensors, tokenizers from `tokenizer.json`, and the forward passes are
Flux layers.

The package is split into six submodules, each exported from the top level:

- [`HFHub`](@ref HuggingFaceTransformers.HFHub): resolve and download
  repositories from the Hub into the Python-compatible cache layout.
- [`Tokenizers`](@ref HuggingFaceTransformers.Tokenizers): BPE, Unigram, and
  WordPiece tokenizers driven by `tokenizer.json`.
- [`Layers`](@ref HuggingFaceTransformers.Layers): the layer kit that models are
  assembled from, including attention and the KV-cache.
- [`Models`](@ref HuggingFaceTransformers.Models): concrete architectures and
  the state-dict tables that map HF parameter names onto them.
- [`Generation`](@ref HuggingFaceTransformers.Generation): sampling, chat
  templating, and speculative decoding.
- [`GGUF`](@ref HuggingFaceTransformers.GGUF): a reader for llama.cpp GGUF
  files.
"""
module HuggingFaceTransformers

include("HFHub/HFHub.jl")
include("GGUF/GGUF.jl")
include("Tokenizers/Tokenizers.jl")
include("Layers/Layers.jl")
include("Models/Models.jl")
include("Generation/Generation.jl")

using .HFHub
using .GGUF
using .Tokenizers
using .Layers
using .Models
using .Generation

export HFHub, GGUF, Tokenizers, Layers, Models, Generation

end # module HuggingFaceTransformers
