"""
    Allspark.Layers

The minimal layer kit used to assemble transformer models: `RMSNorm`,
`RoPE` (with Llama-3 scaling hooks), bias-less `Linear`, SiLU-gated MLP,
grouped-query attention with KV-cache support. All layers are
`Flux.@layer`-annotated and `Functors.fmap`-friendly.
"""
module Layers

end # module Layers
