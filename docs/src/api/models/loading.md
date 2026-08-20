```@meta
CurrentModule = HuggingFaceTransformers.Models
```

# Loading machinery

```@docs
Models
```

The rest of this page covers the generic pieces every architecture shares:
reading a checkpoint, walking a state-dict table, and allocating caches.

```@docs
DecoderLayer
DecoderModel
TokenEmbedding
build_caches
forward_embeds
load_into!
load_state_dict!
load_weights
```

## Loading by architecture

[`HuggingFaceTransformers.load`](@ref) is the front door and downloads first.
These are the pieces underneath it, which work on a directory that is already
present.

```@docs
load_pretrained
read_config
config_from_json
AUTO_ARCHITECTURES
```
