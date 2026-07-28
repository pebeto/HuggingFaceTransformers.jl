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
