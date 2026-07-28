# Internals

Documented helpers that are not exported. They are here because the public
docstrings link to them, and because knowing they exist helps when you are
adding a model. Treat them as unstable: they can change without a breaking
release.

```@meta
CurrentModule = HuggingFaceTransformers.Models
```

## Models

The shared state-dict table that the Llama-family loaders delegate to.

```@docs
_decoder_state_dict_map
_resolve
_apply_transform
```

```@meta
CurrentModule = HuggingFaceTransformers.Tokenizers
```

## Tokenizers

The byte-level mapping that ByteLevel BPE uses to make arbitrary bytes
representable as characters.

```@docs
bytes_to_string
string_to_bytes
```
