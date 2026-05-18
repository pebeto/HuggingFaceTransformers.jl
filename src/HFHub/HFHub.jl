"""
    Allspark.HFHub

Client for the HuggingFace Hub: resolves `org/repo[@revision]`, downloads
`config.json`, `tokenizer.json`, and `model.safetensors` (or sharded
equivalents), and caches them to `~/.cache/huggingface/hub` in the same
layout the Python `huggingface_hub` library uses, so downloads are shared
across Python and Julia tooling.
"""
module HFHub

end # module HFHub
