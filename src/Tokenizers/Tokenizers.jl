"""
    Allspark.Tokenizers

Generic tokenizers driven by HuggingFace's `tokenizer.json` schema
(byte-level BPE first; SentencePiece and WordPiece later). No per-model
tokenizer subclasses — `load_tokenizer("org/repo")` should be the only
public entry point for the vast majority of models.
"""
module Tokenizers

end # module Tokenizers
