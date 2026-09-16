# Normalization pipeline. Decoder-only models in the HuggingFaceTransformers set (Llama,
# Mistral, Qwen, …) tokenize raw bytes and don't need normalization, so
# `Tokenizer` defaulted to skipping this stage. BERT-family encoders need
# it: lowercasing for `*-uncased` checkpoints, optional NFD-based accent
# stripping, control-char cleanup.

using Unicode

abstract type Normalizer end

"""
    IdentityNormalizer

Pass text through unchanged. The default for every model family that
doesn't ship a normalizer in its `tokenizer.json`.
"""
struct IdentityNormalizer <: Normalizer end

apply_norm(::IdentityNormalizer, s::AbstractString) = String(s)

"""
    BertNormalizer

Mirrors HuggingFace's `BertNormalizer`. Each step is independently
toggleable:

- `clean_text`: replace ASCII control characters (other than `\\t`, `\\n`,
  `\\r`) and the BOM with a single space.
- `handle_chinese_chars`: pad every CJK ideograph with surrounding spaces
  so the pre-tokenizer treats each ideograph as its own token (BERT's
  Chinese segmentation convention).
- `strip_accents`: NFD-decompose and drop combining marks (Unicode
  category `Mn`).
- `lowercase`: Unicode-aware lowercasing.
"""
Base.@kwdef struct BertNormalizer <: Normalizer
    clean_text::Bool = true
    handle_chinese_chars::Bool = true
    strip_accents::Bool = true
    lowercase::Bool = true
end

# CJK ranges per the BERT reference implementation.
@inline _is_cjk_char(c::Char) =
    let cp = UInt32(c)
        (0x4E00 <= cp <= 0x9FFF) ||
            (0x3400 <= cp <= 0x4DBF) ||
            (0x20000 <= cp <= 0x2A6DF) ||
            (0x2A700 <= cp <= 0x2B73F) ||
            (0x2B740 <= cp <= 0x2B81F) ||
            (0x2B820 <= cp <= 0x2CEAF) ||
            (0xF900 <= cp <= 0xFAFF) ||
            (0x2F800 <= cp <= 0x2FA1F)
    end

@inline function _is_control_keep(c::Char)
    # Returns true for control chars to be REPLACED with a space.
    # \t \n \r are preserved.
    c == '\t' && return false
    c == '\n' && return false
    c == '\r' && return false
    cp = UInt32(c)
    cp == 0 && return true       # null
    cp == 0xFFFD && return true  # replacement char
    return iscntrl(c)
end

function apply_norm(n::BertNormalizer, s::AbstractString)
    str = String(s)

    if n.clean_text
        io = IOBuffer()
        for c in str
            if c == '﻿'
                continue                 # BOM is dropped, not spaced
            elseif _is_control_keep(c)
                print(io, ' ')
            else
                print(io, c)
            end
        end
        str = String(take!(io))
    end

    if n.handle_chinese_chars
        io = IOBuffer()
        for c in str
            if _is_cjk_char(c)
                print(io, ' ', c, ' ')
            else
                print(io, c)
            end
        end
        str = String(take!(io))
    end

    if n.strip_accents
        decomposed = Unicode.normalize(str, :NFD)
        # Strip combining marks (Unicode category Mn).
        str = replace(decomposed, r"\p{Mn}" => "")
    end

    if n.lowercase
        str = lowercase(str)
    end

    return str
end

"""
    PrependNormalizer

Prepend a fixed string to the input. SentencePiece checkpoints (Llama, Phi-3,
Gemma) prepend `▁` so the first word carries the same word-boundary marker as
every later one.
"""
struct PrependNormalizer <: Normalizer
    prepend::String
end

apply_norm(n::PrependNormalizer, s::AbstractString) = n.prepend * String(s)

"""
    ReplaceNormalizer

Literal string substitution. SentencePiece checkpoints pair this with
[`PrependNormalizer`](@ref) to turn spaces into `▁`, which is what makes a word
boundary visible to a BPE model that has no pre-tokenizer.
"""
struct ReplaceNormalizer <: Normalizer
    pattern::String
    content::String
end

apply_norm(n::ReplaceNormalizer, s::AbstractString) =
    replace(String(s), n.pattern => n.content)

"""
    SequenceNormalizer

Apply several normalizers in order, which is how `tokenizer.json` expresses a
pipeline such as prepend-then-replace.
"""
struct SequenceNormalizer <: Normalizer
    normalizers::Vector{Normalizer}
end

function apply_norm(n::SequenceNormalizer, s::AbstractString)
    out = String(s)
    for step in n.normalizers
        out = apply_norm(step, out)
    end
    return out
end

"""
    UnicodeNormalizer

One of the Unicode normalization forms (`NFC`, `NFD`, `NFKC`, `NFKD`).
"""
struct UnicodeNormalizer <: Normalizer
    form::Symbol
end

apply_norm(n::UnicodeNormalizer, s::AbstractString) =
    Unicode.normalize(String(s), n.form)

"""
    LowercaseNormalizer

Unicode-aware lowercasing, as a standalone step rather than a
[`BertNormalizer`](@ref) flag.
"""
struct LowercaseNormalizer <: Normalizer end

apply_norm(::LowercaseNormalizer, s::AbstractString) = lowercase(String(s))
