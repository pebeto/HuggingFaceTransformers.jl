# The decoder families llama.cpp writes that map onto ours, with the HF
# architecture each one loads as.
const _GGUF_ARCHITECTURES = Dict(
    "llama" => "LlamaForCausalLM",
    "qwen2" => "Qwen2ForCausalLM",
    "qwen3" => "Qwen3ForCausalLM",
    "gemma2" => "Gemma2ForCausalLM",
)

const _GGUF_LAYER_NAMES = Dict(
    "attn_norm" => "input_layernorm",
    "attn_q" => "self_attn.q_proj",
    "attn_k" => "self_attn.k_proj",
    "attn_v" => "self_attn.v_proj",
    "attn_output" => "self_attn.o_proj",
    "attn_q_norm" => "self_attn.q_norm",
    "attn_k_norm" => "self_attn.k_norm",
    "ffn_norm" => "post_attention_layernorm",
    "ffn_gate" => "mlp.gate_proj",
    "ffn_up" => "mlp.up_proj",
    "ffn_down" => "mlp.down_proj",
)

# Gemma 2 norms both sides of attention and of the MLP.
const _GGUF_GEMMA2_LAYER_NAMES = merge(
    _GGUF_LAYER_NAMES,
    Dict(
        "post_attention_norm" => "post_attention_layernorm",
        "ffn_norm" => "pre_feedforward_layernorm",
        "post_ffw_norm" => "post_feedforward_layernorm",
    ),
)

function _hf_tensor_name(arch::AbstractString, name::AbstractString)
    name == "token_embd.weight" && return "model.embed_tokens.weight"
    name == "output_norm.weight" && return "model.norm.weight"
    name == "output.weight" && return "lm_head.weight"
    m = match(r"^blk\.(\d+)\.(\w+)\.(weight|bias)$", name)
    isnothing(m) && return nothing
    table = arch == "gemma2" ? _GGUF_GEMMA2_LAYER_NAMES : _GGUF_LAYER_NAMES
    hf = get(table, m[2], nothing)
    return isnothing(hf) ? nothing : "model.layers.$(m[1]).$(hf).$(m[3])"
end

"""
    GGUFWeights

A GGUF file's tensors under the names an HF checkpoint gives them, decoded to
`Float32` one at a time as the model loader asks for each. Built by
[`load_gguf_model`](@ref).

llama.cpp stores a few tensors differently from HF, and reading one undoes that:
the llama family's query and key rows are interleaved for its rotary layout, and
Gemma 2's norm weights have the 1 that HF adds at run time folded in.
"""
struct GGUFWeights <: AbstractDict{String,AbstractArray}
    index::GGUF._GGUFIndex
    names::Dict{String,String}
    arch::String
    heads::Int
    kv_heads::Int
end

Base.length(w::GGUFWeights) = length(w.names)
Base.haskey(w::GGUFWeights, key::AbstractString) = haskey(w.names, key)
Base.keys(w::GGUFWeights) = keys(w.names)
function Base.iterate(w::GGUFWeights, state...)
    next = iterate(w.names, state...)
    isnothing(next) && return nothing
    (key, _), s = next
    return key => w[key], s
end

function Base.getindex(w::GGUFWeights, key::AbstractString)
    haskey(w.names, key) || throw(KeyError(key))
    t = GGUF._tensor(w.index, w.names[key])
    if w.arch == "gemma2" && endswith(key, "norm.weight")
        t .-= 1
    end
    ndims(t) == 2 || return t
    if w.arch == "llama"
        endswith(key, "q_proj.weight") && (t = _unpermute_rotary(t, w.heads))
        endswith(key, "k_proj.weight") && (t = _unpermute_rotary(t, w.kv_heads))
    end
    # GGML's (in, out) is HF's (out, in). A lazy transpose, as the safetensors
    # loader returns, lets the loader copy it blockwise or take it back for free.
    return PermutedDimsArray(t, (2, 1))
end

# llama.cpp's converter reorders each head's query and key outputs from HF's
# split-halves rotary layout to interleaved pairs: HF output `a * half + b` of a
# head becomes output `2b + a`. This is the file's output for each HF output.
function _rotary_order(n_out::Integer, heads::Integer)
    head_dim = n_out ÷ heads
    half = head_dim ÷ 2
    order = Vector{Int}(undef, n_out)
    for h in 0:(heads - 1), a in 0:1, b in 0:(half - 1)
        order[h * head_dim + a * half + b + 1] = h * head_dim + 2b + a + 1
    end
    return order
end

# Undo the reordering on GGML's (in, out) layout.
_unpermute_rotary(w::AbstractMatrix, heads::Integer) = w[:, _rotary_order(size(w, 2), heads)]

# GGUF stores these floats as Float32. Their shortest decimal form is the value
# the HF config held: `1f-6` reads back as `1e-6`, not `9.99999997e-7`.
_gguf_float(x::Float32) = parse(Float64, string(x))
_gguf_float(x) = isnothing(x) ? nothing : Float64(x)

# The `config.json` an HF checkpoint of the same model would carry, from the
# `{arch}.*` metadata keys llama.cpp's converter writes.
function _gguf_config(idx::GGUF._GGUFIndex, arch::AbstractString)
    meta = idx.metadata
    get_meta(key, default=nothing) = get(meta, "$(arch).$(key)", default)
    function required(key)
        v = get_meta(key)
        isnothing(v) && throw(ArgumentError("GGUF metadata is missing `$(arch).$(key)`"))
        v isa AbstractVector && throw(
            ArgumentError("`$(arch).$(key)` varies per layer, which is not supported"),
        )
        return v
    end

    scaling = get_meta("rope.scaling.type", "none")
    scaling == "none" ||
        throw(ArgumentError("GGUF RoPE scaling `$(scaling)` is not supported yet"))
    haskey(idx.tensors, "rope_freqs.weight") && throw(
        ArgumentError(
            "this GGUF scales RoPE frequencies per dimension (Llama 3.1 and later), " *
            "which is not supported from GGUF yet; load the safetensors checkpoint",
        ),
    )

    heads = Int(required("attention.head_count"))
    hidden = Int(required("embedding_length"))
    raw = Dict{Symbol,Any}(
        :architectures => [_GGUF_ARCHITECTURES[arch]],
        :model_type => arch,
        :vocab_size => idx.tensors["token_embd.weight"].dims[2],
        :hidden_size => hidden,
        :intermediate_size => Int(required("feed_forward_length")),
        :num_hidden_layers => Int(required("block_count")),
        :num_attention_heads => heads,
        :num_key_value_heads => Int(get_meta("attention.head_count_kv", heads)),
        :max_position_embeddings => Int(required("context_length")),
        :rms_norm_eps => _gguf_float(required("attention.layer_norm_rms_epsilon")),
        # llama.cpp's default base when a converter wrote none.
        :rope_theta => _gguf_float(get_meta("rope.freq_base", 10_000.0)),
        :tie_word_embeddings => !haskey(idx.tensors, "output.weight"),
    )
    key_length = get_meta("attention.key_length")
    isnothing(key_length) || (raw[:head_dim] = Int(key_length))
    if arch == "gemma2"
        head_dim = Int(something(key_length, hidden ÷ heads))
        raw[:sliding_window] = get_meta("attention.sliding_window")
        raw[:attn_logit_softcapping] = _gguf_float(get_meta("attn_logit_softcapping"))
        raw[:final_logit_softcapping] = _gguf_float(get_meta("final_logit_softcapping"))
        # GGUF does not carry it; llama.cpp scales by the head size except on the
        # 27B, whose 32 heads of 128 use hidden_size ÷ heads instead.
        raw[:query_pre_attn_scalar] = hidden == 4608 && heads == 32 ? hidden ÷ heads : head_dim
    end
    return raw
end

# A weight's shape with no storage behind it, which the model constructors
# allocate in place of every matrix when the matrices come from the file instead.
struct _Unallocated{N} <: AbstractArray{Float32,N}
    dims::NTuple{N,Int}
end
Base.size(a::_Unallocated) = a.dims
Base.getindex(::_Unallocated, ::Int...) = error("this weight was never loaded")
_unallocated(dims::Integer...) = _Unallocated(Int.(dims))

# A 2-D GGUF tensor as the HF-shaped (out, in) matrix it stores, left in the file.
function _gguf_matrix(idx::GGUF._GGUFIndex, name::AbstractString)
    info = idx.tensors[name]
    n_in, n_out = info.dims
    per_block, block_bytes = GGUF._block(info.type)
    nbytes = n_out * (n_in ÷ per_block) * block_bytes
    return GGUFMatrix(info.type, view(idx.buf, (info.offset + 1):(info.offset + nbytes)), n_out, n_in)
end

# Build the model around placeholders, then put every weight in by its path: each
# matrix as a `GGUFMatrix` over the file, the embedding as a `GGUFEmbedding`, and
# the norms and biases, which are small, decoded to `Float32`.
function _load_gguf_packed(model_type, cfg, weights::GGUFWeights, mapping)
    idx, names = weights.index, weights.names
    leaves = IdDict{Any,Any}()
    paths = Dict{Tuple,Any}()
    missing_keys = setdiff(keys(mapping), keys(names))
    isempty(missing_keys) ||
        throw(ArgumentError("GGUF file is missing tensors for $(sort(collect(missing_keys)))"))
    for (key, (path, transform)) in mapping
        name = names[key]
        paths[path] = if length(idx.tensors[name].dims) == 2
            m = _gguf_matrix(idx, name)
            if weights.arch == "llama" && occursin(r"[qk]_proj\.weight$", key)
                heads = endswith(key, "q_proj.weight") ? weights.heads : weights.kv_heads
                m = _permute_rows(m, _rotary_order(size(m, 1), heads))
            end
            transform === :transpose ? GGUFEmbedding(m) : m
        else
            weights[key]
        end
    end
    if cfg.tie_word_embeddings
        paths[(:lm_head, :weight)] = _gguf_matrix(idx, "token_embd.weight")
    end

    lm = model_type(cfg; init=_unallocated)
    exclude(_, x) = x isa AbstractArray{<:Number} || Functors.isleaf(x)
    loaded = Functors.fmap_with_path(lm; exclude=exclude) do kp, x
        path = Tuple(kp.keys)
        haskey(paths, path) || return x
        w = paths[path]
        size(w) == size(x) ||
            throw(DimensionMismatch("$(join(path, '.')): model $(size(x)) vs file $(size(w))"))
        return w
    end
    unfilled = String[]
    Functors.fmap_with_path(loaded; exclude=exclude) do kp, x
        x isa _Unallocated && push!(unfilled, join(kp.keys, '.'))
        return x
    end
    isempty(unfilled) || throw(ArgumentError("GGUF file left weights unset: $(unfilled)"))
    return loaded
end

"""
    load_gguf_model(path; packed=true) -> model

Build a model from a llama.cpp GGUF file. The architecture and hyperparameters
come from the file's metadata.

Weights stay in the file's encoding by default: every projection matrix becomes a
[`GGUFMatrix`](@ref) over the memory-mapped file and the embedding a
[`GGUFEmbedding`](@ref), decoded block by block as they are multiplied, so a
Q4_K_M model takes about its file size in memory. Only the norms and biases are
decoded up front. This path runs on the CPU. With `packed=false` every tensor is
dequantized to `Float32` as it is copied in, for a dense model that trains and
moves to a GPU like any other; memory then peaks at the dense model plus one
tensor.

Supported: GGUF architectures `llama` (Llama 2 and 3.0, Mistral, SmolLM and the
like), `qwen2`, `qwen3` and `gemma2`, in any tensor type
[`GGUF`](@ref HuggingFaceTransformers.GGUF) decodes. Llama 3.1 and later store
per-dimension RoPE factors, which raise rather than load as plain RoPE.

The tokenizer, chat template and stop tokens the file carries load from the same
path with `load_tokenizer`, `load_chat_template` and `load_generation_config`.
"""
function load_gguf_model(path::AbstractString; packed::Bool=true)
    idx = GGUF._open_gguf(path)
    arch = String(get(idx.metadata, "general.architecture", ""))
    haskey(_GGUF_ARCHITECTURES, arch) || throw(
        ArgumentError(
            "unsupported GGUF architecture `$(arch)`; supported are " *
            "$(sort(collect(keys(_GGUF_ARCHITECTURES))))",
        ),
    )
    raw = _gguf_config(idx, arch)

    names = Dict{String,String}()
    unknown = String[]
    for name in keys(idx.tensors)
        name == "rope_freqs.weight" && continue
        hf = _hf_tensor_name(arch, name)
        isnothing(hf) ? push!(unknown, name) : (names[hf] = name)
    end
    isempty(unknown) ||
        throw(ArgumentError("GGUF tensors with no known mapping: $(sort(unknown))"))

    config_type, model_type = AUTO_ARCHITECTURES[_GGUF_ARCHITECTURES[arch]]
    cfg = config_from_json(config_type, raw)
    weights = GGUFWeights(idx, names, arch, raw[:num_attention_heads], raw[:num_key_value_heads])
    packed || return _build_and_load(model_type, cfg, weights)
    return _load_gguf_packed(model_type, cfg, weights, _state_dict_map(model_type, cfg))
end

_state_dict_map(::Type{<:LlamaForCausalLM}, cfg) = llama_state_dict_map(cfg)
_state_dict_map(::Type{<:QwenForCausalLM}, cfg) = qwen_state_dict_map(cfg)
_state_dict_map(::Type{<:GemmaForCausalLM}, cfg) = gemma_state_dict_map(cfg)
