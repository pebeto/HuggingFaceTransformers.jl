"""
    NeoXConfig

Architectural hyperparameters for a GPT-NeoX / Pythia model. Field names
mirror HuggingFace's `config.json` keys (Julianized to snake_case).

GPT-NeoX diverges from the Llama family on three axes:

- **Parallel residual** decoder block: both the attention and MLP
  branches normalize from the same `x` (instead of attention writing
  into `x` first). See [`NeoXDecoderLayer`](@ref).
- **Partial RoPE**: only the first `partial_rotary_factor * head_dim`
  channels of each head are rotated; the remaining channels pass
  through. Pythia and the original GPT-NeoX-20B both use 0.25.
- **Interleaved fused QKV**: `query_key_value.weight` is laid out per
  head as `[Q_h; K_h; V_h; Q_{h+1}; K_{h+1}; V_{h+1}; ...]`, not
  GPT-2's concatenated `[Q; K; V]`.

LayerNorm (with bias), GeluMLP, and the `wo_bias=true` knob were
already in the layer kit from GPT-2 — NeoX reuses them.
"""
Base.@kwdef struct NeoXConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    num_hidden_layers::Int
    num_attention_heads::Int
    head_dim::Int
    max_position_embeddings::Int
    rope_theta::Float64 = 10_000.0
    partial_rotary_factor::Float64 = 0.25
    layer_norm_eps::Float64 = 1.0e-5
    tie_word_embeddings::Bool = false
end

"""
    NeoXDecoderLayer{A, N1, N2, M}

GPT-NeoX parallel-residual decoder block. The two branches read from
the same `x`:

```
x = x + self_attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))
```

This is structurally different from Llama's sequential pre-norm block,
which would write the attention output into `x` before the MLP norm
sees it.
"""
struct NeoXDecoderLayer{A,N1,N2,M}
    self_attn::A
    input_layernorm::N1
    post_attention_layernorm::N2
    mlp::M
end

function (layer::NeoXDecoderLayer)(
    x::AbstractArray; cache=nothing, step=nothing, position_ids=nothing
)
    h_attn = layer.input_layernorm(x)
    h_attn = layer.self_attn(
        h_attn; cache=cache, step=step, position_ids=position_ids
    )
    h_mlp = layer.post_attention_layernorm(x)
    h_mlp = layer.mlp(h_mlp)
    return x .+ h_attn .+ h_mlp
end

Flux.@layer NeoXDecoderLayer

"""
    NeoXForCausalLM{C, M, H}

GPT-NeoX top-level container: a `DecoderModel` (reused — its forward is
generic over layer type) populated with `NeoXDecoderLayer`s, plus a
bias-less `Linear` LM head. Embeddings are not tied on standard
checkpoints.
"""
struct NeoXForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::NeoXForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer NeoXForCausalLM
function Flux.Optimisers.trainable(lm::NeoXForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    NeoXForCausalLM(cfg::NeoXConfig)

Materialize a fresh, randomly-initialized `NeoXForCausalLM` matching the
shapes in `cfg`. Attention is MHA (`num_kv = num_q`) with biased Q/K/V
and biased output projection, and the RoPE inside attention rotates
only `floor(head_dim * partial_rotary_factor)` channels.
"""
function NeoXForCausalLM(cfg::NeoXConfig)
    eps = Float32(cfg.layer_norm_eps)
    rotary_dim = Int(floor(cfg.head_dim * cfg.partial_rotary_factor))
    # NeoX requires an even number of rotated channels (rotation operates on pairs).
    iseven(rotary_dim) || throw(
        ArgumentError(
            "partial RoPE dim $(rotary_dim) (= head_dim * partial_rotary_factor) " *
            "must be even"
        ),
    )

    rope = RoPE(cfg.head_dim; base=cfg.rope_theta, rotary_dim=rotary_dim)

    layers = [
        NeoXDecoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_attention_heads,     # MHA
                cfg.head_dim,
                rope;
                qkv_bias=true,
                wo_bias=true,
            ),
            LayerNorm(cfg.hidden_size, eps),
            LayerNorm(cfg.hidden_size, eps),
            GeluMLP(cfg.hidden_size, cfg.intermediate_size),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    model = DecoderModel(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        layers,
        LayerNorm(cfg.hidden_size, eps),
    )
    lm_head = Linear(cfg.hidden_size, cfg.vocab_size)
    return NeoXForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::NeoXForCausalLM, max_seq, batch_size; eltype=Float32)
"""
function build_caches(
    lm::NeoXForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    return [
        KVCache(
            cfg.head_dim, cfg.num_attention_heads, max_seq, batch_size; eltype=eltype
        ) for _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    neox_state_dict_map(cfg::NeoXConfig) -> Dict{String, Tuple{Tuple, Symbol}}

1:1 entries for GPT-NeoX. HF's layout puts the trunk under `gpt_neox.…`
and the LM head at the root as `embed_out.weight`.

Fused `query_key_value` weights and biases are NOT in this table —
[`load_state_dict!`](@ref) handles their interleaved-per-head slicing
directly.
"""
function neox_state_dict_map(cfg::NeoXConfig)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    # HF stores embeddings as (vocab, hidden); we want (hidden, vocab).
    out["gpt_neox.embed_in.weight"] =
        ((:model, :embed_tokens, :weight), :transpose)

    for i in 0:(cfg.num_hidden_layers - 1)
        layer_path = (:model, :layers, i + 1)
        hf_prefix = "gpt_neox.layers.$(i)"

        out["$(hf_prefix).input_layernorm.weight"] =
            ((layer_path..., :input_layernorm, :weight), :as_is)
        out["$(hf_prefix).input_layernorm.bias"] =
            ((layer_path..., :input_layernorm, :bias), :as_is)
        out["$(hf_prefix).post_attention_layernorm.weight"] =
            ((layer_path..., :post_attention_layernorm, :weight), :as_is)
        out["$(hf_prefix).post_attention_layernorm.bias"] =
            ((layer_path..., :post_attention_layernorm, :bias), :as_is)

        # NeoX uses `nn.Linear` (not Conv1D); weights are already in
        # (out, in) orientation, so no transpose needed for these.
        out["$(hf_prefix).attention.dense.weight"] =
            ((layer_path..., :self_attn, :wo, :weight), :as_is)
        out["$(hf_prefix).attention.dense.bias"] =
            ((layer_path..., :self_attn, :wo, :bias), :as_is)

        out["$(hf_prefix).mlp.dense_h_to_4h.weight"] =
            ((layer_path..., :mlp, :c_fc, :weight), :as_is)
        out["$(hf_prefix).mlp.dense_h_to_4h.bias"] =
            ((layer_path..., :mlp, :c_fc, :bias), :as_is)
        out["$(hf_prefix).mlp.dense_4h_to_h.weight"] =
            ((layer_path..., :mlp, :c_proj, :weight), :as_is)
        out["$(hf_prefix).mlp.dense_4h_to_h.bias"] =
            ((layer_path..., :mlp, :c_proj, :bias), :as_is)
    end

    out["gpt_neox.final_layer_norm.weight"] = ((:model, :norm, :weight), :as_is)
    out["gpt_neox.final_layer_norm.bias"] = ((:model, :norm, :bias), :as_is)

    if !cfg.tie_word_embeddings
        # embed_out lives outside the `gpt_neox.` namespace in HF.
        out["embed_out.weight"] = ((:lm_head, :weight), :as_is)
    end

    return out
end

"""
    load_state_dict!(lm::NeoXForCausalLM, weights) -> lm

Populate `lm` in-place. Runs the 1:1 loader, then per layer unpacks the
fused `query_key_value` weight and bias. The HF layout is per-head
interleaved: for each head `h`, the corresponding output slice of
`query_key_value.weight` (and of the bias) holds
`[Q_h; K_h; V_h]` concatenated, and these per-head slices are
concatenated across `h`. We walk heads and copy each slice into the
right rows of `wq` / `wk` / `wv`.
"""
function load_state_dict!(
    lm::NeoXForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    cfg = lm.config
    load_into!(lm, weights, neox_state_dict_map(cfg))

    n_heads = cfg.num_attention_heads
    hd = cfg.head_dim

    for i in 0:(cfg.num_hidden_layers - 1)
        layer = lm.model.layers[i + 1]
        attn = layer.self_attn
        prefix = "gpt_neox.layers.$(i).attention.query_key_value"

        # Weight: HF stores (3*n_heads*hd, hidden_size), per-head interleaved.
        qkv_w_key = "$(prefix).weight"
        haskey(weights, qkv_w_key) ||
            throw(KeyError("missing NeoX fused QKV weight: $(qkv_w_key)"))
        qkv_w = weights[qkv_w_key]
        expected_rows = 3 * n_heads * hd
        size(qkv_w, 1) == expected_rows || throw(
            DimensionMismatch(
                "$(qkv_w_key) first dim is $(size(qkv_w, 1)), expected " *
                "$(expected_rows) (= 3 * num_heads * head_dim)",
            ),
        )

        # Bias: length 3*n_heads*hd, same per-head interleaving.
        qkv_b_key = "$(prefix).bias"
        haskey(weights, qkv_b_key) ||
            throw(KeyError("missing NeoX fused QKV bias: $(qkv_b_key)"))
        qkv_b = weights[qkv_b_key]
        length(qkv_b) == expected_rows || throw(
            DimensionMismatch(
                "$(qkv_b_key) length is $(length(qkv_b)), expected $(expected_rows)",
            ),
        )

        for h in 0:(n_heads - 1)
            src_base = h * 3 * hd
            q_rows = (src_base + 1):(src_base + hd)
            k_rows = (src_base + hd + 1):(src_base + 2 * hd)
            v_rows = (src_base + 2 * hd + 1):(src_base + 3 * hd)

            dst_rows = (h * hd + 1):((h + 1) * hd)

            copyto!(view(attn.wq.weight, dst_rows, :), view(qkv_w, q_rows, :))
            copyto!(view(attn.wk.weight, dst_rows, :), view(qkv_w, k_rows, :))
            copyto!(view(attn.wv.weight, dst_rows, :), view(qkv_w, v_rows, :))

            copyto!(view(attn.wq.bias, dst_rows), view(qkv_b, q_rows))
            copyto!(view(attn.wk.bias, dst_rows), view(qkv_b, k_rows))
            copyto!(view(attn.wv.bias, dst_rows), view(qkv_b, v_rows))
        end
    end

    if cfg.tie_word_embeddings
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
