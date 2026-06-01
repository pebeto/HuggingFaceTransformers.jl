"""
    GPT2Config

Architectural hyperparameters for a GPT-2 model. Field names mirror
HuggingFace's `config.json` keys (Julianized to snake_case).

GPT-2 predates most of the Llama-family conventions:

- `LayerNorm` (with bias) for normalization, not `RMSNorm`.
- Learned absolute position embeddings added to token embeddings in the
  trunk — no RoPE in the attention layer.
- Non-gated GELU MLP (tanh approximation), not SwiGLU.
- Fused `c_attn.weight` Q/K/V projection stored as a single tensor.
- HF stores GPT-2 linear weights in `Conv1D` orientation `(in, out)`,
  so the state-dict loader transposes them on load.
- `tie_word_embeddings = true` (lm_head shares the input embedding).
"""
Base.@kwdef struct GPT2Config
    vocab_size::Int
    hidden_size::Int                     # `n_embd` in HF's config
    intermediate_size::Int               # MLP inner dim; defaults to 4 * hidden
    num_hidden_layers::Int               # `n_layer`
    num_attention_heads::Int             # `n_head`
    max_position_embeddings::Int         # `n_positions`
    layer_norm_eps::Float64 = 1.0e-5
    tie_word_embeddings::Bool = true
end

"""
    GPT2Model{TE, PE, L, N}

GPT-2 trunk: token embedding + learned position embedding (summed into
the input), stack of `DecoderLayer`s with `LayerNorm`s, and a final
`LayerNorm`. Returns hidden states; the LM head lives in
`GPT2ForCausalLM`.
"""
struct GPT2Model{TE,PE,L,N}
    embed_tokens::TE
    embed_positions::PE
    layers::L
    norm::N
end

function (m::GPT2Model)(
    input_ids::AbstractMatrix{<:Integer}; caches=nothing, step=nothing, position_ids=nothing
)
    # `input_ids` is (seq, batch) — matches TokenEmbedding's convention.
    seq_len = size(input_ids, 1)
    if isnothing(position_ids)
        start_pos = isnothing(step) ? 0 : step - 1
        position_ids = collect(start_pos:(start_pos + seq_len - 1))
    end

    # Token embedding: (hidden, seq, batch)
    h = m.embed_tokens(input_ids)
    # Position embedding looked up via the same TokenEmbedding mechanism:
    # (hidden, seq). Broadcast over the batch dim.
    pe = m.embed_positions(position_ids)
    h = h .+ reshape(pe, size(pe, 1), size(pe, 2), 1)

    for i in eachindex(m.layers)
        cache_i = isnothing(caches) ? nothing : caches[i]
        h = m.layers[i](h; cache=cache_i, step=step, position_ids=position_ids)
    end
    return m.norm(h)
end

Flux.@layer GPT2Model

"""
    GPT2ForCausalLM{C, M, H}

`GPT2Model` + bias-less `Linear` LM head. `tie_word_embeddings=true` on
every standard GPT-2 checkpoint, so the LM head is materialized from
the input embedding at load time (matching the Llama tied path).
"""
struct GPT2ForCausalLM{C,M,H}
    config::C
    model::M
    lm_head::H
end

function (lm::GPT2ForCausalLM)(input_ids::AbstractMatrix{<:Integer}; kwargs...)
    h = lm.model(input_ids; kwargs...)
    return lm.lm_head(h)
end

Flux.@layer GPT2ForCausalLM
function Flux.Optimisers.trainable(lm::GPT2ForCausalLM)
    return (; model=lm.model, lm_head=lm.lm_head)
end

"""
    GPT2ForCausalLM(cfg::GPT2Config)

Materialize a fresh, randomly-initialized `GPT2ForCausalLM` matching
the shapes in `cfg`. Attention is built with `rope=nothing` (GPT-2's
positional info comes from `embed_positions`, not RoPE) and
`qkv_bias=true` (GPT-2 biases everything).
"""
function GPT2ForCausalLM(cfg::GPT2Config)
    head_dim = cfg.hidden_size ÷ cfg.num_attention_heads
    eps = Float32(cfg.layer_norm_eps)

    layers = [
        DecoderLayer(
            GQA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_attention_heads,    # MHA: num_kv == num_q
                head_dim,
                nothing;                    # no RoPE
                qkv_bias=true,
                wo_bias=true,               # GPT-2 biases c_proj
            ),
            LayerNorm(cfg.hidden_size, eps),
            LayerNorm(cfg.hidden_size, eps),
            GeluMLP(cfg.hidden_size, cfg.intermediate_size),
        ) for _ in 1:(cfg.num_hidden_layers)
    ]

    model = GPT2Model(
        TokenEmbedding(cfg.vocab_size, cfg.hidden_size),
        TokenEmbedding(cfg.max_position_embeddings, cfg.hidden_size),
        layers,
        LayerNorm(cfg.hidden_size, eps),
    )
    lm_head = Linear(cfg.hidden_size, cfg.vocab_size)
    return GPT2ForCausalLM(cfg, model, lm_head)
end

"""
    build_caches(lm::GPT2ForCausalLM, max_seq, batch_size; eltype=Float32)
"""
function build_caches(
    lm::GPT2ForCausalLM, max_seq::Integer, batch_size::Integer; eltype=Float32
)
    cfg = lm.config
    head_dim = cfg.hidden_size ÷ cfg.num_attention_heads
    return [
        KVCache(head_dim, cfg.num_attention_heads, max_seq, batch_size; eltype=eltype) for
        _ in 1:(cfg.num_hidden_layers)
    ]
end

"""
    gpt2_state_dict_map(cfg::GPT2Config) -> Dict{String, Tuple{Tuple, Symbol}}

Pure-data state-dict map for the entries GPT-2 stores 1:1 with the
internal model. The HF naming is `transformer.h.{i}.…`, not Llama's
`model.layers.{i}.…`.

The fused `c_attn.weight` / `c_attn.bias` (concatenated Q/K/V) and the
Conv1D-orientation weights (`c_proj`, `c_fc`) are NOT in this table —
[`load_state_dict!`](@ref) slices and transposes them at load time.
LayerNorm biases ARE in this table because they're 1:1 vectors.
"""
function gpt2_state_dict_map(cfg::GPT2Config)
    out = Dict{String,Tuple{Tuple,Symbol}}()

    # HF stores (vocab, hidden); we want (hidden, vocab).
    out["transformer.wte.weight"] = ((:model, :embed_tokens, :weight), :transpose)
    out["transformer.wpe.weight"] = ((:model, :embed_positions, :weight), :transpose)

    for i in 0:(cfg.num_hidden_layers - 1)
        layer_path = (:model, :layers, i + 1)
        hf_prefix = "transformer.h.$(i)"

        out["$(hf_prefix).ln_1.weight"] = (
            (layer_path..., :input_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).ln_1.bias"] = ((layer_path..., :input_layernorm, :bias), :as_is)
        out["$(hf_prefix).ln_2.weight"] = (
            (layer_path..., :post_attention_layernorm, :weight), :as_is
        )
        out["$(hf_prefix).ln_2.bias"] = (
            (layer_path..., :post_attention_layernorm, :bias), :as_is
        )
    end

    out["transformer.ln_f.weight"] = ((:model, :norm, :weight), :as_is)
    out["transformer.ln_f.bias"] = ((:model, :norm, :bias), :as_is)

    return out
end

"""
    load_state_dict!(lm::GPT2ForCausalLM, weights) -> lm

Populate `lm` in-place. Runs the 1:1 loader for LayerNorm tensors and
embeddings, then per layer:

- Transposes the Conv1D-format `c_attn.weight` from `(hidden, 3*hidden)`
  to `(3*hidden, hidden)` and slices it into `wq`/`wk`/`wv` weights.
- Slices the `c_attn.bias` of length `3*hidden` into `wq`/`wk`/`wv`
  biases.
- Transposes `attn.c_proj.weight` for the attention output projection
  (`wo.weight`), copies `attn.c_proj.bias` into `wo.bias`.
- Transposes `mlp.c_fc.weight` and `mlp.c_proj.weight` (Conv1D), copies
  their biases.

Finally, `lm_head.weight` is materialized from the input embedding
(GPT-2 always ties).
"""
function load_state_dict!(
    lm::GPT2ForCausalLM, weights::AbstractDict{String,<:AbstractArray}
)
    cfg = lm.config
    load_into!(lm, weights, gpt2_state_dict_map(cfg))

    h_size = cfg.hidden_size

    for i in 0:(cfg.num_hidden_layers - 1)
        layer = lm.model.layers[i + 1]
        attn = layer.self_attn
        prefix = "transformer.h.$(i)"

        # Fused QKV weight: Conv1D-shape (hidden, 3*hidden) → transpose →
        # (3*hidden, hidden) → slice along the output dim into wq / wk / wv.
        c_attn_w_key = "$(prefix).attn.c_attn.weight"
        haskey(weights, c_attn_w_key) ||
            throw(KeyError("missing GPT-2 fused QKV weight: $(c_attn_w_key)"))
        c_attn_w = weights[c_attn_w_key]
        size(c_attn_w) == (h_size, 3 * h_size) || throw(
            DimensionMismatch(
                "$(c_attn_w_key) has size $(size(c_attn_w)), expected " *
                "$((h_size, 3 * h_size)) (Conv1D orientation)",
            ),
        )
        c_attn_w_t = permutedims(c_attn_w, (2, 1))   # → (3*hidden, hidden)
        copyto!(attn.wq.weight, @view c_attn_w_t[1:h_size, :])
        copyto!(attn.wk.weight, @view c_attn_w_t[(h_size + 1):(2 * h_size), :])
        copyto!(attn.wv.weight, @view c_attn_w_t[(2 * h_size + 1):(3 * h_size), :])

        # Fused QKV bias: length 3*hidden → slice.
        c_attn_b_key = "$(prefix).attn.c_attn.bias"
        haskey(weights, c_attn_b_key) ||
            throw(KeyError("missing GPT-2 fused QKV bias: $(c_attn_b_key)"))
        c_attn_b = weights[c_attn_b_key]
        length(c_attn_b) == 3 * h_size || throw(
            DimensionMismatch(
                "$(c_attn_b_key) has length $(length(c_attn_b)), expected $(3 * h_size)"
            ),
        )
        copyto!(attn.wq.bias, @view c_attn_b[1:h_size])
        copyto!(attn.wk.bias, @view c_attn_b[(h_size + 1):(2 * h_size)])
        copyto!(attn.wv.bias, @view c_attn_b[(2 * h_size + 1):(3 * h_size)])

        # Attention output projection (c_proj): Conv1D transpose + bias.
        # The GPT-2 factory builds `wo` with `wo_bias=true` so attn.wo.bias
        # is preallocated.
        c_proj_w = weights["$(prefix).attn.c_proj.weight"]
        c_proj_b = weights["$(prefix).attn.c_proj.bias"]
        copyto!(attn.wo.weight, permutedims(c_proj_w, (2, 1)))
        copyto!(attn.wo.bias, c_proj_b)

        # MLP: c_fc and c_proj are both Conv1D-format.
        c_fc_w = weights["$(prefix).mlp.c_fc.weight"]
        c_fc_b = weights["$(prefix).mlp.c_fc.bias"]
        c_mlp_proj_w = weights["$(prefix).mlp.c_proj.weight"]
        c_mlp_proj_b = weights["$(prefix).mlp.c_proj.bias"]
        copyto!(layer.mlp.c_fc.weight, permutedims(c_fc_w, (2, 1)))
        copyto!(layer.mlp.c_fc.bias, c_fc_b)
        copyto!(layer.mlp.c_proj.weight, permutedims(c_mlp_proj_w, (2, 1)))
        copyto!(layer.mlp.c_proj.bias, c_mlp_proj_b)
    end

    if cfg.tie_word_embeddings
        # embed_tokens.weight is (hidden, vocab); lm_head.weight is (vocab, hidden).
        copyto!(lm.lm_head.weight, permutedims(lm.model.embed_tokens.weight, (2, 1)))
    end

    return lm
end
