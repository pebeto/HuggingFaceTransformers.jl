"""
    LlavaMultiModalProjector{L1, L2}

LLaVA's vision→text projector: `linear_2(gelu(linear_1(x)))`, mapping vision
patch features `(vision_hidden, n_patches, batch)` to the LLM hidden size.
"""
struct LlavaMultiModalProjector{L1,L2}
    linear_1::L1
    linear_2::L2
end

function (p::LlavaMultiModalProjector)(x::AbstractArray)
    d, n, batch = size(x)
    h = _gelu_exact.(p.linear_1(reshape(x, d, n * batch)))
    return reshape(p.linear_2(h), :, n, batch)
end

Flux.@layer LlavaMultiModalProjector

"""
    LlavaMultiModalProjector(vision_hidden, text_hidden)

Build a projector from `vision_hidden` to `text_hidden` (both linears biased).
"""
function LlavaMultiModalProjector(vision_hidden::Integer, text_hidden::Integer)
    return LlavaMultiModalProjector(
        Linear(vision_hidden, text_hidden; bias=true),
        Linear(text_hidden, text_hidden; bias=true),
    )
end

"""
    LlavaForConditionalGeneration{V, P, L}

Vision-language model: a vision tower, the [`LlavaMultiModalProjector`](@ref),
and a decoder `language_model` (e.g. `LlamaForCausalLM`). Projected image patch
features are spliced into the text embeddings at `image_token_index` positions
(the caller pre-expands the placeholder into one token per patch), then the
language model runs on the merged embeddings.

The wrapper is vision-tower-agnostic: `vision_tower(pixel_values)` must return
patch features `(vision_hidden, n_patches, batch)` (CLS already dropped, feature
layer already selected).
"""
struct LlavaForConditionalGeneration{V,P,L}
    vision_tower::V
    projector::P
    language_model::L
    image_token_index::Int
end

Flux.@layer LlavaForConditionalGeneration

"""
    LlavaForConditionalGeneration(vision_tower, language_model, vision_hidden; image_token_index)

Assemble a LLaVA model, building the projector from `vision_hidden` to the
language model's hidden size.
"""
function LlavaForConditionalGeneration(
    vision_tower, language_model, vision_hidden::Integer; image_token_index::Integer
)
    projector = LlavaMultiModalProjector(vision_hidden, language_model.config.hidden_size)
    return LlavaForConditionalGeneration(
        vision_tower, projector, language_model, Int(image_token_index)
    )
end

# Replace the columns of `text_embeds` at `image_token` positions with the
# projected patch features, in order. Each batch item must carry exactly as many
# image tokens as there are patches.
function _splice_image(text_embeds, image_features, input_ids, image_token::Integer)
    _, seq, batch = size(text_embeds)
    merged = copy(text_embeds)
    for b in 1:batch
        positions = findall(==(image_token), view(input_ids, :, b))
        length(positions) == size(image_features, 2) || throw(
            DimensionMismatch(
                "batch $(b): $(length(positions)) image tokens but " *
                "$(size(image_features, 2)) image patches",
            ),
        )
        for (j, p) in enumerate(positions)
            @views merged[:, p, b] .= image_features[:, j, b]
        end
    end
    return merged
end

function _merge_and_run(
    m::LlavaForConditionalGeneration,
    vision_features::AbstractArray,
    input_ids::AbstractMatrix{<:Integer},
)
    image_features = m.projector(vision_features)                    # (d_text, n, batch)
    text_embeds = m.language_model.model.embed_tokens(input_ids)     # (d_text, seq, batch)
    merged = _splice_image(text_embeds, image_features, input_ids, m.image_token_index)
    h = forward_embeds(m.language_model.model, merged)
    return m.language_model.lm_head(h)                               # (vocab, seq, batch)
end

function (m::LlavaForConditionalGeneration)(
    pixel_values, input_ids::AbstractMatrix{<:Integer}
)
    return _merge_and_run(m, m.vision_tower(pixel_values), input_ids)
end

"""
    forward_from_features(m, vision_features, input_ids) -> logits

Run the projector + splice + language model on precomputed vision patch features,
skipping the vision tower. Useful for testing/parity that isolates the tower.
"""
function forward_from_features(
    m::LlavaForConditionalGeneration,
    vision_features::AbstractArray,
    input_ids::AbstractMatrix{<:Integer},
)
    return _merge_and_run(m, vision_features, input_ids)
end

"""
    generate_multimodal(m, pixel_values, input_ids; max_new_tokens=32, eos_token_id=nothing) -> Vector{Int}

Greedy multimodal decode (batch-1). The vision features are computed once; the
decoder is recomputed each step (no KV cache; correctness first).
"""
function generate_multimodal(
    m::LlavaForConditionalGeneration,
    pixel_values,
    input_ids::AbstractVector{<:Integer};
    max_new_tokens::Integer=32,
    eos_token_id=nothing,
)
    vision_features = m.vision_tower(pixel_values)
    tokens = Int[Int(t) for t in input_ids]
    for _ in 1:max_new_tokens
        logits = _merge_and_run(m, vision_features, reshape(tokens, :, 1))
        next = argmax(logits[:, end, 1]) - 1
        push!(tokens, next)
        next == eos_token_id && break
    end
    return tokens
end

"""
    llava_state_dict_map(m::LlavaForConditionalGeneration) -> Dict{String, Tuple{Tuple, Symbol}}

Projector keys plus the language model's own map re-rooted under
`language_model.`. Vision-tower keys are intentionally absent, so load the
tower with its own loader.
"""
function llava_state_dict_map(m::LlavaForConditionalGeneration)
    out = Dict{String,Tuple{Tuple,Symbol}}()
    out["multi_modal_projector.linear_1.weight"] = (
        (:projector, :linear_1, :weight), :as_is
    )
    out["multi_modal_projector.linear_1.bias"] = ((:projector, :linear_1, :bias), :as_is)
    out["multi_modal_projector.linear_2.weight"] = (
        (:projector, :linear_2, :weight), :as_is
    )
    out["multi_modal_projector.linear_2.bias"] = ((:projector, :linear_2, :bias), :as_is)
    for (k, (path, transform)) in llama_state_dict_map(m.language_model.config)
        out["language_model.$(k)"] = ((:language_model, path...), transform)
    end
    return out
end

"""
    load_state_dict!(m::LlavaForConditionalGeneration, weights) -> m

Load the projector and language model (vision-tower keys are ignored). Ties the
LM head when the language config requests it, mirroring `LlamaForCausalLM`.
"""
function load_state_dict!(
    m::LlavaForConditionalGeneration, weights::AbstractDict{String,<:AbstractArray}
)
    load_into!(m, weights, llava_state_dict_map(m))
    if m.language_model.config.tie_word_embeddings
        copyto!(
            m.language_model.lm_head.weight,
            permutedims(m.language_model.model.embed_tokens.weight, (2, 1)),
        )
    end
    return m
end
