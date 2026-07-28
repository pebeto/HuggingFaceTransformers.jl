using Test
using Random
using Flux
using Functors: isleaf
using HuggingFaceTransformers.Models
using HuggingFaceTransformers.Layers: Linear, checkpoint

# Integration test for the LoRA fine-tuning path (mirrors examples/lora_finetune.jl):
# wrap Linears with LoRA, freeze the base, train through a checkpointed forward,
# and confirm only the LoRA parameters move.
@testset "LoRA fine-tuning reduces loss with the base frozen" begin
    Random.seed!(0x71)
    vocab, seqlen = 16, 8
    cfg = LlamaConfig(;
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=seqlen,
        rope_theta=10000.0,
        tie_word_embeddings=false,
    )
    base = LlamaForCausalLM(cfg)
    model = fmap(base; exclude=(x -> x isa Linear || isleaf(x))) do x
        x isa Linear ? lora_wrap(x, 4, 8.0) : x
    end

    frozen_base = copy(model.model.layers[1].self_attn.wq.base.weight)
    frozen_emb = copy(model.model.embed_tokens.weight)
    lora_B_before = copy(model.model.layers[1].self_attn.wq.lora_B)

    data = reshape(rand(0:(vocab - 1), seqlen), :, 1)
    function loss(m)
        logits = checkpoint(m, data)
        pred = reshape(logits[:, 1:(end - 1), :], vocab, :)
        targets = Flux.onehotbatch(vec(data[2:end, :]), 0:(vocab - 1))
        return Flux.logitcrossentropy(pred, targets)
    end

    opt = Flux.setup(Flux.Adam(0.05), model)
    Flux.freeze!(opt.model.embed_tokens)
    Flux.freeze!(opt.model.norm)
    for layer in opt.model.layers
        Flux.freeze!(layer.input_layernorm)
        Flux.freeze!(layer.post_attention_layernorm)
    end

    l0 = loss(model)
    for _ in 1:30
        _, grads = Flux.withgradient(loss, model)
        Flux.update!(opt, model, grads[1])
    end
    l1 = loss(model)

    @test l1 < l0 / 2                                                      # loss dropped a lot
    @test model.model.layers[1].self_attn.wq.base.weight == frozen_base   # base frozen
    @test model.model.embed_tokens.weight == frozen_emb                   # embeddings frozen
    @test model.model.layers[1].self_attn.wq.lora_B != lora_B_before      # LoRA params moved
end
