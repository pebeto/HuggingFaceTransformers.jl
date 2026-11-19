using Test
using JSON3
using Random
using HuggingFaceTransformers
using HuggingFaceTransformers.GGUF
using HuggingFaceTransformers.GGUF: load_gguf, GGUFFile
using HuggingFaceTransformers.Models:
    LlamaConfig, QwenConfig, GemmaConfig, LlamaForCausalLM, QwenForCausalLM, GemmaForCausalLM,
    llama_state_dict_map, qwen_state_dict_map, gemma_state_dict_map, config_from_json,
    load_gguf_model, _resolve, GGUFMatrix, GGUFEmbedding

# Minimal GGUF writer, used only to build round-trip fixtures in-process.
_w(io, x) = write(io, htol(x))

function _write_str(io, s::AbstractString)
    _w(io, UInt64(ncodeunits(s)))
    return write(io, codeunits(s))
end

# Metadata value types we exercise: u32=4, f32=6, bool=7, string=8, array=9,
# i32=5. The array writer takes (elem_type, elements).
function _write_value(io, vtype::Int, val)
    if vtype == 4
        _w(io, UInt32(val))
    elseif vtype == 5
        _w(io, Int32(val))
    elseif vtype == 6
        _w(io, Float32(val))
    elseif vtype == 7
        _w(io, UInt8(val ? 1 : 0))
    elseif vtype == 8
        _write_str(io, val)
    elseif vtype == 9
        elem_type, elems = val
        _w(io, UInt32(elem_type))
        _w(io, UInt64(length(elems)))
        for e in elems
            _write_value(io, elem_type, e)
        end
    else
        error("test writer: unhandled vtype $(vtype)")
    end
end

_align_up(n, a) = ((n + a - 1) ÷ a) * a

# tensors: name => (ggml_type::Int, dims::Vector{Int}, raw::Vector{UInt8})
# metadata: key => (vtype::Int, value)
function _write_gguf(path; metadata, tensors)
    align = 32
    hdr = IOBuffer()
    write(hdr, b"GGUF")
    _w(hdr, UInt32(3))
    _w(hdr, UInt64(length(tensors)))
    _w(hdr, UInt64(length(metadata)))
    for (k, (vtype, val)) in metadata
        _write_str(hdr, k)
        _w(hdr, UInt32(vtype))
        _write_value(hdr, vtype, val)
    end

    # Lay out the data section, recording each tensor's aligned offset.
    data = IOBuffer()
    offsets = Dict{String,Int}()
    for (name, (_, _, raw)) in tensors
        offsets[name] = position(data)
        write(data, raw)
        pad = _align_up(position(data), align) - position(data)
        write(data, zeros(UInt8, pad))
    end

    for (name, (ggml_type, dims, _)) in tensors
        _write_str(hdr, name)
        _w(hdr, UInt32(length(dims)))
        for d in dims
            _w(hdr, UInt64(d))
        end
        _w(hdr, UInt32(ggml_type))
        _w(hdr, UInt64(offsets[name]))
    end

    header = take!(hdr)
    pad = _align_up(length(header), align) - length(header)
    open(path, "w") do f
        write(f, header)
        write(f, zeros(UInt8, pad))
        write(f, take!(data))
    end
    return path
end

# Quantizers that exactly invert the reader's dequant (d stored as Float16).
function _pack_q8_0(x::Vector{Float32})
    io = IOBuffer()
    for b in 1:(length(x) ÷ 32)
        blk = @view x[((b - 1) * 32 + 1):(b * 32)]
        amax = maximum(abs, blk)
        d = amax == 0 ? Float16(1) : Float16(amax / 127)
        _w(io, d)
        df = Float32(d)
        for v in blk
            write(io, Int8(clamp(round(Int, v / df), -127, 127)))
        end
    end
    return take!(io)
end

function _pack_q4_0(x::Vector{Float32})
    io = IOBuffer()
    for b in 1:(length(x) ÷ 32)
        blk = @view x[((b - 1) * 32 + 1):(b * 32)]
        amax = maximum(abs, blk)
        d = amax == 0 ? Float16(1) : Float16(amax / 8)
        _w(io, d)
        df = Float32(d)
        for k in 1:16
            lo = clamp(round(Int, blk[k] / df) + 8, 0, 15)
            hi = clamp(round(Int, blk[k + 16] / df) + 8, 0, 15)
            write(io, UInt8(lo | (hi << 4)))
        end
    end
    return take!(io)
end

@testset verbose = true "GGUF metadata" begin
    mktempdir() do dir
        path = joinpath(dir, "meta.gguf")
        _write_gguf(
            path;
            metadata=[
                "general.architecture" => (8, "llama"),
                "llama.block_count" => (4, 2),
                "llama.attention.layer_norm_rms_epsilon" => (6, 1.0f-5),
                "general.quantized" => (7, true),
                "tokenizer.ggml.tokens" => (9, (8, ["<s>", "</s>", "a"])),
                "some.ints" => (9, (5, Int32[1, 2, 3])),
            ],
            tensors=Dict{String,Tuple{Int,Vector{Int},Vector{UInt8}}}(),
        )
        f = load_gguf(path)
        @test f.version == 3
        @test f.metadata["general.architecture"] == "llama"
        @test f.metadata["llama.block_count"] == 2
        @test f.metadata["llama.attention.layer_norm_rms_epsilon"] ≈ 1.0f-5
        @test f.metadata["general.quantized"] === true
        @test f.metadata["tokenizer.ggml.tokens"] == ["<s>", "</s>", "a"]
        @test f.metadata["some.ints"] == Int32[1, 2, 3]
    end
end

@testset verbose = true "GGUF tensor dtypes" begin
    @testset "F32 round-trips exactly, dims kept contiguous-first" begin
        mktempdir() do dir
            path = joinpath(dir, "f32.gguf")
            W = Float32[1 2 3 4; 5 6 7 8]              # (2, 4) Julia
            flat = vec(W)                              # column-major
            raw = collect(reinterpret(UInt8, flat))
            _write_gguf(
                path;
                metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (0, [2, 4], raw)),  # ggml F32
            )
            f = load_gguf(path)
            @test f.tensor_types["t"] == "F32"
            @test size(f.tensors["t"]) == (2, 4)
            @test f.tensors["t"] == W
        end
    end

    @testset "F16 round-trips within Float16 precision" begin
        mktempdir() do dir
            path = joinpath(dir, "f16.gguf")
            orig = Float32[0.5, -1.25, 3.0, 100.0]
            raw = collect(reinterpret(UInt8, Float16.(orig)))
            _write_gguf(
                path;
                metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (1, [4], raw)),     # ggml F16
            )
            f = load_gguf(path)
            @test f.tensor_types["t"] == "F16"
            @test f.tensors["t"] ≈ Float32.(Float16.(orig))
        end
    end

    @testset "Q8_0 dequantizes within one quant step" begin
        mktempdir() do dir
            path = joinpath(dir, "q8.gguf")
            orig = Float32.(collect(range(-3.1, 2.7; length=32)))
            amax = maximum(abs, orig)
            d = Float32(Float16(amax / 127))
            _write_gguf(
                path;
                metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (8, [32], _pack_q8_0(orig))),
            )
            f = load_gguf(path)
            @test f.tensor_types["t"] == "Q8_0"
            @test length(f.tensors["t"]) == 32
            @test maximum(abs.(f.tensors["t"] .- orig)) <= d + 1.0f-6
        end
    end

    @testset "Q4_0 dequantizes within one quant step" begin
        mktempdir() do dir
            path = joinpath(dir, "q4.gguf")
            orig = Float32.(collect(range(-2.0, 2.0; length=32)))
            amax = maximum(abs, orig)
            d = Float32(Float16(amax / 8))
            _write_gguf(
                path;
                metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (2, [32], _pack_q4_0(orig))),
            )
            f = load_gguf(path)
            @test f.tensor_types["t"] == "Q4_0"
            # Q4_0 clips the most-positive value (nibble range -8..7); bound
            # the interior where the step is the only error.
            @test maximum(abs.(f.tensors["t"][1:31] .- orig[1:31])) <= d + 1.0f-6
        end
    end

    @testset "unsupported dtype errors with its name" begin
        mktempdir() do dir
            path = joinpath(dir, "iq4xs.gguf")
            # IQ4_XS (23) with bogus bytes; the reader should refuse before
            # touching them.
            _write_gguf(
                path;
                metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (23, [256], zeros(UInt8, 256))),
            )
            @test_throws ArgumentError load_gguf(path)
        end
    end
end

@testset "load_gguf rejects a non-GGUF file" begin
    mktempdir() do dir
        path = joinpath(dir, "nope.bin")
        write(path, b"NOTGGUF\0\0\0\0")
        @test_throws ArgumentError load_gguf(path)
    end
end

@testset verbose = true "every decoded type matches gguf-py on real blocks" begin
    # Two blocks per type cut from published GGUF files, with gguf-py's
    # dequantization of them; see test/fixtures/record_gguf_blocks.py.
    blocks = JSON3.read(read(joinpath(@__DIR__, "fixtures", "gguf_quant_blocks.json"), String))
    mktempdir() do dir
        for (name, b) in pairs(blocks)
            values = Float32.(b.values)
            path = joinpath(dir, "$(name).gguf")
            _write_gguf(path; metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (Int(b.type), [length(values)], hex2bytes(String(b.bytes)))))
            f = load_gguf(path)
            @testset "$(name)" begin
                @test f.tensor_types["t"] == String(name)
                @test f.tensors["t"] == values
            end
        end
    end
end

@testset "BF16 and F64 decode from their bits" begin
    mktempdir() do dir
        x = Float32[1.5, -2.25, 3.0f-3, 65504]
        bits = UInt16.(reinterpret(UInt32, x) .>> 16)
        path = joinpath(dir, "bf16.gguf")
        _write_gguf(path; metadata=Pair{String,Tuple{Int,Any}}[],
            tensors=Dict("t" => (30, [4], collect(reinterpret(UInt8, bits)))))
        @test load_gguf(path).tensors["t"] == reinterpret(Float32, UInt32.(bits) .<< 16)
        path = joinpath(dir, "f64.gguf")
        y = [1 / 3, -2.0]
        _write_gguf(path; metadata=Pair{String,Tuple{Int,Any}}[],
            tensors=Dict("t" => (28, [2], collect(reinterpret(UInt8, y)))))
        f = load_gguf(path)
        @test f.tensor_types["t"] == "F64" && f.tensors["t"] == Float32.(y)
    end
end

# llama.cpp's names for HF's, as its converter writes them.
function _gguf_name(hf, arch)
    hf == "model.embed_tokens.weight" && return "token_embd.weight"
    hf == "model.norm.weight" && return "output_norm.weight"
    hf == "lm_head.weight" && return "output.weight"
    m = match(r"^model\.layers\.(\d+)\.(.+)\.(weight|bias)$", hf)
    table = Dict(
        "input_layernorm" => "attn_norm", "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k", "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output", "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm", "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up", "mlp.down_proj" => "ffn_down",
        "post_attention_layernorm" => arch == "gemma2" ? "post_attention_norm" : "ffn_norm",
        "pre_feedforward_layernorm" => "ffn_norm",
        "post_feedforward_layernorm" => "post_ffw_norm",
    )
    return "blk.$(m[1]).$(table[m[2]]).$(m[3])"
end

# The converter's rotary reordering for llama: HF output `a * half + b` of each
# head moves to `2b + a`.
function _permute_rotary(w, heads)
    head_dim = size(w, 1) ÷ heads
    half = head_dim ÷ 2
    out = similar(w)
    for h in 0:(heads - 1), a in 0:1, b in 0:(half - 1)
        out[h * head_dim + 2b + a + 1, :] = w[h * head_dim + a * half + b + 1, :]
    end
    return out
end

# Fill every mapped weight of `lm` with random values, then write the model as
# llama.cpp would: F32 tensors, GGML's (in, out) layout, its names and metadata.
function _write_model_gguf(path, lm, cfg, mapping, arch; extra_meta=[], extra_tensors=Dict())
    rng = Xoshiro(31)
    tensors = Dict{String,Tuple{Int,Vector{Int},Vector{UInt8}}}()
    for (key, (p, transform)) in mapping
        a = _resolve(lm, p)
        a .= endswith(key, "norm.weight") ? 0.1f0 .* randn(rng, Float32, size(a)) .+ (arch == "gemma2" ? 0 : 1) :
            0.2f0 .* randn(rng, Float32, size(a))
        hf = transform === :transpose ? permutedims(a, (2, 1)) : copy(a)
        arch == "gemma2" && endswith(key, "norm.weight") && (hf = hf .+ 1)
        if arch == "llama" && occursin(r"[qk]_proj\.weight$", key)
            hf = _permute_rotary(hf, endswith(key, "q_proj.weight") ? cfg.num_attention_heads : cfg.num_key_value_heads)
        end
        ggml = ndims(hf) == 2 ? permutedims(hf, (2, 1)) : hf
        tensors[_gguf_name(key, arch)] = (0, collect(size(ggml)), collect(reinterpret(UInt8, vec(ggml))))
    end
    cfg.tie_word_embeddings && (lm.lm_head.weight .= permutedims(lm.model.embed_tokens.weight, (2, 1)))
    merge!(tensors, extra_tensors)
    meta = Pair{String,Tuple{Int,Any}}[
        "general.architecture" => (8, arch),
        "$(arch).embedding_length" => (4, cfg.hidden_size),
        "$(arch).feed_forward_length" => (4, cfg.intermediate_size),
        "$(arch).block_count" => (4, cfg.num_hidden_layers),
        "$(arch).attention.head_count" => (4, cfg.num_attention_heads),
        "$(arch).attention.head_count_kv" => (4, cfg.num_key_value_heads),
        "$(arch).attention.key_length" => (4, cfg.head_dim),
        "$(arch).context_length" => (4, cfg.max_position_embeddings),
        "$(arch).attention.layer_norm_rms_epsilon" => (6, cfg.rms_norm_eps),
        "$(arch).rope.freq_base" => (6, cfg.rope_theta),
        extra_meta...,
    ]
    return _write_gguf(path; metadata=meta, tensors=tensors)
end

@testset verbose = true "a GGUF file builds the model it was written from" begin
    common = Dict{Symbol,Any}(
        :vocab_size => 64, :hidden_size => 32, :intermediate_size => 48,
        :num_hidden_layers => 2, :num_attention_heads => 4, :num_key_value_heads => 2,
        :head_dim => 8, :max_position_embeddings => 64, :rope_theta => 10_000.0,
        :rms_norm_eps => 1.0e-6,
    )
    ids = reshape([1, 5, 9, 2, 40, 7], :, 1)
    cases = (
        ("llama", LlamaForCausalLM, config_from_json(LlamaConfig, merge(common, Dict(:tie_word_embeddings => false))),
            llama_state_dict_map, []),
        ("qwen3", QwenForCausalLM, config_from_json(QwenConfig, merge(common, Dict(:model_type => "qwen3", :tie_word_embeddings => true))),
            qwen_state_dict_map, []),
        ("qwen2", QwenForCausalLM, config_from_json(QwenConfig, merge(common, Dict(:model_type => "qwen2", :tie_word_embeddings => true))),
            qwen_state_dict_map, []),
        ("gemma2", GemmaForCausalLM,
            config_from_json(GemmaConfig, merge(common, Dict(:sliding_window => 4, :attn_logit_softcapping => 50.0,
                :final_logit_softcapping => 30.0, :query_pre_attn_scalar => 8))),
            gemma_state_dict_map,
            ["gemma2.attention.sliding_window" => (4, 4), "gemma2.attn_logit_softcapping" => (6, 50.0),
             "gemma2.final_logit_softcapping" => (6, 30.0)]),
    )
    mktempdir() do dir
        for (arch, T, cfg, map_fn, meta) in cases
            @testset "$(arch)" begin
                lm = T(cfg)
                path = _write_model_gguf(joinpath(dir, "$(arch).gguf"), lm, cfg, map_fn(cfg), arch; extra_meta=meta)
                loaded = load_gguf_model(path)
                @test loaded isa T && loaded.config == cfg
                @test loaded.model.layers[1].self_attn.wq.weight isa GGUFMatrix
                @test loaded.model.embed_tokens.weight isa GGUFEmbedding
                @test loaded(ids) ≈ lm(ids) rtol = 1.0e-5
                dense = load_gguf_model(path; packed=false)
                @test dense.model.layers[1].self_attn.wq.weight isa Matrix{Float32}
                @test dense(ids) ≈ lm(ids) rtol = 1.0e-5
                arch == "llama" && @test HuggingFaceTransformers.load(path)(ids) ≈ lm(ids) rtol = 1.0e-5
            end
        end

        cfg = cases[1][3]
        lm = LlamaForCausalLM(cfg)
        extra(name) = Dict(name => (0, [4], zeros(UInt8, 16)))
        path = _write_model_gguf(joinpath(dir, "extra.gguf"), lm, cfg, llama_state_dict_map(cfg), "llama";
            extra_tensors=extra("blk.0.attn_unknown.weight"))
        @test_throws ArgumentError load_gguf_model(path)
        path = _write_model_gguf(joinpath(dir, "freqs.gguf"), lm, cfg, llama_state_dict_map(cfg), "llama";
            extra_tensors=extra("rope_freqs.weight"))
        @test_throws ArgumentError load_gguf_model(path)
        path = joinpath(dir, "phi.gguf")
        _write_gguf(path; metadata=["general.architecture" => (8, "phi2")],
            tensors=Dict{String,Tuple{Int,Vector{Int},Vector{UInt8}}}())
        @test_throws ArgumentError load_gguf_model(path)
    end
end

@testset verbose = true "packed GGUF matrices multiply as their decoded values do" begin
    # Rows built from the real blocks of each type, two block orders alternating, so
    # the fused kernels and the decode-then-multiply ones meet the same bytes the
    # reference decoder reads exactly.
    blocks = JSON3.read(read(joinpath(@__DIR__, "fixtures", "gguf_quant_blocks.json"), String))
    rng = Xoshiro(41)
    for (name, b) in pairs(blocks)
        bytes = hex2bytes(String(b.bytes))
        half = length(bytes) ÷ 2
        rows = [isodd(r) ? bytes : [bytes[(half + 1):end]; bytes[1:half]] for r in 1:9]
        n_in = length(b.values)
        m = GGUFMatrix(Int(b.type), reduce(vcat, rows), 9, n_in)
        dense = reduce(hcat, [HuggingFaceTransformers.GGUF._dequantize!(Vector{Float32}(undef, n_in), Val(Int(b.type)), r) for r in rows])'
        @testset "$(name)" begin
            @test size(m) == (9, n_in) && m[2, n_in] == dense[2, n_in]
            @test all(n -> (x = randn(rng, Float32, n_in, n); m * x ≈ dense * x), (1, 2, 5, 9))
            @test m * ones(Float32, n_in) ≈ dense * ones(Float32, n_in)
            @test GGUFEmbedding(m)[:, [3, 1]] == permutedims(dense[[3, 1], :], (2, 1))
        end
    end
    @test_throws DimensionMismatch GGUFMatrix(12, zeros(UInt8, 144), 1, 128)
    @test_throws DimensionMismatch GGUFMatrix(12, zeros(UInt8, 140), 1, 256)
end
