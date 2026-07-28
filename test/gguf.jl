using Test
using HuggingFaceTransformers.GGUF
using HuggingFaceTransformers.GGUF: load_gguf, GGUFFile

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
            path = joinpath(dir, "q6k.gguf")
            # Q6_K (14) with bogus bytes; the reader should refuse before
            # touching them.
            _write_gguf(
                path;
                metadata=Pair{String,Tuple{Int,Any}}[],
                tensors=Dict("t" => (14, [256], zeros(UInt8, 256))),
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
