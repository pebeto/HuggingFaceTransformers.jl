using Test
using Base64
using JSON3
using ColorTypes: RGB, RGBA, Gray, HSV
using FixedPointNumbers: N0f8
using HuggingFaceTransformers.Models:
    ImageProcessor, load_image_processor, ViTConfig, ViTForImageClassification

isdefined(@__MODULE__, :image_pattern) || include("parity_inputs.jl")

const _IMAGE_PROCESSOR_FIXTURE = joinpath(@__DIR__, "fixtures", "image_processor_parity.json")

function _processor_from_config(config; model_type=nothing)
    dir = mktempdir()
    write(joinpath(dir, "preprocessor_config.json"), JSON3.write(config))
    isnothing(model_type) ||
        write(joinpath(dir, "config.json"), JSON3.write(Dict("model_type" => model_type)))
    return load_image_processor(dir)
end

@testset verbose = true "matches HF's image processors" begin
    # No weights and no download: each case is a small preprocessor config applied
    # to a regenerated image, so this runs in the default suite.
    if !isfile(_IMAGE_PROCESSOR_FIXTURE)
        @info "Skipping image processor parity: run " *
            "`python3 test/fixtures/record_image_processor_parity.py`"
    else
        for case in JSON3.read(read(_IMAGE_PROCESSOR_FIXTURE, String)).cases
            @testset "$(case.name)" begin
                proc = _processor_from_config(case.config)
                image = image_pattern(case.height, case.width)
                case.dtype == "float32" && (image = Float32.(image))
                c, h, w = Int.(case.shape)
                # Row-major (C, H, W) bytes read column-major give (W, H, C).
                ref = permutedims(
                    reshape(reinterpret(Float32, base64decode(case.pixel_values)), w, h, c),
                    (3, 2, 1),
                )
                pixels = proc(image)
                @test size(pixels) == (c, h, w, 1)
                @test maximum(abs.(pixels[:, :, :, 1] .- ref)) < case.tolerance
            end
        end
    end
end

@testset verbose = true "ImageProcessor" begin
    image = image_pattern(67, 91)

    @testset "a resize to the input size is the identity" begin
        proc = ImageProcessor(; size=(67, 91), image_mean=nothing, image_std=nothing,
            rescale_factor=nothing)
        @test proc(image)[:, :, :, 1] == Float32.(image)
    end

    @testset "uniform images stay uniform" begin
        # The weights of each output pixel sum to one, in fixed point too.
        flat = fill(UInt8(173), 3, 41, 29)
        for resample in (:bilinear, :bicubic), dims in ((16, 16), (80, 50))
            proc = ImageProcessor(; size=dims, resample, image_mean=nothing, image_std=nothing,
                rescale_factor=nothing)
            @test all(==(173.0f0), proc(flat))
        end
    end

    @testset "shortest edge keeps the aspect ratio" begin
        proc = ImageProcessor(; shortest_edge=20, image_mean=nothing, image_std=nothing)
        @test size(proc(image)) == (3, 20, 27, 1)             # 91 * 20 / 67 = 27.2
        @test size(proc(image_pattern(91, 67))) == (3, 27, 20, 1)
    end

    @testset "center crop pads small images with zeros" begin
        proc = ImageProcessor(; size=nothing, crop_size=(5, 4), image_mean=nothing,
            image_std=nothing, rescale_factor=nothing)
        px = proc(fill(UInt8(9), 1, 2, 3))[1, :, :, 1]
        # Rows: 3 of padding split 1 above, 2 below. Columns: 1 split 0 left, 1 right.
        @test px == Float32[0 0 0 0; 9 9 9 0; 9 9 9 0; 0 0 0 0; 0 0 0 0]
    end

    @testset "normalization folds the rescale in" begin
        proc = ImageProcessor(; size=nothing, image_mean=(0.25, 0.5, 0.75),
            image_std=(0.5, 0.25, 0.125))
        px = proc(fill(UInt8(255), 3, 1, 1))
        @test vec(px) ≈ [1.5, 2.0, 2.0]
    end

    @testset "batches stack along the fourth dimension" begin
        proc = ImageProcessor(; size=(16, 16))
        other = image_pattern(40, 30)          # portrait, unlike `image`
        batch = proc([image, other])
        @test size(batch) == (3, 16, 16, 2)
        @test batch[:, :, :, 2:2] == proc(other)
        @test_throws ArgumentError ImageProcessor(; shortest_edge=16)([image, other])
        @test_throws ArgumentError proc(Array{UInt8,3}[])
    end

    @testset "float input in [0, 1] warns" begin
        proc = ImageProcessor(; size=(8, 8))
        @test_logs (:warn, r"already lies in \[0, 1\]") proc(rand(Float32, 3, 10, 10))
    end

    @testset "invalid settings are refused" begin
        @test_throws ArgumentError ImageProcessor(; size=(16, 16), shortest_edge=16)
        @test_throws ArgumentError ImageProcessor(; resample=:lanczos)
        @test_throws ArgumentError ImageProcessor(; image_std=nothing)
        @test_throws ArgumentError ImageProcessor(; image_std=(0.5, 0.0, 0.5))
        @test_throws ArgumentError ImageProcessor(; crop_size=(0, 4))
        @test_throws ArgumentError ImageProcessor()(rand(UInt8, 4, 8, 8))   # 4 channels, 3 means
    end

    @testset "pixel values fit the model" begin
        cfg = ViTConfig(; image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, num_labels=5)
        out = ViTForImageClassification(cfg)(ImageProcessor(; size=(32, 32))(image))
        @test size(out) == (5, 1)
        @test all(isfinite, out)
    end
end

@testset verbose = true "color images" begin
    bytes = image_pattern(19, 23)
    colors = [RGB{N0f8}(reinterpret.(N0f8, bytes[:, h, w])...) for h in 1:19, w in 1:23]
    proc = ImageProcessor(; size=(8, 8))

    @test proc(colors) == proc(bytes)
    @test proc([colors, colors]) == proc([bytes, bytes])

    gray = [Gray{N0f8}(reinterpret(N0f8, bytes[1, h, w])) for h in 1:19, w in 1:23]
    @test proc(gray) == proc(repeat(bytes[1:1, :, :]; outer=(3, 1, 1)))

    # Transparency composites onto white, so a clear pixel reads as white whatever
    # color it carries.
    clear = fill(RGBA{N0f8}(0, 0, 0, 0), 4, 4)
    @test proc(clear) == proc(fill(0xff, 3, 4, 4))
    half = fill(RGBA{Float32}(0, 0, 0, 0.5), 4, 4)
    @test proc(half) == proc(fill(0x80, 3, 4, 4))           # 127.5 rounds to 128

    @test_throws ArgumentError proc(fill(HSV(0.0, 0.0, 0.0), 4, 4))
end

@testset verbose = true "load_image_processor" begin
    @testset "class defaults fill omitted keys" begin
        proc = _processor_from_config(Dict("image_processor_type" => "BitImageProcessor"))
        @test proc.shortest_edge == 224 && isnothing(proc.size)
        @test proc.crop_size == (224, 224)
        @test proc.resample === :bicubic
        @test proc.image_mean ≈ [0.48145466, 0.4578275, 0.40821073]

        proc = _processor_from_config(Dict("image_processor_type" => "SiglipImageProcessor"))
        @test proc.size == (224, 224) && isnothing(proc.crop_size)
        @test proc.resample === :bicubic
    end

    @testset "a bare integer size follows the class" begin
        square = _processor_from_config(Dict("image_processor_type" => "ViTImageProcessor", "size" => 32))
        @test square.size == (32, 32)
        edge = _processor_from_config(Dict("image_processor_type" => "CLIPImageProcessor", "size" => 32))
        @test edge.shortest_edge == 32
    end

    @testset "falls back on config.json's model type" begin
        # google/vit-base-patch16-224 ships exactly this: no class named.
        proc = _processor_from_config(
            Dict("do_normalize" => true, "do_resize" => true, "size" => 224,
                "image_mean" => [0.5, 0.5, 0.5], "image_std" => [0.5, 0.5, 0.5]);
            model_type="vit",
        )
        @test proc.size == (224, 224) && proc.resample === :bilinear
        @test_throws ArgumentError _processor_from_config(Dict("size" => 224); model_type="bert")
        @test_throws ArgumentError _processor_from_config(Dict("size" => 224))
    end

    @testset "disabled steps become nothing" begin
        proc = _processor_from_config(
            Dict("image_processor_type" => "BitImageProcessor", "do_resize" => false,
                "do_center_crop" => false, "do_rescale" => false, "do_normalize" => false),
        )
        @test isnothing(proc.size) && isnothing(proc.shortest_edge)
        @test isnothing(proc.crop_size) && isnothing(proc.rescale_factor)
        @test isnothing(proc.image_mean) && isnothing(proc.image_std)
    end

    @testset "unsupported settings raise" begin
        vit(extra) = merge(Dict("image_processor_type" => "ViTImageProcessor"), extra)
        @test_throws ArgumentError _processor_from_config(vit(Dict("resample" => 1)))
        @test_throws ArgumentError _processor_from_config(
            vit(Dict("size" => Dict("shortest_edge" => 224, "longest_edge" => 448))),
        )
        @test_throws ArgumentError _processor_from_config(
            Dict("image_processor_type" => "LlavaNextImageProcessor"),
        )
        @test_throws ArgumentError load_image_processor(mktempdir())
    end
end
