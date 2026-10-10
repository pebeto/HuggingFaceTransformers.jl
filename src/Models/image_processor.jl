const _IMAGENET_STANDARD = (0.5, 0.5, 0.5)
const _IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
const _IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
const _OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
const _OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

"""
    ImageProcessor(; size = (224, 224), shortest_edge = nothing,
                   resample = :bilinear, crop_size = nothing,
                   rescale_factor = 1 / 255, image_mean = (0.5, 0.5, 0.5),
                   image_std = (0.5, 0.5, 0.5))

Turns decoded images into the `pixel_values` that [`ViTModel`](@ref),
[`SiglipModel`](@ref), [`Dinov2Model`](@ref) and LLaVA's vision tower take,
matching HF's default (torchvision) image processors. The defaults are ViT's; use
[`load_image_processor`](@ref) to read a checkpoint's own settings.

Call it on a `(channels, height, width)` array to get a `(channels, height,
width, 1)` `Float32` array, or on a vector of images for a batch along the fourth
dimension. With ColorTypes loaded (FileIO and Images.jl load it), a matrix of
colors works too: gray is expanded to RGB and transparency is composited onto
white, as HF does.

The steps run in HF's order, and each is skipped when its setting is `nothing`:

1. Resize to `size = (height, width)`, or so the shorter side equals
   `shortest_edge` with the aspect ratio kept. `resample` is `:bilinear` or
   `:bicubic`, antialiased on a downscale.
2. Center crop to `crop_size = (height, width)`, padding with zeros first if
   the image is smaller.
3. Rescale by `rescale_factor` and normalize each channel with
   `image_mean` and `image_std`.

`UInt8` input is resized with torchvision's fixed-point uint8 kernel and the
result matches HF's `pixel_values` exactly. A float array is resized in `Float32`
and taken at face value, as HF does, so pixel values belong in `[0, 255]` when
`rescale_factor` is set.
"""
struct ImageProcessor
    size::Union{Nothing,Tuple{Int,Int}}
    shortest_edge::Union{Nothing,Int}
    resample::Symbol
    crop_size::Union{Nothing,Tuple{Int,Int}}
    rescale_factor::Union{Nothing,Float64}
    image_mean::Union{Nothing,Vector{Float64}}
    image_std::Union{Nothing,Vector{Float64}}

    function ImageProcessor(size, shortest_edge, resample, crop_size, rescale_factor, mean, std)
        isnothing(size) || isnothing(shortest_edge) ||
            throw(ArgumentError("set either size or shortest_edge, not both"))
        resample in (:bilinear, :bicubic) ||
            throw(ArgumentError("resample must be :bilinear or :bicubic, got :$(resample)"))
        isnothing(mean) == isnothing(std) ||
            throw(ArgumentError("image_mean and image_std must be given together"))
        if !isnothing(mean)
            length(mean) == length(std) ||
                throw(ArgumentError("image_mean and image_std differ in length"))
            all(!iszero, std) || throw(ArgumentError("image_std must be nonzero"))
        end
        for dims in (size, crop_size, shortest_edge)
            isnothing(dims) || all(>(0), dims) ||
                throw(ArgumentError("image sizes must be positive, got $(dims)"))
        end
        return new(size, shortest_edge, resample, crop_size, rescale_factor, mean, std)
    end
end

function ImageProcessor(;
    shortest_edge=nothing,
    size=isnothing(shortest_edge) ? (224, 224) : nothing,
    resample::Symbol=:bilinear,
    crop_size=nothing,
    rescale_factor=1 / 255,
    image_mean=_IMAGENET_STANDARD,
    image_std=_IMAGENET_STANDARD,
)
    return ImageProcessor(
        isnothing(size) ? nothing : Tuple{Int,Int}(size),
        isnothing(shortest_edge) ? nothing : Int(shortest_edge),
        resample,
        isnothing(crop_size) ? nothing : Tuple{Int,Int}(crop_size),
        isnothing(rescale_factor) ? nothing : Float64(rescale_factor),
        isnothing(image_mean) ? nothing : collect(Float64, image_mean),
        isnothing(image_std) ? nothing : collect(Float64, image_std),
    )
end

function _bicubic_kernel(x::T) where {T}
    # Keys' cubic with a = -0.5, as both PyTorch and Pillow use.
    a = T(-0.5)
    x = abs(x)
    x < 1 && return ((a + 2) * x - (a + 3)) * x * x + 1
    x < 2 && return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return zero(T)
end

_bilinear_kernel(x::T) where {T} = max(zero(T), one(T) - abs(x))

# Antialiased resampling weights along one axis, computed as PyTorch's
# `_compute_weights_aa` does: on a downscale the kernel widens by the scale factor
# so every input pixel contributes, and each output's weights are normalised to
# sum to one. Returns 0-based window starts and one weight vector per output.
function _resample_weights(::Type{T}, n_in::Int, n_out::Int, resample::Symbol) where {T}
    kernel = resample === :bicubic ? _bicubic_kernel : _bilinear_kernel
    half_width = resample === :bicubic ? 2 : 1
    scale = T(n_in) / T(n_out)
    support = scale >= 1 ? half_width * scale : T(half_width)
    invscale = scale >= 1 ? 1 / scale : one(T)
    max_taps = ceil(Int, support) * 2 + 1

    starts = Vector{Int}(undef, n_out)
    weights = Vector{Vector{T}}(undef, n_out)
    for i in 0:(n_out - 1)
        center = scale * (i + T(0.5))
        lo = max(trunc(Int, center - support + T(0.5)), 0)
        taps = clamp(min(trunc(Int, center + support + T(0.5)), n_in) - lo, 0, max_taps)
        w = T[kernel((j + lo - center + T(0.5)) * invscale) for j in 0:(taps - 1)]
        total = sum(w; init=zero(T))
        iszero(total) || (w ./= total)
        starts[i + 1] = lo
        weights[i + 1] = w
    end
    return starts, weights
end

# torchvision resizes uint8 images in fixed point: weights become integers scaled
# by 2^precision, with the precision as high as int16 allows for the largest
# weight. Rounding is half away from zero, as in PyTorch.
function _fixed_point_weights(weights::Vector{Vector{Float64}})
    wmax = maximum(w -> maximum(w; init=0.0), weights)
    precision = 0
    while precision < 22 && trunc(Int, 0.5 + wmax * (1 << (precision + 1))) < (1 << 15)
        precision += 1
    end
    scale = Float64(1 << precision)
    fixed = [[trunc(Int, v < 0 ? v * scale - 0.5 : v * scale + 0.5) for v in w] for w in weights]
    return fixed, precision
end

_shift(I::CartesianIndex, dim::Int, k::Int) = CartesianIndex(Base.setindex(Tuple(I), k, dim))

# One separable pass along `dim` of a (channels, height, width) image.
function _resample_dim(img::AbstractArray{UInt8,3}, dim::Int, n_out::Int, resample::Symbol)
    starts, weights = _resample_weights(Float64, size(img, dim), n_out, resample)
    fixed, precision = _fixed_point_weights(weights)
    rounding = 1 << (precision - 1)
    out = Array{UInt8,3}(undef, Base.setindex(size(img), n_out, dim))
    for I in CartesianIndices(out)
        i = I[dim]
        acc = rounding
        for (j, w) in enumerate(fixed[i])
            acc += Int(img[_shift(I, dim, starts[i] + j)]) * w
        end
        out[I] = clamp(acc >> precision, 0, 255)
    end
    return out
end

function _resample_dim(img::AbstractArray{Float32,3}, dim::Int, n_out::Int, resample::Symbol)
    starts, weights = _resample_weights(Float32, size(img, dim), n_out, resample)
    out = Array{Float32,3}(undef, Base.setindex(size(img), n_out, dim))
    for I in CartesianIndices(out)
        i = I[dim]
        acc = 0.0f0
        for (j, w) in enumerate(weights[i])
            acc += img[_shift(I, dim, starts[i] + j)] * w
        end
        out[I] = acc
    end
    return out
end

# Width first, then height, as torchvision does. A pass whose size is unchanged
# is skipped; for uint8 the rounding between passes is part of the result.
function _resize(img::AbstractArray{<:Real,3}, height::Int, width::Int, resample::Symbol)
    if size(img, 3) != width
        img = _resample_dim(img, 3, width, resample)
    end
    if size(img, 2) != height
        img = _resample_dim(img, 2, height, resample)
    end
    return img
end

# HF's `get_resize_output_image_size` with `default_to_square=False`.
function _shortest_edge_size(height::Int, width::Int, target::Int)
    short, long = width <= height ? (width, height) : (height, width)
    new_long = trunc(Int, target * long / short)
    return width <= height ? (new_long, target) : (target, new_long)
end

function _center_crop(img::AbstractArray{T,3}, crop_height::Int, crop_width::Int) where {T}
    c, h, w = size(img)
    if crop_height > h || crop_width > w
        # Zero padding first, with the odd pixel going to the bottom and right.
        top, left = max(crop_height - h, 0) ÷ 2, max(crop_width - w, 0) ÷ 2
        padded = zeros(T, c, max(h, crop_height), max(w, crop_width))
        padded[:, (top + 1):(top + h), (left + 1):(left + w)] .= img
        img = padded
        _, h, w = size(img)
    end
    top, left = (h - crop_height) ÷ 2, (w - crop_width) ÷ 2
    return img[:, (top + 1):(top + crop_height), (left + 1):(left + crop_width)]
end

function _rescale_and_normalize(img::AbstractArray{<:Real,3}, proc::ImageProcessor)
    x = Float32.(img)
    if !isnothing(proc.image_mean)
        size(x, 1) == length(proc.image_mean) || throw(
            ArgumentError(
                "image has $(size(x, 1)) channels but image_mean has $(length(proc.image_mean))",
            ),
        )
        # HF folds the rescale into the statistics, in Float32:
        # (x - mean / factor) / (std / factor).
        factor = Float32(isnothing(proc.rescale_factor) ? 1.0 : 1.0 / proc.rescale_factor)
        mean = Float32.(proc.image_mean) .* factor
        std = Float32.(proc.image_std) .* factor
        x .= (x .- mean) ./ std
    elseif !isnothing(proc.rescale_factor)
        x .*= Float32(proc.rescale_factor)
    end
    return x
end

function (proc::ImageProcessor)(image::AbstractArray{<:Real,3})
    img = image isa AbstractArray{UInt8} ? image : Float32.(image)
    if img isa AbstractArray{Float32} && !isnothing(proc.rescale_factor) &&
        all(v -> 0 <= v <= 1, img)
        @warn "Float image already lies in [0, 1] but will be rescaled by " *
            "$(proc.rescale_factor); pass values in [0, 255] or a UInt8 array." maxlog = 1
    end

    _, h, w = size(img)
    if !isnothing(proc.size)
        img = _resize(img, proc.size..., proc.resample)
    elseif !isnothing(proc.shortest_edge)
        img = _resize(img, _shortest_edge_size(h, w, proc.shortest_edge)..., proc.resample)
    end
    isnothing(proc.crop_size) || (img = _center_crop(img, proc.crop_size...))
    x = _rescale_and_normalize(img, proc)
    return reshape(x, size(x)..., 1)
end

function (proc::ImageProcessor)(images::AbstractVector)
    isempty(images) && throw(ArgumentError("images must be non-empty"))
    batch = [proc(img) for img in images]
    all(x -> size(x) == size(first(batch)), batch) || throw(
        ArgumentError(
            "processed images differ in size; set size or crop_size to batch them",
        ),
    )
    return cat(batch...; dims=4)
end

# Class-level defaults of the HF processors mirrored here, for keys a config
# omits. `square` is HF's `default_to_square`: how a bare integer `size` reads.
const _IMAGE_PROCESSOR_DEFAULTS = Dict(
    "ViTImageProcessor" => (;
        square=true, size=224, resample=2, center_crop=false, crop_size=nothing,
        mean=_IMAGENET_STANDARD, std=_IMAGENET_STANDARD,
    ),
    "SiglipImageProcessor" => (;
        square=false, size=(224, 224), resample=3, center_crop=false, crop_size=nothing,
        mean=_IMAGENET_STANDARD, std=_IMAGENET_STANDARD,
    ),
    "BitImageProcessor" => (;
        square=false, size=224, resample=3, center_crop=true, crop_size=(224, 224),
        mean=_OPENAI_CLIP_MEAN, std=_OPENAI_CLIP_STD,
    ),
    "CLIPImageProcessor" => (;
        square=false, size=224, resample=3, center_crop=true, crop_size=(224, 224),
        mean=_OPENAI_CLIP_MEAN, std=_OPENAI_CLIP_STD,
    ),
)

# ViT's own `preprocessor_config.json` names no processor class, so HF falls back
# on the model type in `config.json`.
const _IMAGE_PROCESSOR_BY_MODEL_TYPE = Dict(
    "vit" => "ViTImageProcessor",
    "siglip" => "SiglipImageProcessor",
    "dinov2" => "BitImageProcessor",
    "bit" => "BitImageProcessor",
    "clip" => "CLIPImageProcessor",
)

function _image_processor_class(raw, dir::AbstractString)
    kind = _hf(raw, :image_processor_type, nothing)
    if isnothing(kind)
        legacy = _hf(raw, :feature_extractor_type, nothing)
        isnothing(legacy) || (kind = replace(legacy, "FeatureExtractor" => "ImageProcessor"))
    end
    if isnothing(kind)
        config = joinpath(dir, "config.json")
        model_type = isfile(config) ? _hf_str(read_config(config), :model_type, "") : ""
        kind = get(_IMAGE_PROCESSOR_BY_MODEL_TYPE, model_type, nothing)
        isnothing(kind) && throw(
            ArgumentError(
                "preprocessor_config.json names no image processor and model type " *
                "`$(model_type)` has no default",
            ),
        )
    end
    # The fast (torchvision) variants carried a suffix before becoming the default.
    kind = String(chopsuffix(kind, "Fast"))
    haskey(_IMAGE_PROCESSOR_DEFAULTS, kind) || throw(
        ArgumentError(
            "unsupported image processor `$(kind)`; supported: " *
            join(sort!(collect(keys(_IMAGE_PROCESSOR_DEFAULTS))), ", "),
        ),
    )
    return kind
end

# A size entry is an integer or a dict, as HF's `get_size_dict` accepts. Returns
# `(height_width, shortest_edge)` with exactly one of the two set.
function _parse_size(value, square::Bool)
    if value isa Integer
        return square ? ((Int(value), Int(value)), nothing) : (nothing, Int(value))
    elseif value isa Tuple
        return (value, nothing)
    end
    height, width = _hf(value, :height, nothing), _hf(value, :width, nothing)
    shortest = _hf(value, :shortest_edge, nothing)
    others = filter(k -> !(k in (:height, :width, :shortest_edge)) && !isnothing(value[k]), keys(value))
    if !isnothing(height) && !isnothing(width) && isnothing(shortest) && isempty(others)
        return ((Int(height), Int(width)), nothing)
    elseif !isnothing(shortest) && isnothing(height) && isnothing(width) && isempty(others)
        return (nothing, Int(shortest))
    end
    throw(ArgumentError("unsupported image size $(JSON3.write(value))"))
end

const _PIL_RESAMPLE = Dict(2 => :bilinear, 3 => :bicubic)

# HF accepts one number for all three channels.
_channel_stats(v::Real) = fill(Float64(v), 3)
_channel_stats(v) = collect(Float64, v)

"""
    load_image_processor(path) -> ImageProcessor

Build an [`ImageProcessor`](@ref) from a `preprocessor_config.json`, given the
file or a snapshot directory containing it. Keys the file omits take the defaults
of the HF class it names: `ViTImageProcessor`, `SiglipImageProcessor`,
`BitImageProcessor` (DINOv2) or `CLIPImageProcessor` (CLIP and LLaVA-1.5). Other
classes raise, as do settings these processors would honour that are not
implemented here, such as Lanczos resampling or a `longest_edge` size.
"""
function load_image_processor(path::AbstractString)
    file = isdir(path) ? joinpath(path, "preprocessor_config.json") : path
    isfile(file) || throw(ArgumentError("preprocessor_config.json not found at $(file)"))
    raw = JSON3.read(read(file, String))
    defaults = _IMAGE_PROCESSOR_DEFAULTS[_image_processor_class(raw, dirname(file))]

    size, shortest_edge = nothing, nothing
    if _hf_bool(raw, :do_resize, true)
        size, shortest_edge = _parse_size(_hf(raw, :size, defaults.size), defaults.square)
    end
    resample_id = _hf_int(raw, :resample, defaults.resample)
    resample = get(_PIL_RESAMPLE, resample_id, nothing)
    isnothing(resample) && throw(
        ArgumentError("unsupported resample $(resample_id); only bilinear (2) and bicubic (3)"),
    )
    crop_size = nothing
    if _hf_bool(raw, :do_center_crop, defaults.center_crop)
        crop = _hf(raw, :crop_size, defaults.crop_size)
        isnothing(crop) && throw(ArgumentError("do_center_crop is set but crop_size is not"))
        crop_size, _ = _parse_size(crop, true)
    end
    rescale_factor =
        _hf_bool(raw, :do_rescale, true) ? _hf_float(raw, :rescale_factor, 1 / 255) : nothing
    mean, std = nothing, nothing
    if _hf_bool(raw, :do_normalize, true)
        mean = _channel_stats(_hf(raw, :image_mean, defaults.mean))
        std = _channel_stats(_hf(raw, :image_std, defaults.std))
    end
    return ImageProcessor(;
        size, shortest_edge, resample, crop_size, rescale_factor,
        image_mean=mean, image_std=std,
    )
end
