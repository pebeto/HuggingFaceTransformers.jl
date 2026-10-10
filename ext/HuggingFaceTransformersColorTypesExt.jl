"""
    HuggingFaceTransformersColorTypesExt

Loaded when `ColorTypes` is, which FileIO and Images.jl both pull in. Lets an
[`HuggingFaceTransformers.Models.ImageProcessor`](@ref) take a decoded image, a
matrix of colors, directly.
"""
module HuggingFaceTransformersColorTypesExt

using HuggingFaceTransformers.Models: ImageProcessor
using ColorTypes:
    Colorant, AbstractRGB, AbstractGray, TransparentColor, red, green, blue, gray, alpha, color

# An 8-bit channel round-trips exactly: N0f8 stores k/255, and scaling back by
# 255 lands within an ulp of k.
_byte(v::Real) = round(UInt8, clamp(Float64(v), 0.0, 1.0) * 255)

_rgb(c::AbstractRGB) = (red(c), green(c), blue(c))
_rgb(c::AbstractGray) = (gray(c), gray(c), gray(c))
function _rgb(c::TransparentColor)
    # HF's `convert_to_rgb` composites onto an opaque white background.
    a = Float64(alpha(c))
    return map(v -> Float64(v) * a + (1.0 - a), _rgb(color(c)))
end
_rgb(c::Colorant) = throw(ArgumentError("convert $(typeof(c)) images to RGB first"))

# (height, width) colors -> (3, height, width) bytes, the layout ImageProcessor
# takes and the one `channelview` gives.
function _channels(img::AbstractMatrix{<:Colorant})
    out = Array{UInt8,3}(undef, 3, size(img)...)
    for I in CartesianIndices(img)
        r, g, b = _rgb(img[I])
        out[1, I], out[2, I], out[3, I] = _byte(r), _byte(g), _byte(b)
    end
    return out
end

(proc::ImageProcessor)(img::AbstractMatrix{<:Colorant}) = proc(_channels(img))

end # module
