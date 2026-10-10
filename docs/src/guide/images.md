```@meta
CurrentModule = HuggingFaceTransformers
```

# Images

Vision models take `pixel_values`, a `(channels, height, width, batch)` `Float32`
array. [`Models.ImageProcessor`](@ref) builds it from a decoded image the way the
checkpoint's HF processor would, and [`Models.load_image_processor`](@ref) reads
its settings from the checkpoint's `preprocessor_config.json`:

```julia
using HuggingFaceTransformers
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Models: load_image_processor, read_config
import FileIO                       # plus ImageIO, or use JpegTurbo directly

dir = snapshot_download("google/vit-base-patch16-224")
model = load(dir)
processor = load_image_processor(dir)

pixel_values = processor(FileIO.load("cats.jpg"))     # (3, 224, 224, 1)
logits = model(pixel_values)                          # (1000, 1)
read_config(dir).id2label[Symbol(argmax(logits[:, 1]) - 1)]   # "Egyptian cat"
```

FileIO exports a `load` of its own, which is why it is imported rather than
used here: with both in scope, an unqualified `load` is ambiguous.

## What it accepts

A matrix of colors, which is what FileIO, JpegTurbo, and Images.jl return, works
once ColorTypes is loaded (they all load it). Gray images are expanded to RGB, and
transparent pixels are composited onto white, as HF's `convert_to_rgb` does.

A `(3, height, width)` `UInt8` array works without any extra package; from a
color image, ImageCore's `rawview(channelview(img))` gives one. A float array is
accepted too, but it is taken at face value as HF takes it, so with rescaling on
its values belong in `[0, 255]`, not `[0, 1]`. A warning fires when they look
like the latter.

A vector of images comes back as one batch along the fourth dimension, provided
the processor fixes the output size with `size` or `crop_size`.

## Matching HF

The steps are HF's: resize, center crop, then rescale and normalize. With `UInt8`
input the output equals HF's `pixel_values` bit for bit. Getting there means
reproducing torchvision's fixed-point uint8 resize. Its integer weights, and the
rounding between the horizontal and vertical passes, put about one pixel in seven
at least a level away from a floating-point resize rounded once. On a COCO photo,
decoded by JpegTurbo here and by Pillow in Python (the bytes agree), the ViT,
SigLIP, and DINOv2 processors reproduce HF's output exactly, and ViT's logits
agree to `1e-5`.

HF's own output depends on where it runs. On x86 CPUs with AVX2 and on ARM,
torchvision uses the fixed-point kernel. On a GPU, or an x86 CPU without AVX2, it
resizes in floating point and rounds, and some pixels land one level away.

## Supported processors

[`Models.load_image_processor`](@ref) reads the `ViTImageProcessor`,
`SiglipImageProcessor`, `BitImageProcessor` (DINOv2), and `CLIPImageProcessor`
(CLIP, llava-1.5) configs, filling omitted keys with that class's defaults.
ViT's own config names no class, so the model type in `config.json` decides, as
it does in HF. Anything else raises, as does a setting these classes would
honour but this package does not implement, such as Lanczos resampling or a
`longest_edge` size.

A processor can also be built by hand. DINOv2's, for example:

```julia
ImageProcessor(;
    shortest_edge=256, crop_size=(224, 224), resample=:bicubic,
    image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
)
```
