using JET
using HuggingFaceTransformers

JET.test_package(
    HuggingFaceTransformers;
    target_modules=(
        HuggingFaceTransformers,
        HuggingFaceTransformers.HFHub,
        HuggingFaceTransformers.Tokenizers,
        HuggingFaceTransformers.Layers,
        HuggingFaceTransformers.Models,
        HuggingFaceTransformers.Generation,
    ),
)
