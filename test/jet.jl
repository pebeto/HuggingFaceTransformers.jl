using JET
using Allspark

JET.test_package(
    Allspark;
    target_modules=(
        Allspark,
        Allspark.HFHub,
        Allspark.Tokenizers,
        Allspark.Layers,
        Allspark.Models,
        Allspark.Generation,
    ),
)
