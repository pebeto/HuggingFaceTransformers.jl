using Documenter
using HuggingFaceTransformers

# Two published versions, both tracking a branch rather than a tag:
#
#   dev branch    -> /dev/
#   master branch -> /stable/
#
# `deploydocs` only deploys when the checked-out branch equals `devbranch`, so
# the pair is chosen from the branch CI is running on. Locally both are unset
# and `deploydocs` becomes a no-op.
const BRANCH = get(ENV, "GITHUB_REF_NAME", "")
const DEVBRANCH, DEVURL = BRANCH == "master" ? ("master", "stable") : ("dev", "dev")

makedocs(;
    modules=[
        HuggingFaceTransformers,
        HuggingFaceTransformers.HFHub,
        HuggingFaceTransformers.GGUF,
        HuggingFaceTransformers.Tokenizers,
        HuggingFaceTransformers.Layers,
        HuggingFaceTransformers.Models,
        HuggingFaceTransformers.Generation,
    ],
    authors="Jose Esparza <joseesparzadc@gmail.com>",
    repo=Remotes.GitHub("pebeto", "HuggingFaceTransformers.jl"),
    sitename="HuggingFaceTransformers.jl",
    format=Documenter.HTML(;
        canonical="https://pebeto.github.io/HuggingFaceTransformers.jl/stable/",
        edit_link=DEVBRANCH,
        assets=String[],
        size_threshold=500_000,
    ),
    pages=[
        "Home" => "index.md",
        "Guide" => [
            "guide/loading.md",
            "guide/generation.md",
            "guide/tokenizers.md",
            "guide/performance.md",
            "guide/finetuning.md",
        ],
        "API" => [
            "api/hfhub.md",
            "api/tokenizers.md",
            "api/layers.md",
            "Models" => [
                "api/models/loading.md",
                "api/models/decoders.md",
                "api/models/encoders.md",
                "api/models/vision.md",
                "api/models/adapters.md",
            ],
            "api/generation.md",
            "api/gguf.md",
            "api/internals.md",
        ],
    ],
    # Every exported symbol must appear in an @docs block, and a broken
    # cross-reference fails the build rather than shipping a dead link.
    checkdocs=:exports,
    warnonly=false,
)

deploydocs(;
    repo="github.com/pebeto/HuggingFaceTransformers.jl",
    devbranch=DEVBRANCH,
    devurl=DEVURL,
    versions=["stable" => "stable", "dev" => "dev"],
    push_preview=true,
)
