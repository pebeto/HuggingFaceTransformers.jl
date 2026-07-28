using Aqua
using HuggingFaceTransformers

Aqua.test_all(
    HuggingFaceTransformers;
    ambiguities=false,
    # Several declared deps (Flux, NNlib, SafeTensors, ...) are wired up
    # across multiple submodules; re-enable once every declared dep is
    # `using`-ed somewhere.
    stale_deps=false,
)
