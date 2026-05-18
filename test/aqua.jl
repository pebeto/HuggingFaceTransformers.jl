using Aqua
using Allspark

Aqua.test_all(
    Allspark;
    ambiguities=false,
    # Several declared deps (Flux, NNlib, SafeTensors, ...) are wired up
    # across multiple submodules; re-enable once every declared dep is
    # `using`-ed somewhere.
    stale_deps=false,
)
