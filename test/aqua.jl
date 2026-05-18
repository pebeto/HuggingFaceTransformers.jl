using Aqua
using Allspark

Aqua.test_all(
    Allspark;
    ambiguities=false,
    # Phase 0: submodules are stubs, so declared deps are not yet `using`-ed.
    # Re-enable once Phase 1 starts pulling in Flux/NNlib/SafeTensors/etc.
    stale_deps=false,
)
