# Device selection shared by the GPU test files. `HFT_GPU_BACKEND` picks the target:
#
#   jlarrays (default)  GPUArrays reference backend. No hardware, runs in the
#                       default suite, and enforces the no-scalar-indexing
#                       discipline that real backends impose.
#   cuda                NVIDIA, needs CUDA.jl in the active environment.
#   amdgpu              AMD, needs AMDGPU.jl and a ROCm-supported card.
#   metal               Apple Silicon, needs Metal.jl. Float32 only.
#
# Loading a real backend also loads the matching HuggingFaceTransformers extension,
# which routes `sdpa` to the tiled `flash_sdpa`. That is a different summation
# order than the CPU materialized path, so device results differ by more than
# rounding; `DEV_RTOL` absorbs it and stays tight on the reference backend.

const GPU_BACKEND = lowercase(get(ENV, "HFT_GPU_BACKEND", "jlarrays"))

DeviceArray, dev, BACKEND_NAME = if GPU_BACKEND == "jlarrays"
    using JLArrays
    JLArrays.allowscalar(false)
    JLArray, JLArrays.jl, "JLArrays (reference)"
elseif GPU_BACKEND == "cuda"
    using CUDA
    CUDA.functional() ||
        error("HFT_GPU_BACKEND=cuda but CUDA.functional() is false; check the driver")
    CUDA.allowscalar(false)
    CuArray, CuArray, "CUDA"
elseif GPU_BACKEND == "amdgpu"
    using AMDGPU
    AMDGPU.functional() ||
        error("HFT_GPU_BACKEND=amdgpu but AMDGPU.functional() is false; check ROCm")
    AMDGPU.allowscalar(false)
    ROCArray, ROCArray, "AMDGPU"
elseif GPU_BACKEND == "metal"
    using Metal
    Metal.functional() ||
        error("HFT_GPU_BACKEND=metal but Metal.functional() is false")
    Metal.allowscalar(false)
    MtlArray, MtlArray, "Metal"
else
    error("Unknown HFT_GPU_BACKEND=$(GPU_BACKEND); use jlarrays|cuda|amdgpu|metal")
end

# Device results are close to the CPU reference, not bit-identical: the reference
# backend's matmul and the loop shim below accumulate in a different order than
# BLAS, so Float32's default relative tolerance is the honest bar. Real backends
# get more room because their `sdpa` is a different algorithm (tiled `flash_sdpa`,
# not the materialized path the CPU runs).
const DEV_RTOL = GPU_BACKEND == "jlarrays" ? sqrt(eps(Float32)) : 1.0f-3

# Backward passes accumulate through several matmuls, so they need at least as
# much room as the forward.
const GRAD_RTOL = max(DEV_RTOL, 1.0f-4)

# Move every array parameter onto the device. This is what `Flux.gpu` does once a
# backend's MLDataDevices trigger is fully loaded, spelled explicitly because
# `Flux.gpu` silently no-ops when that trigger is missing.
to_device(m) = fmap(x -> x isa AbstractArray ? dev(x) : x, m)

dev_approx(a, b) = isapprox(Array(a), Array(b); rtol=DEV_RTOL)

if GPU_BACKEND == "jlarrays"
    # JLArrays is the one backend with no batched GEMM; real ones provide it for
    # every argument combination. Real eltypes only, so an adjoint is a transpose.
    _undo_batched(A::JLArray{T,3}) where {T} = A
    function _undo_batched(A::NNlib.BatchedAdjoint{T,<:JLArray{T,3}}) where {T}
        return permutedims(parent(A), (2, 1, 3))
    end

    function _jl_batched_mul(A, B)
        Ad, Bd = _undo_batched(A), _undo_batched(B)
        nb = max(size(Ad, 3), size(Bd, 3))
        C = similar(Ad, size(Ad, 1), size(Bd, 2), nb)
        for i in 1:nb
            ia = size(Ad, 3) == 1 ? 1 : i
            ib = size(Bd, 3) == 1 ? 1 : i
            C[:, :, i] = Ad[:, :, ia] * Bd[:, :, ib]
        end
        return C
    end

    # The forward needs one combination; `batched_mul`'s pullback wraps its
    # arguments in `BatchedAdjoint`, so the backward needs the others.
    const _JLBatched{T} = Union{JLArray{T,3},NNlib.BatchedAdjoint{T,<:JLArray{T,3}}}

    NNlib.batched_mul(A::JLArray{T,3}, B::JLArray{T,3}) where {T} = _jl_batched_mul(A, B)
    function NNlib.batched_mul(
        A::NNlib.BatchedAdjoint{T,<:JLArray{T,3}}, B::_JLBatched{T}
    ) where {T}
        return _jl_batched_mul(A, B)
    end
    function NNlib.batched_mul(
        A::JLArray{T,3}, B::NNlib.BatchedAdjoint{T,<:JLArray{T,3}}
    ) where {T}
        return _jl_batched_mul(A, B)
    end
end
