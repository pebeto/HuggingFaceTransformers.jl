# GPU CI plan

GitHub-hosted runners do not have GPUs, so the matrix in `.github/workflows/CI.yml` is CPU-only. Allspark is fundamentally a GPU library — without GPU coverage we cannot merge anything in Phase 1 (KV-cache mutation) or Phase 4 (FlashAttention, fp16/bf16, quantization) with confidence. This document describes the plan for adding it.

This is not wired up yet. It will be activated as part of Phase 1.

## Options considered

### 1. Buildkite + a self-hosted CUDA agent (recommended)

This is the path the wider JuliaGPU ecosystem (CUDA.jl, Metalhead.jl, Flux.jl) uses. Buildkite's free open-source tier is suitable, and the JuliaGPU project already maintains shared infrastructure that can host new projects.

**Pros:**
- Battle-tested for Julia GPU CI.
- We can request a slot on the existing JuliaGPU Buildkite cluster instead of running our own runner.
- Supports CUDA, ROCm (AMDGPU), and (with extra setup) Metal via a Mac mini.

**Cons:**
- Requires writing a `.buildkite/pipeline.yml` and reproducing some of the GitHub Actions setup logic.
- Status checks land on the PR via webhook rather than as native GH Actions checks; the UX is slightly less integrated.

### 2. Self-hosted GitHub Actions runner on a CUDA host

A single runner attached to a workstation with an NVIDIA GPU; gated by a label (e.g. `gpu`) on the runner and `runs-on: [self-hosted, gpu]` in the workflow.

**Pros:**
- Status checks appear natively in PR UI.
- One CI system to learn instead of two.

**Cons:**
- Owning and operating the runner is on us — disk maintenance, Docker isolation between PRs, GitHub Actions runner upgrades, security patching.
- Security: untrusted PR code runs on hardware we own. We must guard `pull_request` triggers behind a `pull_request_target` review-required workflow, or only run GPU CI on PRs from collaborators.

### 3. Paid cloud GPU CI (e.g. CirrusCI, BuildJet GPU)

Out of scope while the project is pre-alpha. Revisit if funding/sponsorship makes it relevant.

## Decision

Adopt **Buildkite + JuliaGPU cluster**. Once we have a working Phase 1 prototype, open an issue against the JuliaGPU buildkite infrastructure repository requesting a project slot, and land a `.buildkite/pipeline.yml` covering:

- Julia 1.10 + Julia 1.12.
- CUDA backend.
- AMDGPU backend (once a runner is available on the cluster).
- Metal backend (lower priority — handled via a macOS GitHub Actions runner with an Apple Silicon image once that hosted option matures, or skipped until someone with a Mac mini volunteers).

Until then, contributors are expected to run the GPU test suite locally before submitting PRs that touch CUDA/AMDGPU/Metal extensions. The CI matrix will be expanded to require GPU status checks at the same time the buildkite pipeline lands.

## Required GPU-only tests (when wired)

These are the test files that must run on GPU CI in addition to the CPU matrix:

- `test/gpu/` — anything under this directory.
- The numeric-parity tests (Phase 1) should run on both CPU and GPU, with tolerance widened for fp16/bf16 paths.
- KV-cache tests must run on GPU; in-place mutation semantics differ between `Array` and `CuArray`/`ROCArray`/`MtlArray`.
