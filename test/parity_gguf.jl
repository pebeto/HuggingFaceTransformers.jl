using Test
using JSON3
using HuggingFaceTransformers.HFHub: download_file
using HuggingFaceTransformers.Models: load_gguf_model

# Each fixture holds the logits transformers computes after loading the same GGUF
# file with `gguf_file=`, recorded by test/fixtures/record_gguf_parity.py.
const GGUF_VARIANTS = (
    "smollm2-q4_k_m", "smollm2-q3_k_m", "qwen2.5-0.5b-q4_k_m", "qwen3-0.6b-q2_k",
    "gemma2-2b-q4_k_m",
)

function _gguf_selected(raw::AbstractString)
    s = strip(raw)
    s == "all" && return collect(GGUF_VARIANTS)
    names = String[strip(t) for t in split(s, ',') if !isempty(strip(t))]
    unknown = setdiff(names, GGUF_VARIANTS)
    isempty(unknown) ||
        error("Unknown HFT_TEST_PARITY_GGUF variant(s) $(unknown); valid: $(GGUF_VARIANTS) or \"all\"")
    return names
end

for name in _gguf_selected(get(ENV, "HFT_TEST_PARITY_GGUF", ""))
    fixture = JSON3.read(read(joinpath(@__DIR__, "fixtures", "gguf_$(name)_parity.json"), String))
    path = download_file(String(fixture.repo_id), String(fixture.gguf_file))
    @testset "GGUF $(name) ($(fixture.gguf_file))" begin
        lm = load_gguf_model(path)
        ids = Int[Int(x) for x in fixture.input_token_ids]
        logits = collect(lm(reshape(ids, :, 1); step=1)[:, end, 1])
        @test argmax(logits) - 1 == fixture.argmax_token_id
        top = Int[Int(i) for i in fixture.top_indices]
        @test maximum(abs.(logits[top .+ 1] .- Float32.(fixture.top_logits))) < fixture.tolerance
    end
end
