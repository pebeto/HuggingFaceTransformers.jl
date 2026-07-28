using Test
using JSON3
using HuggingFaceTransformers.HFHub: snapshot_download
using HuggingFaceTransformers.Tokenizers: load_tokenizer, encode
using HuggingFaceTransformers.Models:
    load_weights, BertForMaskedLM, BertConfig, load_state_dict!

# The recorded fixtures cover RoBERTa (base and large). WordPiece is
# supported now, so the tokenizer is no longer what blocks BERT proper; it
# simply has no recorded fixture yet.
const VARIANTS = (
    ("base", "roberta_base_parity.json"), ("large", "roberta_large_parity.json")
)

const FIXTURES_DIR = joinpath(@__DIR__, "fixtures")

function _selected_variants(raw::AbstractString)
    s = strip(raw)
    isempty(s) && return String[]
    s == "all" && return String[v[1] for v in VARIANTS]
    parts = String[]
    for token in split(s, ',')
        t = strip(token)
        isempty(t) || push!(parts, String(t))
    end
    return parts
end

const SELECTED = _selected_variants(get(ENV, "HFT_TEST_PARITY_BERT", ""))

function _load_roberta_config(snapshot_dir::AbstractString)
    raw = JSON3.read(read(joinpath(snapshot_dir, "config.json"), String))

    return BertConfig(;
        vocab_size=Int(raw.vocab_size),
        hidden_size=Int(raw.hidden_size),
        intermediate_size=Int(raw.intermediate_size),
        num_hidden_layers=Int(raw.num_hidden_layers),
        num_attention_heads=Int(raw.num_attention_heads),
        max_position_embeddings=Int(raw.max_position_embeddings) - 2,
        type_vocab_size=Int(get(raw, :type_vocab_size, 1)),
        layer_norm_eps=Float64(get(raw, :layer_norm_eps, 1.0e-5)),
        pad_token_id=Int(get(raw, :pad_token_id, 1)),
        position_embedding_offset=2,        # RoBERTa: padding_idx + 1
        hf_prefix="roberta",
        head_prefix="lm_head",
        tie_word_embeddings=true,
    )
end

function _run_variant(name::AbstractString, fixture_filename::AbstractString)
    fixture_path = joinpath(FIXTURES_DIR, fixture_filename)
    if !isfile(fixture_path)
        @info "Skipping RoBERTa-$(name) parity: fixture $(fixture_path) not " *
            "present. Generate it with " *
            "`python3 test/fixtures/record_bert_parity.py $(name)`."
        return nothing
    end

    fixture = JSON3.read(read(fixture_path, String))
    repo_id = String(fixture.repo_id)
    prompt = String(fixture.prompt)
    expected_ids = Int[Int(x) for x in fixture.input_token_ids]
    mask_index = Int(fixture.mask_index)          # 0-indexed position of [MASK]
    expected_argmax = Int(fixture.argmax_token_id)
    expected_top_indices = Int[Int(x) for x in fixture.top_indices]
    expected_top_logits = Float32[Float32(x) for x in fixture.top_logits]
    tolerance = Float32(get(fixture, :tolerance, 1.0e-3))

    snapshot_dir = snapshot_download(repo_id; verbose=false)

    @testset verbose = true "RoBERTa-$(name) ($(repo_id))" begin
        @testset "tokenizer matches HF byte-for-byte" begin
            tk = load_tokenizer(snapshot_dir)
            ids = encode(tk, prompt)
            @test ids == expected_ids
        end

        @testset "masked-position logits within $(tolerance)" begin
            cfg = _load_roberta_config(snapshot_dir)
            lm = BertForMaskedLM(cfg)
            load_state_dict!(lm, load_weights(snapshot_dir))

            input_mat = reshape(expected_ids, :, 1)
            logits = lm(input_mat)
            # HF mask index is 0-indexed; Julia is 1-indexed.
            mask_logits = collect(logits[:, mask_index + 1, 1])

            @test argmax(mask_logits) - 1 == expected_argmax

            max_err = Float32(0)
            for (i, hf_idx) in enumerate(expected_top_indices)
                v = Float32(mask_logits[hf_idx + 1])
                err = abs(v - expected_top_logits[i])
                max_err = max(max_err, err)
            end
            @test max_err < tolerance

            jl_top10 = sortperm(mask_logits; rev=true)[1:10] .- 1
            hf_top10 = expected_top_indices[1:10]
            overlap = length(intersect(Set(jl_top10), Set(hf_top10)))
            @test overlap >= 9
        end
    end
end

unknown = filter(v -> !(v in [first(t) for t in VARIANTS]), SELECTED)
isempty(unknown) || error(
    "Unknown HFT_TEST_PARITY_BERT variant(s): $(unknown). " *
    "Valid: $(String[first(v) for v in VARIANTS]) or \"all\".",
)

for (name, fixture) in VARIANTS
    name in SELECTED || continue
    _run_variant(name, fixture)
end
