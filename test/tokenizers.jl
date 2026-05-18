using Test
using JSON3
using Allspark.Tokenizers
using Allspark.Tokenizers:
    BPEModel,
    bpe_encode_word,
    token_ids,
    bytes_to_string,
    string_to_bytes,
    BYTES_TO_UNICODE,
    UNICODE_TO_BYTES,
    PreTokenizer,
    Decoder,
    ByteLevelPreTokenizer,
    SplitPreTokenizer,
    SequencePreTokenizer,
    IdentityPreTokenizer,
    ByteLevelDecoder,
    apply_pre,
    apply_dec

@testset "byte-level mapping" begin
    @testset "bijective over all 256 bytes" begin
        @test length(BYTES_TO_UNICODE) == 256
        @test length(UNICODE_TO_BYTES) == 256
        for b in 0x00:0xff
            @test UNICODE_TO_BYTES[BYTES_TO_UNICODE[b]] == b
        end
    end

    @testset "string round-trip" begin
        for s in [
            "",
            "hello",
            "hello world",
            "Hello, 世界! 🎉",
            "\n\t  spaces  ",
            "  leading and trailing  ",
            String(rand(UInt8, 200)),
        ]
            @test string_to_bytes(bytes_to_string(s)) == s
        end
    end

    @testset "known mappings" begin
        @test BYTES_TO_UNICODE[UInt8(' ')] == 'Ġ'
        @test BYTES_TO_UNICODE[UInt8('\n')] == 'Ċ'
        @test bytes_to_string(" hello") == "Ġhello"
    end
end

@testset "bpe_encode_word" begin
    model = BPEModel(
        Dict(
            "h" => 0,
            "e" => 1,
            "l" => 2,
            "o" => 3,
            "he" => 4,
            "ll" => 5,
            "hell" => 6,
            "hello" => 7,
        ),
        [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")],
    )

    @test bpe_encode_word(model, "") == String[]
    @test bpe_encode_word(model, "h") == ["h"]
    @test bpe_encode_word(model, "he") == ["he"]
    @test bpe_encode_word(model, "hello") == ["hello"]
    @test bpe_encode_word(model, "hellx") == ["hell", "x"]
    @test bpe_encode_word(model, "ehlo") == ["e", "h", "l", "o"]

    @test token_ids(model, ["hello"]) == [7]
    @test token_ids(model, ["he", "ll", "o"]) == [4, 5, 3]
    @test_throws KeyError token_ids(model, ["unknown"])
end

const FIXTURE_MERGE_PAIRS = [
    ("h", "e"),
    ("l", "l"),
    ("he", "ll"),
    ("hell", "o"),
    ("Ġ", "w"),
    ("o", "r"),
    ("Ġw", "or"),
    ("l", "d"),
    ("Ġwor", "ld"),
]

const FIXTURE_MERGED_TOKENS =
    ["he", "ll", "hell", "hello", "Ġw", "or", "Ġwor", "ld", "Ġworld"]

const FIXTURE_ENDOFTEXT_ID = 1000

function _fixture_vocab()
    vocab = Dict{String,Int}()
    next_id = 0
    for b in 0x00:0xff
        vocab[string(BYTES_TO_UNICODE[b])] = next_id
        next_id += 1
    end
    for tok in FIXTURE_MERGED_TOKENS
        vocab[tok] = next_id
        next_id += 1
    end
    return vocab
end

function _fixture_dict(; merges_as_strings::Bool=false)
    vocab = _fixture_vocab()
    merges = if merges_as_strings
        [a * " " * b for (a, b) in FIXTURE_MERGE_PAIRS]
    else
        FIXTURE_MERGE_PAIRS
    end
    return Dict(
        "added_tokens" => [
            Dict(
                "id" => FIXTURE_ENDOFTEXT_ID,
                "content" => "<|endoftext|>",
                "special" => true,
            ),
        ],
        "pre_tokenizer" => Dict(
            "type" => "ByteLevel", "add_prefix_space" => false, "use_regex" => true
        ),
        "decoder" => Dict("type" => "ByteLevel"),
        "model" => Dict(
            "type" => "BPE", "vocab" => vocab, "merges" => merges, "unk_token" => nothing
        ),
    )
end

function _write_fixture(dir; merges_as_strings::Bool=false)
    path = joinpath(dir, "tokenizer.json")
    open(io -> JSON3.write(io, _fixture_dict(; merges_as_strings)), path, "w")
    return path
end

@testset "load_tokenizer + encode/decode" begin
    @testset "array-format merges" begin
        mktempdir() do dir
            path = _write_fixture(dir)
            for entry in (path, dir)
                tk = load_tokenizer(entry)
                vocab = tk.model.vocab

                @test encode(tk, "hello world") == [vocab["hello"], vocab["Ġworld"]]
                @test decode(tk, encode(tk, "hello world")) == "hello world"
            end
        end
    end

    @testset "string-format merges" begin
        mktempdir() do dir
            path = _write_fixture(dir; merges_as_strings=true)
            tk = load_tokenizer(path)
            @test encode(tk, "hello world") ==
                [tk.model.vocab["hello"], tk.model.vocab["Ġworld"]]
            @test decode(tk, encode(tk, "hello world")) == "hello world"
        end
    end

    @testset "round-trip over varied inputs" begin
        mktempdir() do dir
            tk = load_tokenizer(_write_fixture(dir))
            for s in [
                "hello",
                "hello world",
                " hello world ",
                "world hello",
                "hello\nworld",
                "hello, world!",
                "h",
                "",
            ]
                @test decode(tk, encode(tk, s)) == s
            end
        end
    end

    @testset "special tokens" begin
        mktempdir() do dir
            tk = load_tokenizer(_write_fixture(dir))
            eot = FIXTURE_ENDOFTEXT_ID

            ids = encode(tk, "hello<|endoftext|> world")
            @test eot in ids
            # Order: hello, <|eot|>, Ġworld
            @test ids == [tk.model.vocab["hello"], eot, tk.model.vocab["Ġworld"]]

            @test decode(tk, ids) == "hello<|endoftext|> world"
            @test decode(tk, ids; skip_special_tokens=true) == "hello world"
        end
    end

    @testset "unknown id raises" begin
        mktempdir() do dir
            tk = load_tokenizer(_write_fixture(dir))
            @test_throws KeyError decode(tk, [999_999])
        end
    end

    @testset "missing tokenizer.json" begin
        mktempdir() do dir
            @test_throws ArgumentError load_tokenizer(dir)
            @test_throws ArgumentError load_tokenizer(joinpath(dir, "nope.json"))
        end
    end
end

@testset "pre-tokenizer building blocks" begin
    @testset "ByteLevel with regex" begin
        pre = ByteLevelPreTokenizer(false, true)
        @test apply_pre(pre, ["hello world"]) == ["hello", "Ġworld"]
        @test apply_pre(pre, ["one two three"]) == ["one", "Ġtwo", "Ġthree"]
    end

    @testset "ByteLevel without regex" begin
        pre = ByteLevelPreTokenizer(false, false)
        @test apply_pre(pre, ["hi"]) == ["hi"]
        @test apply_pre(pre, [" hi"]) == ["Ġhi"]
    end

    @testset "Split + ByteLevel sequence (Llama-3 shape)" begin
        split = SplitPreTokenizer(r"\w+|[^\w\s]+|\s+", :isolated)
        bl = ByteLevelPreTokenizer(false, false)
        seq = SequencePreTokenizer(PreTokenizer[split, bl])
        # "hello world!" → ["hello", " ", "world", "!"] via Split → byte-encode each
        out = apply_pre(seq, ["hello world!"])
        @test out == ["hello", "Ġ", "world", "!"]
    end

    @testset "Identity" begin
        @test apply_pre(IdentityPreTokenizer(), ["x", "y"]) == ["x", "y"]
    end
end

@testset "decoder building blocks" begin
    @test apply_dec(ByteLevelDecoder(), "helloĠworld") == "hello world"
end
