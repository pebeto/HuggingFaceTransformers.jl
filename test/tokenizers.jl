using Test
using JSON3
using Allspark.Tokenizers
using Allspark.Tokenizers:
    BPEModel,
    UnigramModel,
    WordPieceModel,
    bpe_encode_word,
    encode_word,
    token_ids,
    bytes_to_string,
    string_to_bytes,
    BYTES_TO_UNICODE,
    UNICODE_TO_BYTES,
    PreTokenizer,
    Decoder,
    Normalizer,
    ByteLevelPreTokenizer,
    SplitPreTokenizer,
    SequencePreTokenizer,
    IdentityPreTokenizer,
    MetaspacePreTokenizer,
    BertPreTokenizer,
    ByteLevelDecoder,
    ReplaceDecoder,
    ByteFallbackDecoder,
    FuseDecoder,
    StripDecoder,
    SequenceDecoder,
    WordPieceDecoder,
    IdentityNormalizer,
    BertNormalizer,
    apply_pre,
    apply_dec,
    apply_norm,
    load_wordpiece_from_vocab_txt

@testset verbose = true "byte-level mapping" begin
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

const FIXTURE_MERGED_TOKENS = [
    "he", "ll", "hell", "hello", "Ġw", "or", "Ġwor", "ld", "Ġworld"
]

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
        "pre_tokenizer" =>
            Dict("type" => "ByteLevel", "add_prefix_space" => false, "use_regex" => true),
        "decoder" => Dict("type" => "ByteLevel"),
        "model" => Dict(
            "type" => "BPE",
            "vocab" => vocab,
            "merges" => merges,
            "unk_token" => nothing,
        ),
    )
end

function _write_fixture(dir; merges_as_strings::Bool=false)
    path = joinpath(dir, "tokenizer.json")
    open(io -> JSON3.write(io, _fixture_dict(; merges_as_strings)), path, "w")
    return path
end

@testset verbose = true "load_tokenizer + encode/decode" begin
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

@testset verbose = true "pre-tokenizer building blocks" begin
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

# ---------------------------------------------------------------------
# Unigram (SentencePiece) — used by Gemma, T5, …
# ---------------------------------------------------------------------

@testset verbose = true "Unigram model (no byte fallback)" begin
    # Tiny vocab: chars + a few likely bigrams. Scores hand-picked so the
    # best segmentation is unambiguous.
    vocab = Tuple{String,Float32}[
        ("<unk>", -100.0f0),
        ("▁", -2.0f0),
        ("a", -3.0f0),
        ("b", -3.0f0),
        ("c", -3.0f0),
        ("ab", -2.5f0),     # higher than 'a' + 'b' = -6.0
        ("▁ab", -2.0f0),    # higher than '▁' + 'ab' = -4.5
        ("bc", -2.5f0),
    ]
    model = UnigramModel(vocab; unk_id=0, byte_fallback=false)

    @testset "vocab + scores indexed by HF id" begin
        @test model.vocab["▁"] == 1
        @test model.vocab["▁ab"] == 6
        @test model.id_to_token[6] == "▁ab"
        @test model.scores[7] == -2.0f0     # 1-based: ids 0..7 → 1..8
    end

    @testset "Viterbi picks the highest-score segmentation" begin
        # "▁abc" → best is "▁ab" + "c" (= -2 + -3 = -5)
        #         not "▁" + "a" + "b" + "c" (= -2 - 3 - 3 - 3 = -11)
        #         not "▁" + "ab" + "c"      (= -2 - 2.5 - 3 = -7.5)
        ids = encode_word(model, "▁abc")
        @test ids == [6, 4]
        @test model.id_to_token[6] == "▁ab"
        @test model.id_to_token[4] == "c"
    end

    @testset "no path → unk" begin
        # Character not in vocab. Without byte_fallback, no path; returns unk.
        ids = encode_word(model, "z")
        @test ids == [0]                    # unk_id
    end
end

@testset verbose = true "Unigram model (byte fallback)" begin
    # Vocab includes some printable chars + the full 256 `<0xHH>` byte tokens,
    # so any UTF-8 input is encodable.
    vocab = Tuple{String,Float32}[("<unk>", -100.0f0), ("a", -1.0f0)]
    for b in 0x00:0xff
        tok = "<0x" * uppercase(string(b; base=16, pad=2)) * ">"
        push!(vocab, (tok, -5.0f0))
    end
    model = UnigramModel(vocab; unk_id=0, byte_fallback=true)

    @testset "byte_to_id is populated" begin
        @test model.byte_to_id[Int('a') + 1] !== nothing   # via the byte token
        @test model.byte_to_id[Int('z') + 1] !== nothing   # also via byte token
        @test count(!isnothing, model.byte_to_id) == 256
    end

    @testset "vocab tokens beat byte fallback when their score is higher" begin
        # 'a' as a vocab token (-1) beats '<0x61>' as a byte fallback (-5).
        ids = encode_word(model, "a")
        @test ids == [1]                   # the 'a' vocab token
    end

    @testset "out-of-vocab character routes through byte fallback" begin
        # 'z' isn't in the vocab; byte fallback emits '<0x7A>'.
        ids = encode_word(model, "z")
        @test length(ids) == 1
        @test model.id_to_token[ids[1]] == "<0x7A>"
    end

    @testset "multi-byte UTF-8 emits one byte token per UTF-8 byte" begin
        # 'é' is C3 A9 in UTF-8 → two byte tokens.
        ids = encode_word(model, "é")
        @test length(ids) == 2
        @test model.id_to_token[ids[1]] == "<0xC3>"
        @test model.id_to_token[ids[2]] == "<0xA9>"
    end
end

@testset verbose = true "MetaspacePreTokenizer" begin
    @testset "replaces spaces and prepends by default" begin
        pre = MetaspacePreTokenizer("▁", :always)
        out = apply_pre(pre, ["hello world"])
        @test out == ["▁hello▁world"]
    end

    @testset "prepend_scheme=:never skips the leading replacement" begin
        pre = MetaspacePreTokenizer("▁", :never)
        out = apply_pre(pre, ["hello world"])
        @test out == ["hello▁world"]
    end

    @testset "no double-prepend when input already starts with replacement" begin
        pre = MetaspacePreTokenizer("▁", :always)
        out = apply_pre(pre, ["▁hello"])
        @test out == ["▁hello"]
    end
end

@testset verbose = true "SentencePiece-style decoders" begin
    @testset "ReplaceDecoder swaps the literal pattern" begin
        @test apply_dec(ReplaceDecoder("▁", " "), "▁hello▁world") == " hello world"
    end

    @testset "ByteFallbackDecoder collapses runs of <0xHH>" begin
        # "é" (C3 A9) split across two byte tokens, then decoded.
        @test apply_dec(ByteFallbackDecoder(), "<0xC3><0xA9>") == "é"
        # Mixed with literal text.
        @test apply_dec(ByteFallbackDecoder(), "hi<0xC3><0xA9>!") == "hié!"
        # No-op when no byte tokens are present.
        @test apply_dec(ByteFallbackDecoder(), "hello") == "hello"
    end

    @testset "FuseDecoder is identity" begin
        @test apply_dec(FuseDecoder(), "anything") == "anything"
    end

    @testset "StripDecoder strips counted occurrences" begin
        @test apply_dec(StripDecoder(" ", 1, 0), " hello") == "hello"
        @test apply_dec(StripDecoder(" ", 1, 0), "hello") == "hello"
        @test apply_dec(StripDecoder(" ", 0, 1), "hello ") == "hello"
        @test apply_dec(StripDecoder(" ", 1, 1), " hello ") == "hello"
    end

    @testset "Gemma-shape decoder sequence round-trips Metaspace output" begin
        # Mirrors Gemma's tokenizer.json: Replace ▁→" ", ByteFallback, Fuse,
        # Strip 1 leading " ".
        gemma_decoder = SequenceDecoder([
            ReplaceDecoder("▁", " "),
            ByteFallbackDecoder(),
            FuseDecoder(),
            StripDecoder(" ", 1, 0),
        ])
        # Pretokenizer would produce "▁hello▁world"; decoder must undo it.
        @test apply_dec(gemma_decoder, "▁hello▁world") == "hello world"
        # With a byte-fallback fragment inside.
        @test apply_dec(gemma_decoder, "▁caf<0xC3><0xA9>") == "café"
    end
end

# ---------------------------------------------------------------------
# WordPiece (BERT family)
# ---------------------------------------------------------------------

@testset verbose = true "WordPieceModel" begin
    # Tiny vocab. Continuation prefix is "##" by convention.
    vocab = Dict{String,Int}(
        "[UNK]" => 0,
        "the" => 1,
        "un" => 2,
        "##able" => 3,
        "##s" => 4,
        "thing" => 5,
        "##ing" => 6,
        "a" => 7,
        "b" => 8,
        "c" => 9,
        "##bc" => 10,
    )
    model = WordPieceModel(vocab; unk_token="[UNK]")

    @testset "greedy longest-match for a multi-piece word" begin
        # "unable" → "un" + "##able"  (greedy from the left).
        @test encode_word(model, "unable") == [2, 3]
        # "things" → "thing" + "##s" (longest-first beats single chars).
        @test encode_word(model, "things") == [5, 4]
    end

    @testset "single-piece word resolves directly" begin
        @test encode_word(model, "the") == [1]
    end

    @testset "unmatchable word collapses to UNK" begin
        # 'z' isn't in the vocab — whole word → UNK.
        @test encode_word(model, "zzz") == [0]
    end

    @testset "continuing-prefix lookup uses ## form" begin
        # "abc": first piece is "a" (id 7), then "##bc" must match
        # (not "##b" which doesn't exist).
        @test encode_word(model, "abc") == [7, 10]
    end

    @testset "max_input_chars_per_word triggers UNK" begin
        short_model = WordPieceModel(
            vocab; unk_token="[UNK]", max_input_chars_per_word=3
        )
        @test encode_word(short_model, "things") == [0]
    end
end

@testset verbose = true "BertPreTokenizer" begin
    pre = BertPreTokenizer()

    @testset "splits on whitespace" begin
        @test apply_pre(pre, ["hello world"]) == ["hello", "world"]
    end

    @testset "isolates punctuation" begin
        @test apply_pre(pre, ["don't!"]) == ["don", "'", "t", "!"]
        @test apply_pre(pre, ["hi, there."]) == ["hi", ",", "there", "."]
    end

    @testset "drops empty fragments from consecutive separators" begin
        @test apply_pre(pre, ["  a  b  "]) == ["a", "b"]
    end
end

@testset verbose = true "BertNormalizer" begin
    @testset "lowercase + strip_accents combined" begin
        n = BertNormalizer(;
            clean_text=false,
            handle_chinese_chars=false,
            strip_accents=true,
            lowercase=true,
        )
        # café → cafe (accent stripped → lowercased; both 'C' and 'é').
        @test apply_norm(n, "Café") == "cafe"
        # ñ decomposes to n + combining tilde; combining mark drops.
        @test apply_norm(n, "Mañana") == "manana"
    end

    @testset "clean_text replaces control characters with spaces" begin
        n = BertNormalizer(;
            clean_text=true,
            handle_chinese_chars=false,
            strip_accents=false,
            lowercase=false,
        )
        @test apply_norm(n, "a\x07b") == "a b"
        # Tabs/newlines/cr are preserved.
        @test apply_norm(n, "a\tb\n") == "a\tb\n"
    end

    @testset "handle_chinese_chars pads CJK with spaces" begin
        n = BertNormalizer(;
            clean_text=false,
            handle_chinese_chars=true,
            strip_accents=false,
            lowercase=false,
        )
        # 你 is at U+4F60 (CJK Unified Ideographs range).
        @test apply_norm(n, "你好") == " 你  好 "
    end

    @testset "IdentityNormalizer is a no-op" begin
        @test apply_norm(IdentityNormalizer(), "Anything") == "Anything"
    end
end

@testset verbose = true "WordPieceDecoder" begin
    @testset "removes ## continuing prefix and the leading space" begin
        @test apply_dec(WordPieceDecoder("##", false), "un ##able") == "unable"
        @test apply_dec(WordPieceDecoder("##", false), "thing ##s") == "things"
    end

    @testset "cleanup folds spaces around punctuation" begin
        @test apply_dec(WordPieceDecoder("##", true), "hi , world .") == "hi, world."
        @test apply_dec(WordPieceDecoder("##", true), "don ' t") == "don't"
        @test apply_dec(WordPieceDecoder("##", true), "do n't") == "don't"
    end
end

@testset verbose = true "load_wordpiece_from_vocab_txt + encode/decode" begin
    @testset "round-trip a tiny BERT-shaped vocab" begin
        mktempdir() do dir
            path = joinpath(dir, "vocab.txt")
            open(path, "w") do io
                # IDs are 0-indexed line numbers.
                println(io, "[UNK]")           # 0
                println(io, "[CLS]")           # 1
                println(io, "[SEP]")           # 2
                println(io, "the")             # 3
                println(io, "quick")           # 4
                println(io, "fox")             # 5
                println(io, ".")               # 6
                println(io, "br")              # 7
                println(io, "##own")           # 8
            end
            tk = load_wordpiece_from_vocab_txt(path)
            ids = encode(tk, "The Quick Brown Fox.")
            @test ids == [3, 4, 7, 8, 5, 6]
            # Decode round-trips (lowercased due to BertNormalizer).
            @test decode(tk, ids) == "the quick brown fox."
        end
    end

    @testset "directory form picks up vocab.txt" begin
        mktempdir() do dir
            open(joinpath(dir, "vocab.txt"), "w") do io
                println(io, "[UNK]")
                println(io, "hi")
            end
            tk = load_wordpiece_from_vocab_txt(dir)
            @test encode(tk, "Hi") == [1]
        end
    end

    @testset "missing vocab.txt errors" begin
        mktempdir() do dir
            @test_throws ArgumentError load_wordpiece_from_vocab_txt(dir)
        end
    end
end
