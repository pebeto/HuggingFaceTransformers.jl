using Test
using HuggingFaceTransformers.Generation: ChatTemplate, apply_chat_template

@testset verbose = true "building blocks" begin
    @testset "literals & basic output" begin
        @test apply_chat_template("hello", []) == "hello"
        @test apply_chat_template("{{ 'hi' }}", []) == "hi"
        @test apply_chat_template("{{ 42 }}", []) == "42"
        @test apply_chat_template("{{ true }}", []) == "True"
        @test apply_chat_template("{{ none }}", []) == ""
    end

    @testset "variables & kwargs context" begin
        @test apply_chat_template("Hi {{ name }}", []; name="world") == "Hi world"
        @test apply_chat_template("{{ bos_token }}x", []; bos_token="<s>") == "<s>x"
    end

    @testset "string concat with `+`" begin
        @test apply_chat_template("{{ 'a' + 'b' }}", []) == "ab"
        @test apply_chat_template("{{ 'n=' + 1 }}", []) == "n=1"
    end

    @testset "arithmetic, comparison, logical" begin
        @test apply_chat_template("{{ 2 + 3 }}", []) == "5"
        @test apply_chat_template("{{ 7 % 2 }}", []) == "1"
        @test apply_chat_template("{% if 1 == 1 %}eq{% endif %}", []) == "eq"
        @test apply_chat_template("{% if 1 != 2 %}ne{% endif %}", []) == "ne"
        @test apply_chat_template("{% if 1 < 2 and 2 < 3 %}y{% endif %}", []) == "y"
        @test apply_chat_template("{% if not (1 == 2) %}y{% endif %}", []) == "y"
        @test apply_chat_template("{% if 1 == 2 or 3 == 3 %}y{% endif %}", []) == "y"
    end

    @testset "if / elif / else / endif" begin
        tmpl = "{% if x == 1 %}one{% elif x == 2 %}two{% else %}many{% endif %}"
        @test apply_chat_template(tmpl, []; x=1) == "one"
        @test apply_chat_template(tmpl, []; x=2) == "two"
        @test apply_chat_template(tmpl, []; x=99) == "many"
    end

    @testset "set" begin
        @test apply_chat_template("{% set x = 'hi' %}{{ x }}", []) == "hi"
        # `set` inside an if leaks into the rest of the template (Jinja2 scoping).
        tmpl = "{% if true %}{% set y = 7 %}{% endif %}{{ y }}"
        @test apply_chat_template(tmpl, []) == "7"
    end

    @testset "for + loop variables" begin
        tmpl = "{% for x in items %}{{ x }}{% endfor %}"
        @test apply_chat_template(tmpl, []; items=["a", "b", "c"]) == "abc"

        tmpl2 = "{% for x in items %}{{ loop.index0 }}:{{ x }};{% endfor %}"
        @test apply_chat_template(tmpl2, []; items=["a", "b"]) == "0:a;1:b;"

        tmpl3 =
            "{% for x in items %}{% if loop.first %}[{% endif %}{{ x }}" *
            "{% if loop.last %}]{% endif %}{% endfor %}"
        @test apply_chat_template(tmpl3, []; items=["a", "b", "c"]) == "[abc]"
    end

    @testset "member access — dict & namedtuple" begin
        msg_dict = Dict("role" => "user", "content" => "hi")
        @test apply_chat_template("{{ m.role }}:{{ m.content }}", []; m=msg_dict) ==
            "user:hi"
        @test apply_chat_template("{{ m['role'] }}", []; m=msg_dict) == "user"

        msg_nt = (role="user", content="hi")
        @test apply_chat_template("{{ m.role }}", []; m=msg_nt) == "user"
    end

    @testset "indexing & slicing" begin
        @test apply_chat_template("{{ xs[0] }}", []; xs=["a", "b", "c"]) == "a"
        @test apply_chat_template("{{ xs[2] }}", []; xs=["a", "b", "c"]) == "c"
        # 1: == drop first; HF Mistral template uses this.
        tmpl = "{% for x in xs[1:] %}{{ x }}{% endfor %}"
        @test apply_chat_template(tmpl, []; xs=["a", "b", "c"]) == "bc"
    end

    @testset "is defined" begin
        @test apply_chat_template(
            "{% if name is defined %}yes{% else %}no{% endif %}", []; name="x"
        ) == "yes"
        @test apply_chat_template(
            "{% if name is defined %}yes{% else %}no{% endif %}", []
        ) == "no"
        @test apply_chat_template("{% if name is not defined %}no{% endif %}", []) == "no"
    end

    @testset "filters" begin
        @test apply_chat_template("{{ '  hi  ' | trim }}", []) == "hi"
        @test apply_chat_template("{{ x | tojson }}", []; x=Dict("a" => 1)) == "{\"a\":1}"
        # chained
        @test apply_chat_template("{{ '  hi  ' | trim | trim }}", []) == "hi"
    end

    @testset "raise_exception" begin
        @test_throws ArgumentError apply_chat_template("{{ raise_exception('nope') }}", [])
    end

    @testset "whitespace control" begin
        # `-` strips whitespace on the marked side(s).
        @test apply_chat_template("a {{- x }} b", []; x="X") == "aX b"
        @test apply_chat_template("a {{ x -}} b", []; x="X") == "a Xb"
        @test apply_chat_template("a\n{%- if true -%}\nB\n{%- endif -%}\nc", []) == "aBc"
    end

    @testset "comments {# … #}" begin
        @test apply_chat_template("a{# comment #}b", []) == "ab"
    end
end

@testset verbose = true "Realistic HF-style chat templates" begin
    LLAMA3_LIKE = raw"""
    {%- for message in messages -%}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
    {%- endif -%}
    """

    @testset "Llama-3-like template" begin
        messages = [
            Dict("role" => "user", "content" => "Hi there  "),  # trailing ws to test |trim
            Dict("role" => "assistant", "content" => "Hello!"),
        ]
        out = apply_chat_template(LLAMA3_LIKE, messages; add_generation_prompt=true)
        @test occursin(
            "<|start_header_id|>user<|end_header_id|>\n\nHi there<|eot_id|>", out
        )
        @test occursin(
            "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>", out
        )
        @test endswith(out, "<|start_header_id|>assistant<|end_header_id|>\n\n")

        out_nogen = apply_chat_template(LLAMA3_LIKE, messages; add_generation_prompt=false)
        @test !endswith(out_nogen, "<|start_header_id|>assistant<|end_header_id|>\n\n")
    end

    QWEN_LIKE = raw"""
    {%- for message in messages -%}
    {%- if loop.first and messages[0]['role'] != 'system' -%}
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    {% endif -%}
    <|im_start|>{{ message['role'] }}
    {{ message['content'] }}<|im_end|>
    {% endfor -%}
    {%- if add_generation_prompt -%}
    <|im_start|>assistant
    {% endif -%}
    """

    @testset "Qwen-like template" begin
        out = apply_chat_template(
            QWEN_LIKE,
            [Dict("role" => "user", "content" => "Hi")];
            add_generation_prompt=true,
        )
        @test occursin("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", out)
        @test occursin("<|im_start|>user\nHi<|im_end|>", out)
        @test endswith(strip(out), "<|im_start|>assistant")

        # No injected system prompt when first message is already a system message.
        out2 = apply_chat_template(
            QWEN_LIKE,
            [
                Dict("role" => "system", "content" => "You are spec."),
                Dict("role" => "user", "content" => "Hi"),
            ];
            add_generation_prompt=false,
        )
        @test !occursin("You are a helpful assistant.", out2)
        @test occursin("<|im_start|>system\nYou are spec.<|im_end|>", out2)
    end

    MISTRAL_LIKE = raw"""
    {{- bos_token -}}
    {%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set loop_messages = messages[1:] -%}
    {%- else -%}
    {%- set loop_messages = messages -%}
    {%- endif -%}
    {%- for message in loop_messages -%}
    {%- if message['role'] == 'user' -%}
    {%- if loop.first and system_message is defined -%}
     [INST] {{ system_message }}

    {{ message['content'] }} [/INST]
    {%- else -%}
     [INST] {{ message['content'] }} [/INST]
    {%- endif -%}
    {%- elif message['role'] == 'assistant' -%}
     {{ message['content'] }}{{ eos_token }}
    {%- endif -%}
    {%- endfor -%}
    """

    @testset "Mistral-like template" begin
        out = apply_chat_template(
            MISTRAL_LIKE,
            [
                Dict("role" => "system", "content" => "be helpful"),
                Dict("role" => "user", "content" => "say hi"),
            ];
            bos_token="<s>",
            eos_token="</s>",
            add_generation_prompt=false,
        )
        @test startswith(out, "<s>")
        @test occursin("[INST] be helpful\n\nsay hi [/INST]", out)

        # Without a system message, no preamble.
        out_no_sys = apply_chat_template(
            MISTRAL_LIKE,
            [Dict("role" => "user", "content" => "hi")];
            bos_token="<s>",
            eos_token="</s>",
            add_generation_prompt=false,
        )
        @test startswith(out_no_sys, "<s>")
        @test occursin("[INST] hi [/INST]", out_no_sys)
        @test !occursin("[INST] be helpful", out_no_sys)
    end

    @testset "ChatTemplate is parsed once, reusable" begin
        tmpl = ChatTemplate(LLAMA3_LIKE)
        a = tmpl([Dict("role" => "user", "content" => "a")])
        b = tmpl([Dict("role" => "user", "content" => "b")])
        @test occursin("\n\na<|eot_id|>", a)
        @test occursin("\n\nb<|eot_id|>", b)
    end

    @testset "errors" begin
        @test_throws ArgumentError apply_chat_template("{% if x %}", [])     # unclosed
        @test_throws ArgumentError apply_chat_template("{% nope %}", [])
        @test_throws ArgumentError apply_chat_template("{{ 1 + }}", [])
    end
end

@testset verbose = true "undefined values" begin
    # Tool-capable templates (Qwen, Llama 3.1+, Mistral v0.3) guard their optional
    # sections with `{% if tools %}`. Jinja skips those branches when the caller
    # passes no tools, so an undefined name has to be falsy rather than an error.
    @testset "undefined is falsy, not an error" begin
        @test apply_chat_template("{% if tools %}T{% else %}none{% endif %}", []) ==
            "none"
        @test apply_chat_template("{% if not tools %}skipped{% endif %}", []) == "skipped"
        @test apply_chat_template("{% if tools %}T{% endif %}ok", []) == "ok"
    end

    @testset "undefined renders empty" begin
        @test apply_chat_template("[{{ missing }}]", []) == "[]"
    end

    @testset "a missing member is undefined too" begin
        # `message.tool_calls` on a message that has no such key.
        msgs = [Dict("role" => "user", "content" => "hi")]
        tmpl = "{% for m in messages %}{% if m.tool_calls %}T{% else %}-{% endif %}" *
               "{% endfor %}"
        @test apply_chat_template(tmpl, msgs) == "-"
        @test apply_chat_template("[{{ messages[0].nope }}]", msgs) == "[]"
    end

    @testset "undefined is not none, and none is still defined" begin
        # Jinja distinguishes the two: a variable set to none exists, an undefined
        # one does not, and neither collapses into the other.
        @test apply_chat_template("{% if x is none %}y{% else %}n{% endif %}", []) == "n"
        @test apply_chat_template(
            "{% if x is none %}y{% else %}n{% endif %}", []; x=nothing
        ) == "y"
        @test apply_chat_template(
            "{% if x is defined %}y{% else %}n{% endif %}", []; x=nothing
        ) == "y"
        @test apply_chat_template("{% if x is defined %}y{% else %}n{% endif %}", []) ==
            "n"
    end

    @testset "reaching through an undefined value still raises" begin
        # Skipping an absent branch is intended; dereferencing one is a template bug.
        @test_throws ArgumentError apply_chat_template("{{ missing.attr }}", [])
        @test_throws ArgumentError apply_chat_template("{{ a.b.c }}", []; a=Dict())
    end
end
