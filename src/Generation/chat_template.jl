"""
    ChatTemplate(source::AbstractString)

Parses `source` once at construction. Calling the result against a
message list renders the template against a context dict built from
the messages and any extra keyword arguments.

Supports the Jinja2 subset needed to render simplified Llama-3,
Qwen2.5, and Mistral chat templates from `tokenizer_config.json`:

- Statements: `if` / `elif` / `else` / `endif`, `for` / `endfor`, `set`.
- Expressions: literals (string / int / bool / `none`), variables,
  attribute and item access, slicing, arithmetic + comparison + logical
  operators, `in` / `not in`, `is defined` / `is not defined`.
- Filters: `|trim`, `|tojson` (no filter arguments).
- The `raise_exception("msg")` function.
- Loop variables `loop.first`, `loop.last`, `loop.index`,
  `loop.index0`, `loop.length`.
- Whitespace control: `{%- … -%}` and `{{- … -}}`.

Not supported: tuple / list / dict literals, float literals, macros,
includes, custom filters, custom tests, slicing with a step.
"""
struct ChatTemplate
    ast::Vector{TmplNode}
end

ChatTemplate(source::AbstractString) = ChatTemplate(parse_template(source))

function (t::ChatTemplate)(messages; add_generation_prompt::Bool=true, kwargs...)
    ctx = Dict{String,Any}(
        "messages" => messages, "add_generation_prompt" => add_generation_prompt
    )
    for (k, v) in pairs(kwargs)
        ctx[String(k)] = v
    end
    io = IOBuffer()
    _render_nodes!(io, t.ast, ctx)
    return String(take!(io))
end

"""
    apply_chat_template(template, messages; add_generation_prompt=true, kwargs...)

Parse `template` and render it against `messages`. Extra keyword
arguments (`bos_token`, `eos_token`, etc.) become context variables
visible to the template.
"""
function apply_chat_template(
    template::AbstractString, messages; add_generation_prompt::Bool=true, kwargs...
)
    return ChatTemplate(template)(
        messages; add_generation_prompt=add_generation_prompt, kwargs...
    )
end
