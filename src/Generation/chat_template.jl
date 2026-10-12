"""
    ChatTemplate(source::AbstractString)

Parses `source` once at construction. Calling the result against a
message list renders the template against a context dict built from
the messages and any extra keyword arguments.

Supports the Jinja2 subset HF chat templates use, under HF's environment
settings (`trim_blocks` and `lstrip_blocks` on). The Qwen3 and Qwen2.5 templates
render byte for byte as `apply_chat_template` does.

- Statements: `if` / `elif` / `else`, `for` with `break` / `continue`, `set`,
  and `set ns.attr = ...` on a `namespace(...)`.
- Expressions: string / int / bool / `none` literals, variables, attribute and
  item access, Python indexing and slicing (negative indices, steps),
  arithmetic, comparison and logical operators, `in` / `not in`, and the
  string methods templates call (`startswith`, `endswith`, `split`, `strip`,
  `lstrip`, `rstrip`, `upper`, `lower`, `replace`) plus a mapping's `get`.
- Tests: `defined`, `undefined`, `none`, `true`, `false`, `boolean`, `string`,
  `number`, `integer`, `float`, `mapping`, `sequence`, `iterable`.
- Filters: `trim`, `length` / `count`, and `tojson`, which writes what Python's
  `json.dumps` does (spaces after `,` and `:`, keys in mapping order).
- Functions: `raise_exception("msg")`, `namespace(...)`.
- Loop variables: `loop.index`, `index0`, `revindex`, `revindex0`, `first`,
  `last`, `length`, `previtem`, `nextitem`.
- Whitespace control: `-` and `+` modifiers on tags.

Not supported: list / dict literals, float literals, filter arguments, macros,
includes, `for` tuple unpacking, and `strftime_now`.
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
    signal = _render_nodes!(io, t.ast, ctx)
    isnothing(signal) || throw(ArgumentError("`$(signal)` outside a for loop"))
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
