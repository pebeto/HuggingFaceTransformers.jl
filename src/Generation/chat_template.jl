"""
    ChatTemplate(source::AbstractString)

Parses `source` once at construction. Calling the result against a
message list renders the template against a context dict built from
the messages and any extra keyword arguments.

Supports the Jinja2 subset HF chat templates use, under HF's environment
settings (`trim_blocks` and `lstrip_blocks` on). The Qwen3, Qwen2.5 and gpt-oss
(harmony) templates render byte for byte as `apply_chat_template` does.

- Statements: `if` / `elif` / `else`, `for` with `break` / `continue` and
  unpacking (`for k, v in d.items()`), `set`, `set ns.attr = ...` on a
  `namespace(...)`, and `macro` with default arguments and recursion.
- Expressions: string / int / bool / `none` literals, list literals, variables,
  attribute and item access, Python indexing and slicing (negative indices,
  steps), arithmetic, comparison and logical operators, `in` / `not in`, inline
  `a if cond else b`, the string methods templates call (`startswith`,
  `endswith`, `split`, `strip`, `lstrip`, `rstrip`, `upper`, `lower`, `replace`)
  and a mapping's `get`, `items`, `keys` and `values`.
- Tests: `defined`, `undefined`, `none`, `true`, `false`, `boolean`, `string`,
  `number`, `integer`, `float`, `mapping`, `sequence`, `iterable`.
- Filters: `trim`, `length` / `count`, `join`, `default`, `upper`, `lower`,
  `string`, `first`, `last`, `list`, and `tojson` (with `indent`), which writes
  what Python's `json.dumps` does: spaces after `,` and `:`, keys in mapping
  order.
- Functions: `raise_exception("msg")`, `namespace(...)`, and `strftime_now(fmt)`.
  A keyword argument of the same name shadows a built-in, so
  `strftime_now = _ -> "2025-08-05"` pins the date.
- Loop variables: `loop.index`, `index0`, `revindex`, `revindex0`, `first`,
  `last`, `length`, `previtem`, `nextitem`.
- Whitespace control: `-` and `+` modifiers on tags.

Not supported: dict and float literals, `{% call %}` blocks, includes and
imports, and filters beyond the ones above. A template that needs them raises,
when it is parsed or when the branch runs.
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

"""
    load_chat_template(path; name = "default") -> ChatTemplate

Read a model's chat template where HF's tokenizer finds it: `chat_template.jinja`
when the snapshot has one (gpt-oss ships its template that way), otherwise the
`chat_template` field of `tokenizer_config.json`. `path` is a snapshot directory
or a file inside it. When the field holds several named templates, the one called
`name` is used.
"""
function load_chat_template(path::AbstractString; name::AbstractString="default")
    dir = isdir(path) ? path : dirname(path)
    jinja = joinpath(dir, "chat_template.jinja")
    isfile(jinja) && return ChatTemplate(read(jinja, String))
    config = joinpath(dir, "tokenizer_config.json")
    isfile(config) ||
        throw(ArgumentError("no chat_template.jinja or tokenizer_config.json in $(dir)"))
    field = get(JSON3.read(read(config, String)), :chat_template, nothing)
    isnothing(field) && throw(ArgumentError("$(config) has no chat_template"))
    field isa AbstractString && return ChatTemplate(String(field))
    for entry in field
        String(entry.name) == name && return ChatTemplate(String(entry.template))
    end
    found = join((String(entry.name) for entry in field), ", ")
    throw(ArgumentError("no chat template named `$(name)`; found $(found)"))
end
