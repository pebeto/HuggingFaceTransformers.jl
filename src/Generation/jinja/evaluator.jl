"""
    Undefined

Stands in for a name or member the context does not provide. Jinja renders an
undefined value as empty and treats it as false in a test, which is what lets a
template's optional branches (`{% if tools %}`) be skipped instead of raising.
Deliberately not `nothing`: `undefined is none` must stay false, and a variable
explicitly set to `none` is still defined.
"""
struct Undefined end

"""
    Namespace

What Jinja's `namespace(...)` returns: the one object a template may assign
attributes on (`{% set ns.found = true %}`). It is mutable and shared by
reference, which is how a value set inside a loop survives the loop.
"""
struct Namespace
    attrs::Dict{String,Any}
end

_truthy(::Undefined) = false
_truthy(::Nothing) = false
_truthy(b::Bool) = b
_truthy(s::AbstractString) = !isempty(s)
_truthy(n::Number) = n != 0
_truthy(c::AbstractArray) = !isempty(c)
_truthy(d::AbstractDict) = !isempty(d)
_truthy(_) = true

_stringify(::Undefined) = ""
_stringify(::Nothing) = ""
_stringify(s::AbstractString) = s
_stringify(b::Bool) = b ? "True" : "False"
_stringify(x::AbstractFloat) = _py_float(x)
_stringify(x) = string(x)

# Python's `repr` of a float: the shortest round-trip digits, positional for
# decimal exponents in [-4, 16) and scientific with a signed two-digit exponent
# otherwise (`1e-05`, `1e+16`).
function _py_float(x::AbstractFloat)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Infinity" : "-Infinity"
    iszero(x) && return signbit(x) ? "-0.0" : "0.0"
    m = match(r"^(-?)(\d+)\.(\d+)(?:e(-?\d+))?$", string(Float64(x)))
    sign, int_part, frac_part, exp_part = m.captures
    digits = int_part * frac_part
    point = length(int_part) + (isnothing(exp_part) ? 0 : parse(Int, exp_part))
    lead = findfirst(!=('0'), digits)
    digits, point = digits[lead:end], point - (lead - 1)
    digits = rstrip(digits, '0')
    exponent = point - 1
    body = if -4 <= exponent < 16
        if point <= 0
            "0." * "0"^(-point) * digits
        elseif point >= length(digits)
            digits * "0"^(point - length(digits)) * ".0"
        else
            digits[1:point] * "." * digits[(point + 1):end]
        end
    else
        mantissa = length(digits) == 1 ? digits : digits[1:1] * "." * digits[2:end]
        mantissa * "e" * (exponent < 0 ? "-" : "+") * lpad(abs(exponent), 2, '0')
    end
    return sign * body
end

# HF's `tojson` is Python's `json.dumps` with its defaults: ", " and ": " between
# items, non-ASCII kept as is, keys in the order the mapping holds them.
_py_json(x) = sprint(_write_py_json, x)

_write_py_json(io::IO, ::Nothing) = print(io, "null")
_write_py_json(io::IO, b::Bool) = print(io, b ? "true" : "false")
_write_py_json(io::IO, n::Integer) = print(io, n)
_write_py_json(io::IO, x::AbstractFloat) = print(io, _py_float(x))
_write_py_json(io::IO, s::Union{AbstractString,Symbol}) = _write_py_string(io, String(s))
function _write_py_json(io::IO, d::Union{AbstractDict,NamedTuple})
    print(io, '{')
    for (i, (k, v)) in enumerate(pairs(d))
        i > 1 && print(io, ", ")
        _write_py_string(io, k isa Union{AbstractString,Symbol} ? String(k) : _stringify(k))
        print(io, ": ")
        _write_py_json(io, v)
    end
    return print(io, '}')
end
function _write_py_json(io::IO, v::Union{AbstractVector,Tuple})
    print(io, '[')
    for (i, item) in enumerate(v)
        i > 1 && print(io, ", ")
        _write_py_json(io, item)
    end
    return print(io, ']')
end
_write_py_json(::IO, x) = throw(ArgumentError("cannot serialize $(typeof(x)) to JSON"))

const _PY_JSON_ESCAPES = Dict(
    '"' => "\\\"", '\\' => "\\\\", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t",
    '\b' => "\\b", '\f' => "\\f",
)
function _write_py_string(io::IO, s::String)
    print(io, '"')
    for c in s
        if haskey(_PY_JSON_ESCAPES, c)
            print(io, _PY_JSON_ESCAPES[c])
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base=16, pad=4))
        else
            print(io, c)
        end
    end
    return print(io, '"')
end

function _get_member(obj::AbstractDict, name)
    sk = String(name)
    haskey(obj, sk) && return obj[sk]
    sym = Symbol(name)
    haskey(obj, sym) && return obj[sym]
    return Undefined()
end
_get_member(ns::Namespace, name) = get(ns.attrs, String(name), Undefined())
function _get_member(obj::NamedTuple, name)
    sym = Symbol(name)
    hasfield(typeof(obj), sym) || return Undefined()
    return getfield(obj, sym)
end
function _get_member(obj, name)
    sym = Symbol(name)
    hasproperty(obj, sym) || return Undefined()
    return getproperty(obj, sym)
end
# Reaching through an undefined value is a template bug, not a skippable branch,
# so this is the one place undefined raises.
function _get_member(::Undefined, name)
    throw(ArgumentError("cannot access `$(name)` on an undefined value"))
end

# Python indexing: 0-based, with negative indices counting from the end.
_py_index(n::Int, key::Integer) = (k = Int(key); (k < 0 ? k + n : k) + 1)
_get_item(obj::AbstractDict, key) = _get_member(obj, key)
_get_item(obj::AbstractVector, key::Integer) = obj[_py_index(length(obj), key)]
function _get_item(obj::AbstractString, key::Integer)
    return string(obj[nextind(obj, 0, _py_index(length(obj), key))])
end
_get_item(obj, key) = _get_member(obj, key)

# The 1-based indices Python's `seq[start:stop:step]` selects, out-of-range
# bounds clamped rather than raising, as Python does.
function _slice_range(n::Int, start, stop, step)
    st = isnothing(step) ? 1 : Int(step)
    st == 0 && throw(ArgumentError("slice step cannot be zero"))
    wrap(i) = (i = Int(i); i < 0 ? i + n : i)
    if st > 0
        lo = isnothing(start) ? 0 : clamp(wrap(start), 0, n)
        hi = isnothing(stop) ? n : clamp(wrap(stop), 0, n)
        return (lo + 1):st:hi
    end
    lo = isnothing(start) ? n - 1 : clamp(wrap(start), -1, n - 1)
    hi = isnothing(stop) ? -1 : clamp(wrap(stop), -1, n - 1)
    return (lo + 1):st:(hi + 2)
end
_slice(obj::AbstractVector, start, stop, step) = obj[_slice_range(length(obj), start, stop, step)]
function _slice(obj::AbstractString, start, stop, step)
    chars = collect(obj)
    return String(chars[_slice_range(length(chars), start, stop, step)])
end

_truthy_eq(a, b) = a == b
_truthy_eq(a::AbstractString, b::AbstractString) = String(a) == String(b)

function _binop(op::Symbol, l, r)
    if op === :+
        if l isa AbstractString || r isa AbstractString
            return string(_stringify(l), _stringify(r))
        end
        return l + r
    elseif op === :-
        return l - r
    elseif op === :*
        return l * r
    elseif op === :/
        return l / r
    elseif op === :%
        return l isa Integer && r isa Integer ? mod(l, r) : l % r
    elseif op === :(==)
        return _truthy_eq(l, r)
    elseif op === :!=
        return !_truthy_eq(l, r)
    elseif op === :<
        return l < r
    elseif op === :>
        return l > r
    elseif op === :<=
        return l <= r
    elseif op === :>=
        return l >= r
    elseif op === :in
        if r isa AbstractString
            return occursin(string(l), r)
        elseif r isa AbstractDict
            return haskey(r, l) || haskey(r, String(l)) || haskey(r, Symbol(l))
        else
            return any(==(l), r)
        end
    else
        throw(ArgumentError("unsupported binary op: $(op)"))
    end
end

function _apply_filter(name::AbstractString, value)
    if name == "trim"
        return strip(_stringify(value))
    elseif name == "tojson"
        return _py_json(value)
    elseif name == "length" || name == "count"
        return length(value)
    else
        throw(ArgumentError("unsupported filter: $(name)"))
    end
end

function eval_expr(e::LiteralExpr, _)
    return e.value
end
function eval_expr(e::NameExpr, ctx::AbstractDict)
    haskey(ctx, e.name) || return Undefined()
    return ctx[e.name]
end
function eval_expr(e::AttrExpr, ctx::AbstractDict)
    return _get_member(eval_expr(e.obj, ctx), e.name)
end
function eval_expr(e::ItemExpr, ctx::AbstractDict)
    return _get_item(eval_expr(e.obj, ctx), eval_expr(e.key, ctx))
end
function eval_expr(e::SliceExpr, ctx::AbstractDict)
    obj = eval_expr(e.obj, ctx)
    start_v = isnothing(e.start) ? nothing : eval_expr(e.start, ctx)
    stop_v = isnothing(e.stop) ? nothing : eval_expr(e.stop, ctx)
    step_v = isnothing(e.step) ? nothing : eval_expr(e.step, ctx)
    return _slice(obj, start_v, stop_v, step_v)
end
function eval_expr(e::BinOpExpr, ctx::AbstractDict)
    return _binop(e.op, eval_expr(e.left, ctx), eval_expr(e.right, ctx))
end
function eval_expr(e::UnaryOpExpr, ctx::AbstractDict)
    v = eval_expr(e.operand, ctx)
    if e.op === :not
        return !_truthy(v)
    elseif e.op === :neg
        return -v
    end
    throw(ArgumentError("unsupported unary op: $(e.op)"))
end
function eval_expr(e::BoolOpExpr, ctx::AbstractDict)
    last = nothing
    for operand in e.operands
        last = eval_expr(operand, ctx)
        if e.op === :and
            _truthy(last) || return last
        elseif e.op === :or
            _truthy(last) && return last
        end
    end
    return last
end
# Jinja's tests, by what they ask of a value. A variable explicitly set to `none`
# is still defined, which is why `defined` asks about the sentinel rather than
# about emptiness; `true` and `false` are identity tests, as in Python.
const _TESTS = Dict{String,Function}(
    "defined" => v -> !(v isa Undefined),
    "undefined" => v -> v isa Undefined,
    "none" => isnothing,
    "true" => v -> v === true,
    "false" => v -> v === false,
    "boolean" => v -> v isa Bool,
    "string" => v -> v isa AbstractString,
    "number" => v -> v isa Number && !(v isa Bool),
    "integer" => v -> v isa Integer && !(v isa Bool),
    "float" => v -> v isa AbstractFloat,
    "mapping" => v -> v isa Union{AbstractDict,NamedTuple},
    "sequence" => v -> v isa Union{AbstractVector,Tuple,AbstractString},
    "iterable" => v -> v isa Union{AbstractVector,Tuple,AbstractString,AbstractDict},
)

function eval_expr(e::TestExpr, ctx::AbstractDict)
    test = get(_TESTS, e.name, nothing)
    isnothing(test) && throw(ArgumentError("unsupported test: is $(e.name)"))
    result = test(eval_expr(e.target, ctx))::Bool
    return e.negated ? !result : result
end
function eval_expr(e::FilterExpr, ctx::AbstractDict)
    v = eval_expr(e.target, ctx)
    return _apply_filter(e.name, v)
end
function eval_expr(e::CallExpr, ctx::AbstractDict)
    args = Any[eval_expr(a, ctx) for a in e.args]
    if e.func == "raise_exception"
        throw(ArgumentError("template raised: $(isempty(args) ? "" : _stringify(args[1]))"))
    elseif e.func == "namespace"
        attrs = Dict{String,Any}(k => eval_expr(v, ctx) for (k, v) in e.kwargs)
        for arg in args
            arg isa AbstractDict || throw(ArgumentError("namespace takes keywords or a mapping"))
            for (k, v) in pairs(arg)
                attrs[String(k)] = v
            end
        end
        return Namespace(attrs)
    end
    throw(ArgumentError("unsupported function: $(e.func)"))
end

# Python's `strip(chars)` removes any of `chars`; with no argument, whitespace.
function _py_strip(f, s::AbstractString, args)
    isempty(args) || isnothing(args[1]) ? String(f(s)) : String(f(s, collect(String(args[1]))))
end

# The Python methods chat templates call on strings and dicts.
function _call_method(obj::AbstractString, name::String, args)
    if name == "startswith"
        return startswith(obj, String(args[1]))
    elseif name == "endswith"
        return endswith(obj, String(args[1]))
    elseif name == "strip"
        return _py_strip(strip, obj, args)
    elseif name == "lstrip"
        return _py_strip(lstrip, obj, args)
    elseif name == "rstrip"
        return _py_strip(rstrip, obj, args)
    elseif name == "split"
        if isempty(args) || isnothing(args[1])
            return String.(split(obj))
        end
        limit = length(args) >= 2 && Int(args[2]) >= 0 ? Int(args[2]) + 1 : 0
        return String.(split(obj, String(args[1]); limit=limit, keepempty=true))
    elseif name == "upper"
        return uppercase(obj)
    elseif name == "lower"
        return lowercase(obj)
    elseif name == "replace"
        return replace(obj, String(args[1]) => String(args[2]))
    end
    throw(ArgumentError("unsupported string method: $(name)"))
end
function _call_method(obj::Union{AbstractDict,Namespace}, name::String, args)
    if name == "get"
        value = _get_member(obj, args[1])
        return value isa Undefined ? (length(args) >= 2 ? args[2] : nothing) : value
    end
    throw(ArgumentError("unsupported mapping method: $(name)"))
end
_call_method(::Undefined, name::String, _) =
    throw(ArgumentError("cannot call `$(name)` on an undefined value"))
_call_method(obj, name::String, _) =
    throw(ArgumentError("unsupported method `$(name)` on $(typeof(obj))"))

function eval_expr(e::MethodCallExpr, ctx::AbstractDict)
    return _call_method(eval_expr(e.obj, ctx), e.name, Any[eval_expr(a, ctx) for a in e.args])
end
