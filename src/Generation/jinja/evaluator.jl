_truthy(::Nothing) = false
_truthy(b::Bool) = b
_truthy(s::AbstractString) = !isempty(s)
_truthy(n::Number) = n != 0
_truthy(c::AbstractArray) = !isempty(c)
_truthy(d::AbstractDict) = !isempty(d)
_truthy(_) = true

_stringify(::Nothing) = ""
_stringify(s::AbstractString) = s
_stringify(b::Bool) = b ? "True" : "False"
_stringify(x) = string(x)

function _get_member(obj::AbstractDict, name)
    sk = String(name)
    haskey(obj, sk) && return obj[sk]
    sym = Symbol(name)
    haskey(obj, sym) && return obj[sym]
    throw(KeyError(name))
end
function _get_member(obj::NamedTuple, name)
    sym = Symbol(name)
    hasfield(typeof(obj), sym) || throw(KeyError(name))
    return getfield(obj, sym)
end
function _get_member(obj, name)
    sym = Symbol(name)
    hasproperty(obj, sym) || throw(KeyError(name))
    return getproperty(obj, sym)
end

_get_item(obj::AbstractDict, key) = _get_member(obj, key)
_get_item(obj::AbstractVector, key::Integer) = obj[Int(key) + 1]
_get_item(obj::AbstractString, key::Integer) = string(obj[nextind(obj, 0, Int(key) + 1)])
_get_item(obj, key) = _get_member(obj, key)

function _slice(obj::AbstractVector, start, stop)
    s = isnothing(start) ? 1 : Int(start) + 1
    e = isnothing(stop) ? length(obj) : Int(stop)
    return obj[s:e]
end
function _slice(obj::AbstractString, start, stop)
    s = isnothing(start) ? 1 : Int(start) + 1
    e = isnothing(stop) ? length(obj) : Int(stop)
    return obj[s:e]
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
        return JSON3.write(value)
    else
        throw(ArgumentError("unsupported filter: $(name)"))
    end
end

function eval_expr(e::LiteralExpr, _)
    return e.value
end
function eval_expr(e::NameExpr, ctx::AbstractDict)
    haskey(ctx, e.name) || throw(KeyError(e.name))
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
    return _slice(obj, start_v, stop_v)
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
function eval_expr(e::TestExpr, ctx::AbstractDict)
    if e.name == "defined"
        result = if e.target isa NameExpr
            haskey(ctx, e.target.name)
        else
            try
                eval_expr(e.target, ctx)
                true
            catch err
                err isa KeyError ? false : rethrow(err)
            end
        end
        return e.negated ? !result : result
    elseif e.name == "none"
        result = try
            isnothing(eval_expr(e.target, ctx))
        catch err
            err isa KeyError ? false : rethrow(err)
        end
        return e.negated ? !result : result
    end
    throw(ArgumentError("unsupported test: is $(e.name)"))
end
function eval_expr(e::FilterExpr, ctx::AbstractDict)
    v = eval_expr(e.target, ctx)
    return _apply_filter(e.name, v)
end
function eval_expr(e::CallExpr, ctx::AbstractDict)
    args = Any[eval_expr(a, ctx) for a in e.args]
    if e.func == "raise_exception"
        throw(ArgumentError("template raised: $(isempty(args) ? "" : _stringify(args[1]))"))
    end
    throw(ArgumentError("unsupported function: $(e.func)"))
end
