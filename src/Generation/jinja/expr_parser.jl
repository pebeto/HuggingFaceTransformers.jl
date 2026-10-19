mutable struct ExprParser
    tokens::Vector{ExprToken}
    pos::Int
end

_peek(p::ExprParser, ahead::Int=0) = p.tokens[p.pos + ahead]

function _check(p::ExprParser, kind::Symbol, value=nothing)
    t = _peek(p)
    return t.kind === kind && (value === nothing || t.value == value)
end

function _consume!(p::ExprParser)
    t = p.tokens[p.pos]
    p.pos += 1
    return t
end

function _eat!(p::ExprParser, kind::Symbol, value=nothing)
    _check(p, kind, value) || throw(
        ArgumentError(
            "expected $(kind)$(value === nothing ? "" : "($(value))"); got $(_peek(p).kind) $(_peek(p).value)",
        ),
    )
    return _consume!(p)
end

function parse_expression(s::AbstractString)
    p = ExprParser(_tokenize_expr(s), 1)
    e = _parse_conditional(p)
    _eat!(p, :eof)
    return e
end

# `then if cond else otherwise`, Jinja's inline if, binds loosest of all.
function _parse_conditional(p)
    e = _parse_or(p)
    _check(p, :name, "if") || return e
    _consume!(p)
    cond = _parse_or(p)
    otherwise = nothing
    if _check(p, :name, "else")
        _consume!(p)
        otherwise = _parse_conditional(p)
    end
    return ConditionalExpr(cond, e, otherwise)
end

# A call's arguments: positional ones, then `name=value` keywords.
function _parse_call_args!(p)
    args = TmplExpr[]
    kwargs = Pair{String,TmplExpr}[]
    while !_check(p, :rparen)
        if _check(p, :name) && _peek(p, 1).kind === :assign
            name = _consume!(p).value
            _consume!(p)
            push!(kwargs, name => _parse_conditional(p))
        else
            push!(args, _parse_conditional(p))
        end
        _check(p, :rparen) || _eat!(p, :comma)
    end
    _eat!(p, :rparen)
    return args, kwargs
end

function _parse_or(p)
    e = _parse_and(p)
    while _check(p, :keyword, "or")
        _consume!(p)
        e = BoolOpExpr(:or, TmplExpr[e, _parse_and(p)])
    end
    return e
end

function _parse_and(p)
    e = _parse_not(p)
    while _check(p, :keyword, "and")
        _consume!(p)
        e = BoolOpExpr(:and, TmplExpr[e, _parse_not(p)])
    end
    return e
end

function _parse_not(p)
    if _check(p, :keyword, "not")
        _consume!(p)
        return UnaryOpExpr(:not, _parse_not(p))
    end
    return _parse_compare(p)
end

function _parse_compare(p)
    left = _parse_add(p)
    while true
        if _check(p, :op, "==") ||
            _check(p, :op, "!=") ||
            _check(p, :op, "<") ||
            _check(p, :op, ">") ||
            _check(p, :op, "<=") ||
            _check(p, :op, ">=")
            op = Symbol(_consume!(p).value)
            right = _parse_add(p)
            left = BinOpExpr(op, left, right)
        elseif _check(p, :keyword, "in")
            _consume!(p)
            right = _parse_add(p)
            left = BinOpExpr(:in, left, right)
        elseif _check(p, :keyword, "not") &&
            _peek(p, 1).kind === :keyword &&
            _peek(p, 1).value == "in"
            _consume!(p)
            _consume!(p)
            right = _parse_add(p)
            left = UnaryOpExpr(:not, BinOpExpr(:in, left, right))
        elseif _check(p, :keyword, "is")
            _consume!(p)
            negated = false
            if _check(p, :keyword, "not")
                _consume!(p)
                negated = true
            end
            # `none`, `true` and `false` lex as literals rather than names, so
            # those tests need their own arms; the rest arrive as names.
            test_name = if _check(p, :none)
                _consume!(p)
                "none"
            elseif _check(p, :bool)
                _consume!(p).value ? "true" : "false"
            else
                _eat!(p, :name).value
            end
            left = TestExpr(left, test_name, negated)
        else
            break
        end
    end
    return left
end

function _parse_add(p)
    e = _parse_mul(p)
    while _check(p, :op, "+") || _check(p, :op, "-")
        op = Symbol(_consume!(p).value)
        e = BinOpExpr(op, e, _parse_mul(p))
    end
    return e
end

function _parse_mul(p)
    e = _parse_unary(p)
    while _check(p, :op, "*") || _check(p, :op, "/") || _check(p, :op, "%")
        op = Symbol(_consume!(p).value)
        e = BinOpExpr(op, e, _parse_unary(p))
    end
    return e
end

function _parse_unary(p)
    if _check(p, :op, "-")
        _consume!(p)
        return UnaryOpExpr(:neg, _parse_unary(p))
    end
    return _parse_filter(p)
end

function _parse_filter(p)
    e = _parse_postfix(p)
    while _check(p, :pipe)
        _consume!(p)
        name_tok = _eat!(p, :name)
        if _check(p, :lparen)
            _consume!(p)
            args, kwargs = _parse_call_args!(p)
            e = FilterExpr(e, name_tok.value, args, kwargs)
        else
            e = FilterExpr(e, name_tok.value)
        end
    end
    return e
end

function _parse_postfix(p)
    e = _parse_atom(p)
    while true
        if _check(p, :dot)
            _consume!(p)
            name_tok = _eat!(p, :name)
            e = AttrExpr(e, name_tok.value)
        elseif _check(p, :lbracket)
            _consume!(p)
            # `[i]`, or a Python slice `[start:stop:step]` with every part optional.
            first_expr = _check(p, :colon) ? nothing : _parse_or(p)
            if _check(p, :colon)
                _consume!(p)
                stop = _check(p, :colon) || _check(p, :rbracket) ? nothing : _parse_or(p)
                step = nothing
                if _check(p, :colon)
                    _consume!(p)
                    step = _check(p, :rbracket) ? nothing : _parse_or(p)
                end
                _eat!(p, :rbracket)
                e = SliceExpr(e, first_expr, stop, step)
            else
                _eat!(p, :rbracket)
                e = ItemExpr(e, first_expr)
            end
        elseif _check(p, :lparen)
            _consume!(p)
            args, kwargs = _parse_call_args!(p)
            if e isa NameExpr
                e = CallExpr(e.name, args, kwargs)
            elseif e isa AttrExpr && isempty(kwargs)
                e = MethodCallExpr(e.obj, e.name, args)
            else
                throw(ArgumentError("cannot call $(typeof(e)) with these arguments"))
            end
        else
            break
        end
    end
    return e
end

function _parse_atom(p)
    t = _peek(p)
    if t.kind === :string || t.kind === :number || t.kind === :bool || t.kind === :none
        _consume!(p)
        return LiteralExpr(t.value)
    elseif t.kind === :name
        _consume!(p)
        return NameExpr(t.value)
    elseif t.kind === :lparen
        _consume!(p)
        e = _parse_conditional(p)
        _eat!(p, :rparen)
        return e
    elseif t.kind === :lbracket
        _consume!(p)
        items = TmplExpr[]
        while !_check(p, :rbracket)
            push!(items, _parse_conditional(p))
            _check(p, :rbracket) || _eat!(p, :comma)
        end
        _eat!(p, :rbracket)
        return ListExpr(items)
    end
    throw(ArgumentError("unexpected token in expression: $(t.kind) $(t.value)"))
end
