mutable struct BlockParser
    blocks::Vector{Tuple{Symbol,String}}
    pos::Int
end

function parse_template(source::AbstractString)
    return _parse_block(BlockParser(_split_blocks(source), 1), Symbol[])
end

function _first_word(s::AbstractString)
    t = strip(s)
    isempty(t) && return ""
    i = findfirst(isspace, t)
    return isnothing(i) ? String(t) : String(t[1:prevind(t, i)])
end

function _parse_block(p::BlockParser, stop_keywords::Vector{Symbol})
    nodes = TmplNode[]
    while p.pos <= length(p.blocks)
        kind, content = p.blocks[p.pos]
        if kind === :stmt && Symbol(_first_word(content)) in stop_keywords
            return nodes
        end
        push!(nodes, _parse_one(p))
    end
    return nodes
end

function _parse_one(p::BlockParser)
    kind, content = p.blocks[p.pos]
    if kind === :text
        p.pos += 1
        return TextNode(content)
    elseif kind === :expr
        p.pos += 1
        return OutputNode(parse_expression(content))
    elseif kind === :stmt
        word = _first_word(content)
        if word == "if"
            return _parse_if(p)
        elseif word == "for"
            return _parse_for(p)
        elseif word == "set"
            return _parse_set(p)
        else
            throw(ArgumentError("unsupported statement at top level: $(strip(content))"))
        end
    else
        throw(ArgumentError("unknown block kind: $(kind)"))
    end
end

function _parse_if(p::BlockParser)
    branches = Tuple{TmplExpr,Vector{TmplNode}}[]
    else_body::Union{Nothing,Vector{TmplNode}} = nothing
    while true
        p.pos <= length(p.blocks) || throw(ArgumentError("unclosed 'if' block"))
        _, content = p.blocks[p.pos]
        trimmed = strip(content)
        if startswith(trimmed, "if ") || startswith(trimmed, "elif ")
            cond_src = replace(String(trimmed), r"^(if|elif)\s+" => "")::AbstractString
            p.pos += 1
            body = _parse_block(p, Symbol[:elif, :else, :endif])
            push!(branches, (parse_expression(cond_src), body))
        elseif trimmed == "else"
            p.pos += 1
            else_body = _parse_block(p, Symbol[:endif])
        elseif trimmed == "endif"
            p.pos += 1
            return IfNode(branches, else_body)
        else
            throw(ArgumentError("unexpected in if block: $(trimmed)"))
        end
    end
end

function _parse_for(p::BlockParser)
    _, content = p.blocks[p.pos]
    m = match(r"^for\s+(\w+)\s+in\s+(.+)$", strip(content))
    isnothing(m) && throw(ArgumentError("invalid for statement: $(content)"))
    var_cap = m.captures[1]::AbstractString
    iter_cap = m.captures[2]::AbstractString
    p.pos += 1
    body = _parse_block(p, Symbol[:endfor])
    p.pos <= length(p.blocks) || throw(ArgumentError("unclosed for block"))
    end_kind, end_content = p.blocks[p.pos]
    end_kind === :stmt && strip(end_content) == "endfor" ||
        throw(ArgumentError("expected endfor, got $(end_content)"))
    p.pos += 1
    return ForNode(String(var_cap), parse_expression(iter_cap), body)
end

function _parse_set(p::BlockParser)
    _, content = p.blocks[p.pos]
    m = match(r"^set\s+(\w+)\s*=\s*(.+)$", strip(content))
    isnothing(m) && throw(ArgumentError("invalid set statement: $(content)"))
    name_cap = m.captures[1]::AbstractString
    expr_cap = m.captures[2]::AbstractString
    p.pos += 1
    return SetNode(String(name_cap), parse_expression(expr_cap))
end
