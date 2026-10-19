# Rendering returns `nothing`, or `:break` / `:continue` from a loop-control
# statement, which unwinds through any enclosing `if` to the nearest `for`.
function _render_nodes!(io::IO, nodes::Vector{TmplNode}, ctx::AbstractDict)
    for node in nodes
        signal = _render_node!(io, node, ctx)
        isnothing(signal) || return signal
    end
    return nothing
end

_render_node!(io::IO, n::TextNode, ::AbstractDict) = (print(io, n.text); nothing)

function _render_node!(io::IO, n::OutputNode, ctx::AbstractDict)
    print(io, _stringify(eval_expr(n.expr, ctx)))
    return nothing
end

function _render_node!(io::IO, n::IfNode, ctx::AbstractDict)
    for (cond, body) in n.branches
        if _truthy(eval_expr(cond, ctx))
            return _render_nodes!(io, body, ctx)
        end
    end
    isnothing(n.else_body) || return _render_nodes!(io, n.else_body, ctx)
    return nothing
end

# Iterating a mapping yields its keys, as in Python.
_loop_items(d::AbstractDict) = [k isa Symbol ? String(k) : k for k in keys(d)]
_loop_items(iter) = collect(iter)

function _render_node!(io::IO, n::ForNode, ctx::AbstractDict)
    items = _loop_items(eval_expr(n.iter, ctx))
    total = length(items)
    for (i, item) in enumerate(items)
        local_ctx = copy(ctx)
        if length(n.vars) == 1
            local_ctx[only(n.vars)] = item
        else
            values = collect(item)
            length(values) == length(n.vars) || throw(
                ArgumentError("cannot unpack $(length(values)) values into $(join(n.vars, ", "))"),
            )
            for (name, value) in zip(n.vars, values)
                local_ctx[name] = value
            end
        end
        local_ctx["loop"] = Dict{String,Any}(
            "index" => i,
            "index0" => i - 1,
            "revindex" => total - i + 1,
            "revindex0" => total - i,
            "first" => i == 1,
            "last" => i == total,
            "length" => total,
            "previtem" => i > 1 ? items[i - 1] : Undefined(),
            "nextitem" => i < total ? items[i + 1] : Undefined(),
        )
        _render_nodes!(io, n.body, local_ctx) === :break && break
    end
    return nothing
end

function _render_node!(io::IO, n::SetNode, ctx::AbstractDict)
    value = eval_expr(n.expr, ctx)
    if isnothing(n.attr)
        ctx[n.name] = value
    else
        target = get(ctx, n.name, Undefined())
        target isa Namespace ||
            throw(ArgumentError("`set $(n.name).$(n.attr)` needs `$(n.name)` to be a namespace"))
        target.attrs[n.attr] = value
    end
    return nothing
end

_render_node!(::IO, n::LoopControlNode, ::AbstractDict) = n.kind

# Defining a macro binds a callable in the current scope. It keeps that scope by
# reference, so a call sees the template's variables and every macro defined at
# top level, including itself for recursion.
function _render_node!(::IO, n::MacroNode, ctx::AbstractDict)
    ctx[n.name] = TemplateMacro(n, ctx)
    return nothing
end
