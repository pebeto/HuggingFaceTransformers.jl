function _render_nodes!(io::IO, nodes::Vector{TmplNode}, ctx::AbstractDict)
    for node in nodes
        _render_node!(io, node, ctx)
    end
    return nothing
end

_render_node!(io::IO, n::TextNode, ::AbstractDict) = print(io, n.text)

function _render_node!(io::IO, n::OutputNode, ctx::AbstractDict)
    return print(io, _stringify(eval_expr(n.expr, ctx)))
end

function _render_node!(io::IO, n::IfNode, ctx::AbstractDict)
    for (cond, body) in n.branches
        if _truthy(eval_expr(cond, ctx))
            _render_nodes!(io, body, ctx)
            return nothing
        end
    end
    if !isnothing(n.else_body)
        _render_nodes!(io, n.else_body, ctx)
    end
    return nothing
end

function _render_node!(io::IO, n::ForNode, ctx::AbstractDict)
    iter = eval_expr(n.iter, ctx)
    items = collect(iter)
    total = length(items)
    for (i, item) in enumerate(items)
        local_ctx = copy(ctx)
        local_ctx[n.var] = item
        local_ctx["loop"] = Dict{String,Any}(
            "index" => i,
            "index0" => i - 1,
            "first" => i == 1,
            "last" => i == total,
            "length" => total,
        )
        _render_nodes!(io, n.body, local_ctx)
    end
    return nothing
end

function _render_node!(io::IO, n::SetNode, ctx::AbstractDict)
    ctx[n.name] = eval_expr(n.expr, ctx)
    return nothing
end
