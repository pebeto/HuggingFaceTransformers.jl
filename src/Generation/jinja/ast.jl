abstract type TmplExpr end

struct LiteralExpr{T} <: TmplExpr
    value::T
end
struct NameExpr <: TmplExpr
    name::String
end
struct AttrExpr <: TmplExpr
    obj::TmplExpr
    name::String
end
struct ItemExpr <: TmplExpr
    obj::TmplExpr
    key::TmplExpr
end
struct SliceExpr <: TmplExpr
    obj::TmplExpr
    start::Union{Nothing,TmplExpr}
    stop::Union{Nothing,TmplExpr}
end
struct BinOpExpr <: TmplExpr
    op::Symbol
    left::TmplExpr
    right::TmplExpr
end
struct UnaryOpExpr <: TmplExpr
    op::Symbol
    operand::TmplExpr
end
struct BoolOpExpr <: TmplExpr
    op::Symbol
    operands::Vector{TmplExpr}
end
struct TestExpr <: TmplExpr
    target::TmplExpr
    name::String
    negated::Bool
end
struct FilterExpr <: TmplExpr
    target::TmplExpr
    name::String
end
struct CallExpr <: TmplExpr
    func::String
    args::Vector{TmplExpr}
end

abstract type TmplNode end

struct TextNode <: TmplNode
    text::String
end
struct OutputNode <: TmplNode
    expr::TmplExpr
end
struct IfNode <: TmplNode
    branches::Vector{Tuple{TmplExpr,Vector{TmplNode}}}
    else_body::Union{Nothing,Vector{TmplNode}}
end
struct ForNode <: TmplNode
    var::String
    iter::TmplExpr
    body::Vector{TmplNode}
end
struct SetNode <: TmplNode
    name::String
    expr::TmplExpr
end
