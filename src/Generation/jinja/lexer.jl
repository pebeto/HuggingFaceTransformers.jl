struct ExprToken
    kind::Symbol
    value::Any
end

const _EXPR_KEYWORDS = Set([
    "and", "or", "not", "in", "is", "true", "True", "false", "False", "none", "None", "null"
])

function _tokenize_expr(s::AbstractString)
    tokens = ExprToken[]
    i = 1
    n = sizeof(s)
    while i <= n
        c = s[i]
        if isspace(c)
            i = nextind(s, i)
            continue
        elseif c == '\'' || c == '"'
            qc = c
            j = nextind(s, i)
            buf = IOBuffer()
            while j <= n && s[j] != qc
                if s[j] == '\\' && nextind(s, j) <= n
                    nxt = nextind(s, j)
                    esc = s[nxt]
                    print(
                        buf,
                        if esc == 'n'
                            '\n'
                        elseif esc == 't'
                            '\t'
                        elseif esc == 'r'
                            '\r'
                        elseif esc == '\\'
                            '\\'
                        elseif esc == '\''
                            '\''
                        elseif esc == '"'
                            '"'
                        else
                            esc
                        end,
                    )
                    j = nextind(s, nxt)
                else
                    print(buf, s[j])
                    j = nextind(s, j)
                end
            end
            j <= n || throw(ArgumentError("unterminated string in expression: $(s)"))
            push!(tokens, ExprToken(:string, String(take!(buf))))
            i = nextind(s, j)
        elseif isdigit(c)
            j = i
            while j <= n && isdigit(s[j])
                j = nextind(s, j)
            end
            push!(tokens, ExprToken(:number, parse(Int, s[i:prevind(s, j)])))
            i = j
        elseif isletter(c) || c == '_'
            j = i
            while j <= n && (isletter(s[j]) || isdigit(s[j]) || s[j] == '_')
                j = nextind(s, j)
            end
            word = s[i:prevind(s, j)]
            if word in ("true", "True")
                push!(tokens, ExprToken(:bool, true))
            elseif word in ("false", "False")
                push!(tokens, ExprToken(:bool, false))
            elseif word in ("none", "None", "null")
                push!(tokens, ExprToken(:none, nothing))
            elseif word in _EXPR_KEYWORDS
                push!(tokens, ExprToken(:keyword, word))
            else
                push!(tokens, ExprToken(:name, word))
            end
            i = j
        elseif c == '='
            nxt = nextind(s, i)
            if nxt <= n && s[nxt] == '='
                push!(tokens, ExprToken(:op, "=="))
                i = nextind(s, nxt)
            else
                push!(tokens, ExprToken(:assign, "="))
                i = nxt
            end
        elseif c == '!'
            nxt = nextind(s, i)
            if nxt <= n && s[nxt] == '='
                push!(tokens, ExprToken(:op, "!="))
                i = nextind(s, nxt)
            else
                throw(ArgumentError("unexpected '!' in expression: $(s)"))
            end
        elseif c == '<' || c == '>'
            nxt = nextind(s, i)
            if nxt <= n && s[nxt] == '='
                push!(tokens, ExprToken(:op, c == '<' ? "<=" : ">="))
                i = nextind(s, nxt)
            else
                push!(tokens, ExprToken(:op, string(c)))
                i = nxt
            end
        elseif c in ('+', '-', '*', '/', '%')
            push!(tokens, ExprToken(:op, string(c)))
            i = nextind(s, i)
        elseif c == '('
            push!(tokens, ExprToken(:lparen, "("))
            i = nextind(s, i)
        elseif c == ')'
            push!(tokens, ExprToken(:rparen, ")"))
            i = nextind(s, i)
        elseif c == '['
            push!(tokens, ExprToken(:lbracket, "["))
            i = nextind(s, i)
        elseif c == ']'
            push!(tokens, ExprToken(:rbracket, "]"))
            i = nextind(s, i)
        elseif c == ','
            push!(tokens, ExprToken(:comma, ","))
            i = nextind(s, i)
        elseif c == '.'
            push!(tokens, ExprToken(:dot, "."))
            i = nextind(s, i)
        elseif c == '|'
            push!(tokens, ExprToken(:pipe, "|"))
            i = nextind(s, i)
        elseif c == ':'
            push!(tokens, ExprToken(:colon, ":"))
            i = nextind(s, i)
        else
            throw(ArgumentError("unexpected character $(c) in expression: $(s)"))
        end
    end
    push!(tokens, ExprToken(:eof, nothing))
    return tokens
end
