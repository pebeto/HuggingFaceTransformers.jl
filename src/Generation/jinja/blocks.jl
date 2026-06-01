function _split_blocks(source::AbstractString)
    raw = Tuple{Symbol,String,Bool,Bool}[]
    i = 1
    n = sizeof(source)
    tag_pairs = (("{{", :expr), ("{%", :stmt), ("{#", :comment))

    while i <= n
        best_pos = nothing
        best_kind = :text
        best_len = 0
        for (tag, kind) in tag_pairs
            r = findnext(tag, source, i)
            if !isnothing(r) && (isnothing(best_pos) || first(r) < best_pos)
                best_pos = first(r)
                best_kind = kind
                best_len = ncodeunits(tag)
            end
        end

        if isnothing(best_pos)
            push!(raw, (:text, source[i:end], false, false))
            break
        end

        if best_pos > i
            push!(raw, (:text, source[i:prevind(source, best_pos)], false, false))
        end

        after_open = best_pos + best_len
        lstrip_flag = false
        if after_open <= n && source[after_open] == '-'
            lstrip_flag = true
            after_open = nextind(source, after_open)
        end

        close_str = if best_kind === :expr
            "}}"
        elseif best_kind === :stmt
            "%}"
        else
            "#}"
        end
        close_r = findnext(close_str, source, after_open)
        isnothing(close_r) &&
            throw(ArgumentError("unclosed template tag starting at byte $(best_pos)"))

        content_end = prevind(source, first(close_r))
        rstrip_flag = false
        if content_end >= after_open && source[content_end] == '-'
            rstrip_flag = true
            content_end = prevind(source, content_end)
        end
        content = after_open <= content_end ? source[after_open:content_end] : ""

        if best_kind !== :comment
            push!(raw, (best_kind, content, lstrip_flag, rstrip_flag))
        end

        i = nextind(source, last(close_r))
    end

    # Apply whitespace control: a tag with `lstrip` strips the trailing
    # whitespace of the previous text block; `rstrip` strips the leading
    # whitespace of the next text block.
    final = Tuple{Symbol,String}[]
    for (idx, (kind, content, _, _)) in enumerate(raw)
        if kind === :text
            text = content
            if idx > 1
                _, _, _, prev_rs = raw[idx - 1]
                prev_rs && (text = lstrip(text))
            end
            if idx < length(raw)
                _, _, next_ls, _ = raw[idx + 1]
                next_ls && (text = rstrip(text))
            end
            push!(final, (:text, text))
        else
            push!(final, (kind, content))
        end
    end
    return final
end
