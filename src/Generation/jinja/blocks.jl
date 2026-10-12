# Splits a template into text and tag blocks, applying whitespace control the way
# HF's Jinja environment does. A `-` inside a tag strips all whitespace on that
# side. HF also turns on `trim_blocks` (a block or comment tag eats the newline
# after it) and `lstrip_blocks` (it eats the indentation before it on its line);
# a `+` in place of the `-` opts that side out.
function _split_blocks(source::AbstractString)
    raw = Tuple{Symbol,String,Symbol,Symbol}[]   # kind, content, left/right modifier
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
            push!(raw, (:text, source[i:end], :none, :none))
            break
        end

        if best_pos > i
            push!(raw, (:text, source[i:prevind(source, best_pos)], :none, :none))
        end

        after_open = best_pos + best_len
        left = :none
        if after_open <= n && (source[after_open] == '-' || source[after_open] == '+')
            left = source[after_open] == '-' ? :minus : :plus
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
        right = :none
        if content_end >= after_open && (source[content_end] == '-' || source[content_end] == '+')
            right = source[content_end] == '-' ? :minus : :plus
            content_end = prevind(source, content_end)
        end
        content = after_open <= content_end ? source[after_open:content_end] : ""
        push!(raw, (best_kind, content, left, right))

        i = nextind(source, last(close_r))
    end

    final = Tuple{Symbol,String}[]
    for (idx, (kind, content, _, _)) in enumerate(raw)
        if kind === :comment
            continue
        elseif kind !== :text
            push!(final, (kind, content))
            continue
        end
        text = content
        if idx > 1
            prev_kind, _, _, prev_right = raw[idx - 1]
            if prev_right === :minus
                text = lstrip(text)
            elseif prev_kind !== :expr && prev_right !== :plus
                text = chopprefix(chopprefix(text, "\r\n"), "\n")
            end
        end
        if idx < length(raw)
            next_kind, _, next_left, _ = raw[idx + 1]
            if next_left === :minus
                text = rstrip(text)
            elseif next_kind !== :expr && next_left !== :plus
                # Only indentation counts: the tag must start its line, so judge on
                # the source text, before trimming removed the newline.
                newline = findlast('\n', content)
                indent = isnothing(newline) ? content : content[(newline + 1):end]
                at_line_start = !isnothing(newline) || idx == 1
                if at_line_start && all(c -> c == ' ' || c == '\t', indent)
                    text = chopsuffix(text, indent)
                end
            end
        end
        push!(final, (:text, String(text)))
    end
    return final
end
