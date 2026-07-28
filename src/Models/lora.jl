"""
    LoRALinear{L, A, B}

A `Linear` (`base`, frozen) plus a low-rank update: `y = base(x) + scaling · B(Ax)`,
where `lora_A` is `(r, in)` and `lora_B` is `(out, r)`. `scaling` is `alpha/r`
(or `alpha/√r` for rsLoRA). Only `lora_A`/`lora_B` are trainable.
"""
struct LoRALinear{L,A,B}
    base::L
    lora_A::A
    lora_B::B
    scaling::Float32
end

function (l::LoRALinear)(x::AbstractArray)
    base_out = l.base(x)
    sz = size(x)
    delta = l.lora_B * (l.lora_A * reshape(x, sz[1], :))
    return base_out .+ l.scaling .* reshape(delta, size(l.lora_B, 1), sz[2:end]...)
end

Flux.@layer LoRALinear
Flux.Optimisers.trainable(l::LoRALinear) = (; lora_A=l.lora_A, lora_B=l.lora_B)

"""
    lora_wrap(base::Linear, r, alpha; use_rslora=false, init=Flux.glorot_uniform) -> LoRALinear

Wrap `base` with a fresh rank-`r` adapter. `lora_B` starts at zero (PEFT
convention), so the initial update is a no-op.
"""
function lora_wrap(
    base::Linear, r::Integer, alpha::Real; use_rslora::Bool=false, init=Flux.glorot_uniform
)
    out, inn = size(base.weight)
    scaling = Float32(use_rslora ? alpha / sqrt(r) : alpha / r)
    return LoRALinear(base, Float32.(init(r, inn)), zeros(Float32, out, r), scaling)
end

"""
    merge_lora(l::LoRALinear) -> Linear

Fold the update into a plain `Linear`: `W + scaling · (B·A)`. For inference
deployment (PEFT's `merge_and_unload`).
"""
function merge_lora(l::LoRALinear)
    W = l.base.weight .+ l.scaling .* (l.lora_B * l.lora_A)
    return Linear(W, l.base.bias === nothing ? nothing : copy(l.base.bias))
end

"""
    LoraConfig

PEFT LoRA adapter configuration. `scaling` derives from `lora_alpha`/`r`
(`lora_alpha`/√`r` when `use_rslora`).
"""
Base.@kwdef struct LoraConfig
    r::Int
    lora_alpha::Float64
    target_modules::Vector{String}
    use_rslora::Bool = false
    lora_dropout::Float64 = 0.0
    bias::String = "none"
    base_model_name_or_path::String = ""
    task_type::String = "CAUSAL_LM"
end

"""
    lora_scaling(c::LoraConfig) -> Float32

The multiplier applied to the low-rank update `B*A`. Standard LoRA uses
`lora_alpha / r`; rank-stabilized LoRA (`use_rslora=true`) uses
`lora_alpha / sqrt(r)` so the update's magnitude stays comparable as `r` grows.
"""
lora_scaling(c::LoraConfig) =
    Float32(c.use_rslora ? c.lora_alpha / sqrt(c.r) : c.lora_alpha / c.r)

# Minimal safetensors writer (SafeTensors.jl only reads). Writes F32 tensors in
# C-order so a standard safetensors reader reloads the same logical arrays.
function _save_safetensors(
    path::AbstractString,
    tensors::AbstractDict{String,<:AbstractArray};
    metadata::AbstractDict{String,String}=Dict("format" => "pt"),
)
    header = Dict{String,Any}()
    bufs = Vector{UInt8}[]
    offset = 0
    for name in sort(collect(keys(tensors)))
        a = Float32.(tensors[name])
        # C-order bytes = column-major of the dim-reversed array.
        c = ndims(a) <= 1 ? a : permutedims(a, reverse(ntuple(identity, ndims(a))))
        bytes = collect(reinterpret(UInt8, vec(collect(c))))
        header[name] = Dict(
            "dtype" => "F32",
            "shape" => collect(Int, size(a)),
            "data_offsets" => [offset, offset + length(bytes)],
        )
        push!(bufs, bytes)
        offset += length(bytes)
    end
    header["__metadata__"] = metadata

    hjson = Vector{UInt8}(codeunits(JSON3.write(header)))
    pad = (8 - length(hjson) % 8) % 8
    append!(hjson, fill(UInt8(' '), pad))   # safetensors pads the header to 8 bytes

    open(path, "w") do io
        write(io, htol(UInt64(length(hjson))))
        write(io, hjson)
        for b in bufs
            write(io, b)
        end
    end
    return path
end

"""
    save_lora(adapter_dir, config::LoraConfig, module_weights) -> adapter_dir

Write a PEFT-format adapter (`adapter_config.json` + `adapter_model.safetensors`).
`module_weights` maps each target module path (e.g.
`"model.layers.0.self_attn.q_proj"`) to a `(lora_A, lora_B)` tuple.
"""
function save_lora(
    adapter_dir::AbstractString, config::LoraConfig, module_weights::AbstractDict
)
    mkpath(adapter_dir)
    tensors = Dict{String,Array{Float32}}()
    for (mp, (A, B)) in module_weights
        tensors["base_model.model.$(mp).lora_A.weight"] = Float32.(A)
        tensors["base_model.model.$(mp).lora_B.weight"] = Float32.(B)
    end
    _save_safetensors(joinpath(adapter_dir, "adapter_model.safetensors"), tensors)

    cfg = Dict{String,Any}(
        "peft_type" => "LORA",
        "r" => config.r,
        "lora_alpha" => config.lora_alpha,
        "target_modules" => config.target_modules,
        "use_rslora" => config.use_rslora,
        "lora_dropout" => config.lora_dropout,
        "bias" => config.bias,
        "base_model_name_or_path" => config.base_model_name_or_path,
        "task_type" => config.task_type,
    )
    open(joinpath(adapter_dir, "adapter_config.json"), "w") do io
        write(io, JSON3.write(cfg))
    end
    return adapter_dir
end

"""
    load_lora(adapter_dir) -> (config::LoraConfig, weights::Dict)

Read a PEFT adapter directory. `weights` is the raw safetensors dict keyed on
`base_model.model.<module>.lora_{A,B}.weight`.
"""
function load_lora(adapter_dir::AbstractString)
    cfg_path = joinpath(adapter_dir, "adapter_config.json")
    w_path = joinpath(adapter_dir, "adapter_model.safetensors")
    isfile(cfg_path) || throw(ArgumentError("no adapter_config.json in $(adapter_dir)"))
    isfile(w_path) || throw(ArgumentError("no adapter_model.safetensors in $(adapter_dir)"))

    raw = JSON3.read(read(cfg_path, String))
    config = LoraConfig(;
        r=Int(raw.r),
        lora_alpha=Float64(raw.lora_alpha),
        target_modules=String[String(t) for t in raw.target_modules],
        use_rslora=Bool(get(raw, :use_rslora, false)),
        lora_dropout=Float64(get(raw, :lora_dropout, 0.0)),
        bias=String(get(raw, :bias, "none")),
        base_model_name_or_path=String(get(raw, :base_model_name_or_path, "")),
        task_type=String(get(raw, :task_type, "CAUSAL_LM")),
    )
    return config, load_safetensors(w_path; mmap=false)
end

# Group PEFT adapter weights by module path: mp => (A, B).
function _group_lora(weights::AbstractDict)
    prefix = "base_model.model."
    As = Dict{String,Any}()
    Bs = Dict{String,Any}()
    for (k, v) in weights
        startswith(k, prefix) || continue
        rest = k[(length(prefix) + 1):end]
        if endswith(rest, ".lora_A.weight")
            As[rest[1:(end - length(".lora_A.weight"))]] = v
        elseif endswith(rest, ".lora_B.weight")
            Bs[rest[1:(end - length(".lora_B.weight"))]] = v
        end
    end
    return Dict(mp => (As[mp], Bs[mp]) for mp in keys(As))
end

"""
    apply_lora!(model, adapter_dir, state_dict_map) -> Int

Merge a PEFT adapter into `model` in place (PEFT's `merge_and_unload`), returning
the number of modules merged. Each adapted module's base weight is located
through `state_dict_map` (e.g. `llama_state_dict_map(model.config)`); the update
`scaling · (B·A)` is added to it.
"""
function apply_lora!(model, adapter_dir::AbstractString, state_dict_map::AbstractDict)
    config, weights = load_lora(adapter_dir)
    scaling = lora_scaling(config)
    applied = 0
    for (mp, (A, B)) in _group_lora(weights)
        wkey = "$(mp).weight"
        haskey(state_dict_map, wkey) ||
            throw(ArgumentError("LoRA target $(mp): no base weight $(wkey) in the map"))
        path, transform = state_dict_map[wkey]
        transform === :as_is ||
            throw(ArgumentError("LoRA on a transposed weight ($(wkey)) is not supported"))
        linear = _resolve(model, path[1:(end - 1)])
        delta = scaling .* (B * A)
        size(delta) == size(linear.weight) || throw(
            DimensionMismatch(
                "LoRA $(mp): update $(size(delta)) ≠ weight $(size(linear.weight))"
            ),
        )
        linear.weight .+= delta
        applied += 1
    end
    return applied
end
