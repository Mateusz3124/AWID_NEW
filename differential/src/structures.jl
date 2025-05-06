abstract type GraphNode end
abstract type Operator <: GraphNode end

import Base: eltype

struct Constant{T} <: GraphNode
    output::T
end

mutable struct Variable{T} <: GraphNode
    output::T
    gradient::Union{Float32,Vector{Float32},Matrix{Float32}}
    Variable(output::T) where {T} = new{T}(output,
        Matrix{Float32}(undef, 0, 0))
end

mutable struct ScalarOperator{F} <: Operator
    inputs::Tuple{Vararg{GraphNode}}
    output::Union{Float32,Vector{Float32},Matrix{Float32}}
    gradient::Union{Float32,Vector{Float32},Matrix{Float32}}
    ScalarOperator(fun, inputs...) = new{typeof(fun)}(inputs,
        Matrix{Float32}(undef, 0, 0), Matrix{Float32}(undef, 0, 0))
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Tuple{Vararg{GraphNode}}
    output::Union{Float32,Vector{Float32},Matrix{Float32}}
    gradient::Union{Float32,Vector{Float32},Matrix{Float32}}
    BroadcastedOperator(fun, inputs...) = new{typeof(fun)}(inputs,
        Matrix{Float32}(undef, 0, 0), Matrix{Float32}(undef, 0, 0))
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "\n ┣━ op (", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "\n ┣━ op. (", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end