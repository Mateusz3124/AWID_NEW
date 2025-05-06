import Base: *
import LinearAlgebra: mul!

ϵ = 1e-8

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

import LinearAlgebra: diagm
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
    let
        𝟏 = ones(Float32, length(node.output))
        Jx = diagm(vec(y .* 𝟏))
        Jy = diagm(vec(x .* 𝟏))
        tuple(Jx' * g, Jy' * g)
    end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y, g) =
    let
        𝟏 = ones(Float32, length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) =
    let
        𝟏 = ones(Float32, length(node.output))
        Jx = y .* x .^ (y .- 𝟏)
        Jy = log.(abs.(x)) .* x .^ y
        return (Jx * g, Jy * g)
    end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(diagm(vec(Float32(1.0) ./ x))' * g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) =
    return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x::GraphNode)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) =
    let
        𝟏 = ones(Float32, length(x))
        tuple(𝟏 * g)
    end


relu(x::GraphNode) = BroadcastedOperator(relu, x::GraphNode)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(zero(x), x)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))


σ(x) = BroadcastedOperator(σ, x::GraphNode)
forward(::BroadcastedOperator{typeof(σ)}, x::Matrix{Float32}) = return vec(Float32(1.0) ./ (Float32(1.0) .+ exp.(-x)))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) =
    let
        y = node.output'
        𝟏 = ones(Float32, 1, length(y))
        tuple(y .* (𝟏 .- y) .* g')
    end


# binarycrossentropy(y::GraphNode, ŷ::GraphNode) = BroadcastedOperator(binarycrossentropy, y, ŷ)
# forward(::BroadcastedOperator{typeof(binarycrossentropy)}, y, ŷ) =
#     let
#         val = @. -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)
#         return mean(val)
#     end
# backward(::BroadcastedOperator{typeof(binarycrossentropy)}, y, ŷ, g) =
#     let
#         # dy = @. -log(ŷ + ϵ) + log(1 - ŷ + ϵ)
#         # dŷ = @. (1 - y) / (1 - ŷ + ϵ) + (-y) / (ŷ + ϵ)
#         # println(size(ŷ))
#         # println(size(y))
#         # println(size(x'))
#         # println(size(g))
#         # return ((y .- ŷ) .* x .* g)
#         dŷ = ((1 .- y) / (1 .- ŷ .+ ϵ) .+ (-y) / (ŷ .+ ϵ)) ./ size(y, 2)
#         return (dŷ * g)
#         # return (dy * g, dŷ * g)
#     end