using differential: Variable, Constant, Operator, relu, σ, topological_sort

function dense(w, x, activation)
    return activation(w * x)
end

function binarycrossentropy(y::Variable, ŷ::Operator)
    sizeConst = Constant(Float32(length(y.output)))
    ϵ = Constant(Float32(1e-8))
    zero = Constant(Float32(0.0))
    one = Constant(Float32(1.0))
    return sum((zero .- (y .* log.(ŷ .+ ϵ))) .- (one .- y) .* (log.(one .- ŷ .+ ϵ))) ./ sizeConst
end

function graphBuild(x::Variable, y::Variable, Wh::Variable, Wo::Variable)
    x̂ = dense(Wh, x, relu)
    ŷ = dense(Wo, x̂, σ)
    E = binarycrossentropy(y, ŷ)
    return (topological_sort(E), ŷ)
end