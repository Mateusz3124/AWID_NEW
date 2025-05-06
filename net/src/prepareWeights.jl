nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])

function glorot_uniform(dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
    (rand(Float32, dims...) .- 0.5f0) .* scale
end

function init_weights(val::Int64)
    return (Variable(glorot_uniform(32, val)), Variable(glorot_uniform(1, 32)))
end