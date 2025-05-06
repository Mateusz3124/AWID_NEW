import LinearAlgebra

function initData(x::Matrix{Float32}, y::Vector{Float32}, batchNum::Int64, batch_size::Int)
    response = Vector{Tuple{Matrix{Float32},Vector{Float32}}}(undef, batchNum)
    for i in 1:batchNum
        batch_indices = ((i-1)*batch_size+1):(i*batch_size)
        @views x_batch = x[:, batch_indices]
        @views y_batch = y[batch_indices]
        response[i] = (x_batch, y_batch)
    end
    return response
end