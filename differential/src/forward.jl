reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = Matrix{Float32}(undef, 0, 0)
reset!(node::Operator) = node.gradient = Matrix{Float32}(undef, 0, 0)

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end