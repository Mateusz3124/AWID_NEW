update!(::Constant, _) = nothing
update!(node::GraphNode, gradient) = if isempty(node.gradient)
        node.gradient = gradient
    else
        node.gradient .+= gradient
    end

function backward!(order::Vector; seed=Float32(1.0))
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    p = [input.output for input in inputs]
    gradients = backward(node, p..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end