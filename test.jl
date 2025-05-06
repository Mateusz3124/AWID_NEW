

include("./differential/structures.jl")
include("./differential/operators.jl")
include("./differential/building.jl")
include("./differential/forward.jl")
include("./differential/backward.jl")

include("./net/adam.jl")
include("./net/buildNet.jl")
include("./net/prepareWeights.jl")
include("./net/prepareData.jl")

using BenchmarkTools

X_train = rand(Float32, 1000, 3200)
y_train = vec(rand(Float32, 1, 3200))

val = size(X_train, 1)
val2 = size(X_train, 2)

batch_size = 64
batchNum = Int(val2 / batch_size)

Wh, Wo = init_weights(val)

x = Variable(X_train)
y = Variable(y_train)

(graph, ŷ) = net(x, y, Wh, Wo)

print(graph)

optimizer = Adam(Wh.output)
optimizer2 = Adam(Wo.output)

display(@benchmark begin
    forward!(graph)
    backward!(graph)
    apply!(optimizer, Wh.output, Wh.gradient, 1)
    apply!(optimizer2, Wo.output, Wo.gradient, 1)
end)
