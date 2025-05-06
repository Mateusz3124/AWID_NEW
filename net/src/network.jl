using differential: forward!, backward!, Variable

include("./adam.jl")
include("./buildNet.jl")
include("./prepareWeights.jl")
include("./prepareData.jl")

using Random

import Statistics: mean
accuracy(ŷ, y) = mean((ŷ .> 0.5) .== (y .> 0.5))

import LinearAlgebra
function startNetwork(X_train::Matrix{Float32}, y_train::Vector{Float32}, X_test::Matrix{Float32}, y_test::Vector{Float32}, batch_size::Int64)

    val = size(X_train, 1)
    val2 = size(X_train, 2)

    if (val2 % batch_size != 0)
        error("Data of size $val2 could not be devided to batches of size $batch_size")
    end

    batchNum = Int(val2 / batch_size)

    Wh, Wo = init_weights(val)
    data = initData(X_train, y_train, batchNum, batch_size)

    x = Variable(data[1][1])
    y = Variable(data[1][2])

    xTest = Variable(X_test)
    yTest = Variable(y_test)

    # (graph, ŷ) = graphBuild(x, y, Wh, Wo)
    (graphTest, ŷTest) = graphBuild(xTest, yTest, Wh, Wo)

    optimizer = Adam(Wh.output)
    optimizer2 = Adam(Wo.output)

    epochs = 5

    graphs = []

    for (x,y) in (data)
        (graph, ~) = graphBuild(Constant(x), Constant(y), Wh, Wo)
        push!(graphs, graph)
    end

    for epoch in 1:epochs
        total_loss = Float32(0.0)
        total_acc = Float32(0.0)

        t = @elapsed begin
            for graph in graphs

                forward!(graph)
                backward!(graph)

                apply!(optimizer, Wh.output, Wh.gradient, epoch)
                apply!(optimizer2, Wo.output, Wo.gradient, epoch)
            end
            shuffle!(data)

            forward!(graphTest)

            test_loss = graphTest[end].output
            test_acc = accuracy(ŷTest.output, y_test)
        end

        println("Epoch: $epoch, time: $t, Loss: $(total_loss/batchNum), Accuracy: $(total_acc/batchNum), LossTest: $(test_loss), AccuracyTest: $(test_acc)")
    end
end