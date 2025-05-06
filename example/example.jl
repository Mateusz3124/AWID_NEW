using JLD2, Printf
@load "../data/imdb_dataset_prepared.jld2" X_train y_train X_test y_test

using Flux, Printf, Statistics

dataset = Flux.DataLoader((X_train, y_train), batchsize=64, shuffle=true)

model = Chain(
    Dense(size(X_train, 1), 32, relu),
    Dense(32, 1, sigmoid)
)

loss(m, x, y) = Flux.Losses.binarycrossentropy(m(x), y)
accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))

opt = Flux.setup(Adam(), model)
epochs = 5
for epoch in 1:epochs
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    t = @elapsed begin
        for (x, y) in dataset
            grads = Flux.gradient(model) do m
                l = loss(m, x, y)
                total_loss += l
                total_acc += accuracy(m, x, y)
                return l
            end
            Optimisers.update!(opt, model, grads[1])
            num_samples += 1
        end

        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples

        test_acc = accuracy(model, X_test, y_test)
        test_loss = loss(model, X_test, y_test)
    end

    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)",
        epoch, t, train_loss, train_acc, test_loss, test_acc))
end