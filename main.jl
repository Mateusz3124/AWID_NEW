using Pkg
Pkg.develop(path="./net")
using net: startNetwork

using JLD2, Printf
@load "./data/imdb_dataset_prepared.jld2" X_train y_train X_test y_test

X_train = Matrix{Float32}(X_train)
y_train = Vector{Float32}(vec(y_train))
X_test = Matrix{Float32}(X_test)
y_test = Vector{Float32}(vec(y_test))

startNetwork(X_train, y_train, X_test, y_test, 64)