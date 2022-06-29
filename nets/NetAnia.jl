include("./../utils/NetFunctions.jl")
diagonal(m) = diagm(0 => vec(m))
J = function jacobian(f, args::Vector{T}) where {T<:Number}
    jacobian_columns = Matrix{T}[]

    for i = 1:length(args)
        x = Dual{T}[]
        for j = 1:length(args)
            seed = (i == j)
            push!(x, seed ?
                     Dual(args[j], one(args[j])) :
                     Dual(args[j], zero(args[j])))
        end
        column = partials.([f(x)...])
        push!(jacobian_columns, column[:, :])
    end
    hcat(jacobian_columns...)
end

mutable struct Layer
    m::Int
    n::Int
    activation::Function
    W::Matrix
    dW::Matrix
    b::Vector
    db::Vector
end

function NeuralNetwork()
    layer = []
    function AddLayer(m, n, activation)
        layer_::Layer = Layer(m, n, activation, randn(n, m), randn(n, m), randn(n), randn(n))
        push!(layer, layer_)
    end
    () -> (AddLayer, layer)
end


forward(net, x, y) =
    let
        for i = 1:(size(net.layer)[1])
            x = net.layer[i].activation.(reshape(net.layer[i].W, net.layer[i].n, net.layer[i].m) * x .+ net.layer[i].b)
        end
        E = mean_squared_loss(y, x)
        return E
    end


forward_w(net, x, y, w, j) =
    let
        tmp = randn(net.layer[j].n, net.layer[j].m)
        tmp[:, :] = net.layer[j].W
        net.layer[j].W = w[:, :]

        for i = 1:(size(net.layer)[1])
            x = net.layer[i].activation.(reshape(net.layer[i].W, net.layer[i].n, net.layer[i].m) * x .+ net.layer[i].b)
        end

        net.layer[j].W = tmp[:, :]
        E = mean_squared_loss(y, x)
        return E
    end

forward_b(net, x, y, b, j) =
    let
        tmp = randn(net.layer[j].n)
        tmp[:] = net.layer[j].b
        net.layer[j].b = b[:]

        for i = 1:(size(net.layer)[1])
            x = net.layer[i].activation.(reshape(net.layer[i].W, net.layer[i].n, net.layer[i].m) * x .+ net.layer[i].b)
        end

        net.layer[j].b = tmp[:]
        E = mean_squared_loss(y, x)
        return E
    end


backpropagation(net, x, y) =
    let
        for i = 1:(size(net.layer)[1])
            net.layer[i].dW[:] = J(w -> forward_w(net, x, y, w, i), net.layer[i].dW[:])
            net.layer[i].db[:] = J(b -> forward_b(net, x, y, b, i), net.layer[i].db[:])
        end
    end


update(net, x, y, α::Float64) =
    let
        for i = 1:(size(net.layer)[1])
            net.layer[i].W -= α * net.layer[i].dW
            net.layer[i].b -= α * net.layer[i].db
        end
    end


trainAnia(net, X_train, y_train, epochs, α) =
    let
        Loss_history = Float64[]
        Acc_history = Float64[]
        for j = 1:epochs
            epoch_L = Float64[]
            epoch_acc = Float64[]
            for i in 1:size(X_train)[2]
                x = X_train[:, i]
                y = y_train[:, i]
                Ei = forward(net, x, y)
                push!(epoch_L, Ei)
                push!(epoch_acc, predict(net, x) == argmax(y) ? 1 : 0)
                backpropagation(net, x, y)
                update(net, x, y, α)
            end
            push!(Loss_history, std(epoch_L))
            push!(Acc_history, sum(epoch_acc) / length(epoch_acc) * 100)
        end

        return Loss_history, Acc_history / 100
    end


predict(net, x) =
    let
        for i = 1:(size(net.layer)[1])
            x = net.layer[i].activation.(reshape(net.layer[i].W, net.layer[i].n, net.layer[i].m) * x .+ net.layer[i].b)
        end

        return argmax(x)
    end

accuracy(network, X, y) =
    let
        accuracy_history = Float64[]
        for i in 1:size(X)[2]
            x = X_train[:, i]
            y = y_train[:, i]
            push!(accuracy_history, predict(network, x) == argmax(y) ? 1 : 0)
        end
        return accuracy_history, sum(accuracy_history) / length(accuracy_history)
    end
accuracyFromAccHistory(history) =
    let
        return sum(history) / length(history)
    end
function getDefaultAniaNet()
    net = NeuralNetwork()
    net.AddLayer(4, 4, ReLU)
    net.AddLayer(4, 3, σ)
    return net
end