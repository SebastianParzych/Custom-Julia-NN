DefaultNet() = randn(4, 4), randn(3, 4), randn(4, 4), randn(3, 4)

function predict(x, wh, wo, y)
    x̂ = fullyconnected(wh, 4, 4, x, ReLU)
    ŷ = fullyconnected(wo, 3, 4, x̂, σ)
    return argmax(ŷ)
end

function foward(x, wh, wo, y)
    x̂ = fullyconnected(wh, 4, 4, x, ReLU)
    ŷ = fullyconnected(wo, 3, 4, x̂, σ)
    E = mean_squared_loss(y, ŷ)
end

mutable struct NetWeights
    Wh
    Wo
    dWh
    dWo
end
function getSebastianDefaultNet()
    Wh, Wo, dWh, dWo = DefaultNet()
    net = NetWeights(Wh, Wo, dWh, dWo)
    return net
end

function trainSebastian(net, X_train, y_train, epochs, lr)
    Loss_history = Float64[]
    Accuracy_history = Float64[]
    for _ = 1:epochs
        epoch_L = []
        epoch_acc = []
        for i in 1:size(X_train)[2]
            x = X_train[:, i]
            y = y_train[:, i]
            L = foward(x, net.Wh[:], net.Wo[:], y)
            dnet_Wh(x, wh, wo, y) = J(w -> foward(x, w, wo, y), wh)
            net.dWh[:] = dnet_Wh(x, net.Wh[:], net.Wo[:], y)
            dnet_Wo(x, wh, wo, y) = J(w -> foward(x, wh, w, y), wo)
            net.dWo[:] = dnet_Wo(x, net.Wh[:], net.Wo[:], y)
            push!(epoch_L, L)
            push!(epoch_acc, predict(x, net.Wh[:], net.Wo[:], y) == argmax(y) ? 1 : 0)
            # Opt
            net.Wh -= lr * net.dWh
            net.Wo -= lr * net.dWo
        end
        push!(Loss_history, std(epoch_L))
        push!(Accuracy_history, std(epoch_acc))
    end
    return Loss_history, Accuracy_history
end

function accuracySebastian(network, X, y)
    let
        accuracy_history = Float64[]
        for i in 1:size(X)[2]
            x = X_train[:, i]
            y = y_train[:, i]
            push!(accuracy_history, predict(network, x) == argmax(y) ? 1 : 0)
        end
        return accuracy_history, sum(accuracy_history) / length(accuracy_history) * 100
    end
end