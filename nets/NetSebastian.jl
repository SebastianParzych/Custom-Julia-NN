DefaultNet(nHidden) = randn(4, nHidden), randn(3, nHidden), randn(4, nHidden), randn(3, nHidden)
fullyconnected(w::Vector, n::Number, m::Number, v::Vector, activation::Function) = activation.(reshape(w, n, m) * v)

function predict(net, x)
    x̂ = fullyconnected(net.Wh[:], net.nHidden, 4, x, ReLU)
    ŷ = fullyconnected(net.Wo[:], 3, net.nHidden, x̂, σ)
    return argmax(ŷ)
end

function foward(x, wh, wo, nhidden, y)
    x̂ = fullyconnected(wh, nhidden, 4, x, ReLU)
    ŷ = fullyconnected(wo, 3, nhidden, x̂, σ)
    E = mean_squared_loss(y, ŷ)
end

mutable struct NetWeights
    Wh
    Wo
    dWh
    dWo
    nHidden
end
function getSebastianDefaultNet(nHidden)
    Wh, Wo, dWh, dWo = DefaultNet(nHidden)
    net = NetWeights(Wh, Wo, dWh, dWo, nHidden)
    return net
end
Sebastianonecold(y, classes) = [classes[argmax(y_col)] for y_col in eachcol(y)]
SebastianAccuracy(m, x, y) = mean(onecold(m(x), classes) .== onecold(y, classes))
dnet_Wh(x, wh, wo, nh, y) = J(w -> foward(x, w, wo, nh, y), wh)
dnet_Wo(x, wh, wo, nh, y) = J(w -> foward(x, wh, w, nh, y), wo)

function trainSebastian(net, X_train, y_train, epochs, lr)
    Loss_history = Float64[]
    Accuracy_history = Float64[]
    for _ = 1:epochs
        epoch_L = []
        epoch_acc = []
        for i in 1:size(X_train)[2]
            x = X_train[:, i]
            y = y_train[:, i]
            L = foward(x, net.Wh[:], net.Wo[:], net.nHidden, y)
            net.dWh[:] = dnet_Wh(x, net.Wh[:], net.Wo[:], net.nHidden, y)
            net.dWo[:] = dnet_Wo(x, net.Wh[:], net.Wo[:], net.nHidden, y)
            push!(epoch_L, L)
            push!(epoch_acc, predict(net, x) == argmax(y) ? 1 : 0)
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
        return accuracy_history, sum(accuracy_history) / length(accuracy_history)
    end
end



function getSebastianResults(nHidden, X_train, y_train, X_test, y_test, epochs, lr)
    sebastianNet = getSebastianDefaultNet(nHidden)
    sebastianTrainLoss, sebastianTrainAccHistory = trainSebastian(sebastianNet, X_train, y_train, epochs, lr)
    sebastianTrainAccVal = getTrainingAccValFromHistory(sebastianTrainAccHistory)
    sebastianTestAccuracy, sebastianTestAccVal = accuracySebastian(sebastianNet, X_test, y_test)
    return sebastianTrainLoss, sebastianTrainAccHistory, sebastianTrainAccVal, sebastianTestAccuracy, sebastianTestAccVal
end
