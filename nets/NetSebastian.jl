DefaultNet() = randn(4, 4), randn(3, 4), randn(4, 4), randn(3, 4)

function predict(x, wh, wo, y)
    x̂ = fullyconnected(wh, 4, 4, x, ReLU)
    ŷ = fullyconnected(wo, 3, 4, x̂, σ)
end

function foward(x, wh, wo, y)
    x̂ = fullyconnected(wh, 4, 4, x, ReLU)
    ŷ = fullyconnected(wo, 3, 4, x̂, σ)
    E = mean_squared_loss(y, ŷ)
end


function trainSebastian(netInitializer, X_train, y_train, epochs, lr)
    Wh, Wo, dWh, dWo = netInitializer()
    Loss_history = Float64[]
    Accuracy_history = Float64[]
    for _ = 1:epochs
        epoch_L = []
        epoch_acc = []
        for i in 1:size(X_train)[2]
            x = X_train[:, i]
            y = y_train[:, i]
            L = foward(x, Wh[:], Wo[:], y)
            dnet_Wh(x, wh, wo, y) = J(w -> foward(x, w, wo, y), wh)
            dWh[:] = dnet_Wh(x, Wh[:], Wo[:], y)
            dnet_Wo(x, wh, wo, y) = J(w -> foward(x, wh, w, y), wo)
            dWo[:] = dnet_Wo(x, Wh[:], Wo[:], y)
            push!(epoch_L, L)
            # Opt
            Wh -= lr * Wh
            Wo -= lr * dWo
        end
        push!(Loss_history, std(epoch_L))
        push!(Accuracy_history, std(epoch_acc))
    end
    return Loss_history, Accuracy_history
end