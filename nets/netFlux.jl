n_hidden = 2
# DEFAULT MODEL SETUP
DEFAULT_MODEL = Chain(
    Dense(size(X_train, 1), n_hidden, relu),
    Dense(n_hidden, size(y_train, 1), identity),
    softmax,
)
DEFAULT_OPT = Descent(0.1)  # Optimalizer
DEFAULT_PARAMS = Flux.params(DEFAULT_MODEL)
DEFAULT_LOSS_FUN(m, x, y) = crossentropy(m(x), y) # Mean square error
# used function
onecold(y, classes) = [classes[argmax(y_col)] for y_col in eachcol(y)]
accuracy(m, x, y) = mean(onecold(m(x), classes) .== onecold(y, classes))


function getFluxAccuracyVal(fluxAccuracyHistory)
    return sum(fluxAccuracyHistory) / length(fluxAccuracyHistory)
end

function getDefaultFlux()
    return DEFAULT_MODEL, DEFAULT_OPT, DEFAULT_LOSS_FUN, DEFAULT_PARAMS
end

function train(net, ps, X_train, y_train, epochs, opt, LossFun)
    acc_test = zeros(epochs)
    Loss_history = similar(acc_test)
    for i in 1:epochs
        Loss = LossFun(net, X_train, y_train)
        Loss_history[i] = Loss
        gs = gradient(() -> LossFun(net, X_train, y_train), ps)
        Flux.Optimise.update!(opt, ps, gs)
        acc_test[i] = accuracy(net, X_test, y_test)
    end
    return Loss_history, acc_test
end