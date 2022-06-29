function plotLossAccuracy(Loss_history, acc_test)
    p1 = plot(Loss_history, xlabel="Iteration", ylabel="Loss function", label="")
    p2 = plot(acc_test, xlabel="Iteration", ylabel="Train accuracy", label="", ylim=(-0.01, 1.01))
    plot(p1, p2, layout=(2, 1))
end

function plotTestAccuracty(acc_test)
    plot(acc_test, xlabel="Samples", ylabel="Test accuracy", label="", ylim=(-0.01, 1.01))
end



function tripleMetricPlot(Loss_history, acc_train, acc_test)
    p1 = plot(Loss_history, xlabel="Iteration", ylabel="Loss function", label="")
    p2 = plot(acc_train, xlabel="Iteration", ylabel="Training accuracy", label="")
    p3 = plot(acc_test, xlabel="Iteration", ylabel="Test accuracy", label="", ylim=(-0.01, 1.01))
    display(plot(p1, p2, p3, layout=(3, 1)))
end