function plotLossAccuracu(Loss_history, acc_test)
    p1 = plot(Loss_history, xlabel="Iteration", ylabel="Loss function", label="")
    p2 = plot(acc_test, xlabel="Iteration", ylabel="Test accuracy", label="", ylim=(-0.01, 1.01))
    plot(p1, p2, layout=(2, 1))
end