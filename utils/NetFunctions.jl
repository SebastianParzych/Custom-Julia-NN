# Lista funkcji aktywacji wykorzystywanych w projekcie

# Funkcje aktywacji
ReLU(x) = max(zero(x), x)
identityFunction(x) = x
σ(x) = one(x) / (one(x) + exp(-x))
tanh(x) = 2.0 / (one(x) + exp(-2.0x)) - one(x)

# Funkcje kosztu
mean_squared_loss(y::Vector, ŷ::Vector) = sum(0.5(y - ŷ) .^ 2)

function binary_cross_entropy(y::Vector, ŷ::Vector)
    epsilon = eps(1.0)
    ## Avoding 0 , 1 in log argument
    ŷ = [max(i, epsilon) for i in ŷ]
    ŷ = [min(i, 1 - epsilon) for i in ŷ]
    return -sum(y .* log.(ŷ) + (1 .- y) .* log.(1 .- ŷ)) / length(y)
end
nothing