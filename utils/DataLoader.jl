
Random.seed!(2)

function custom_split(X, y::AbstractVector; dims=1, ratio_train=0.8, kwargs...)
    n = length(y)
    size(X, dims) == n || throw(DimensionMismatch("..."))

    n_train = round(Int, ratio_train * n)
    i_rand = randperm(n)
    i_train = i_rand[1:n_train]
    i_test = i_rand[n_train+1:end]

    return selectdim(X, dims, i_train), y[i_train], selectdim(X, dims, i_test), y[i_test]
end

function norm(X_train, X_test; dims=1, kwargs...)
    col_mean = mean(X_train; dims)
    col_std = std(X_train; dims)

    return (X_train .- col_mean) ./ col_std, (X_test .- col_mean) ./ col_std
end

function onehot(y, classes)
    y_onehot = falses(length(classes), length(y))
    for (i, class) in enumerate(classes)
        y_onehot[i, y.==class] .= 1
    end
    return y_onehot
end

function prepare_data(X, y; do_normal=true, do_onehot=true, kwargs...)
    X_train, y_train, X_test, y_test = custom_split(X, y; kwargs...)

    if do_normal
        X_train, X_test = norm(X_train, X_test; kwargs...)
    end

    classes = unique(y)
    if do_onehot
        y_train = onehot(y_train, classes)
        y_test = onehot(y_test, classes)
    end

    return X_train, y_train, X_test, y_test, classes
end

function getRawDataset()
    return dataset("datasets", "iris")
end

function getPrepearedData()
    iris = getRawDataset()
    X = Matrix(iris[:, 1:4])
    y = iris.Species
    return prepare_data(X', y; dims=2)
end