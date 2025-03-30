function softmax(x::Matrix{Float64})
    m, n = size(x)  # m: number of classes, n: number of samples
    s = similar(x)
    for j in 1:n
        col = x[:, j]
        ex = exp.(col .- maximum(col))  # subtract max for numerical stability
        s[:, j] = ex ./ sum(ex)
    end
    return s
end

function dsoftmax(x::Matrix{Float64})
    s = softmax(x)
    m, n = size(x)  # m: number of classes, n: number of samples
    jacobians = [zeros(Float64, m, m) for _ in 1:n]
    for j in 1:n
        for i in 1:m
            for k in 1:m
                if i == k
                    jacobians[j][i, k] = s[i, j] * (1 - s[i, j])
                else
                    jacobians[j][i, k] = -s[i, j] * s[k, j]
                end
            end
        end
    end
    return jacobians
end

function tanh_activation(x::Matrix{Float64})
    return tanh.(x)
end

function d_tanh_activation(x::Matrix{Float64})
    return 1 .- x.^2         # Derivative: 1 - tanh(x)^2 for each element
end