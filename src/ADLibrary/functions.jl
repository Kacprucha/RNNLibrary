function ReLU(x) 
    return max(0, x)
end

function Sigmoid(x)
    return 1 / (1 + exp(-x))
end