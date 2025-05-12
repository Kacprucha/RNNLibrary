abstract type Layer end

# Dense Layer
struct Dense <: Layer
    W::Array{Float32,2}
    b::Array{Float32,1}
    activation::Function     
end

function Dense(in_features::Int, out_features::Int, activation::Function)
    weights = glorot_uniform(out_features, in_features)
    baias = zeros(out_features)
    return Dense(weights, baias, activation)
end

function (layer::Dense)(x::AbstractArray)
    z = layer.W * x .+ layer.b  
    return layer.activation.(z)
end

# Embedding Layer
struct Embedding <: Layer
    E::Array{Float32, 2}  
end

function Embedding(vocab_size::Int, embedding_dim::Int)
    E = randn(Float32, embedding_dim, vocab_size) .* 0.01f0 
    return Embedding(E)
end

# SimpleRNN Layer
struct SimpleRNN <: Layer
    Wx::Array{Float32, 2}  
    Wh::Array{Float32, 2}  
    b::Array{Float32, 1}   
    activation::Function
end

function SimpleRNN(input_size::Int, hidden_size::Int, activation::Function)
    Wx = glorot_uniform(hidden_size, input_size)
    Wh = glorot_uniform(hidden_size, hidden_size)
    b = zeros(Float32, hidden_size)
    return SimpleRNN(Wx, Wh, b, activation)
end

# Function layer
struct Flatten <: Layer
    # No parameters
end

# SelectLastTimestep
struct SelectLastTimestep <: Layer
    # No parameters
end

# Model structure 
struct Sequential
    layers::Vector{Layer}
end

# --- Helpers ---
function glorot_uniform(fan_out::Int, fan_in::Int)
    limit = sqrt(6f0 / (fan_in + fan_out))
    return rand(Float32, fan_out, fan_in) .* (2f0 * limit) .- limit
end

function get_params(layer::Dense)
    return [layer.W, layer.b]
end

function get_params(layer::Embedding)
    return [layer.E]
end

function get_params(layer::SimpleRNN)
    return [layer.Wx, layer.Wh, layer.b]
end

function get_params(layer::Flatten)
    return []
end

function get_params(layer::SelectLastTimestep)
    return []
end

function get_params(model::Sequential) # Renamed from MLP
    ps = []
    for layer in model.layers
        append!(ps, get_params(layer))
    end
    return ps
end