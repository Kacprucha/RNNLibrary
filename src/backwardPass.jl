include("ADLibrary/dataTyp.jl")

# For Dense Layer
function layer_backward(layer::Dense, dJ_da::AbstractMatrix{Float32}, x_input::AbstractMatrix{Float32}, cache)
    _, z = cache 

    @diffunction d_activation_func(val) = layer.activation.(val)
    dJ_dz = dJ_da .* grad(d_activation_func, [z])[1] 

    batch_size = size(x_input, 2)
    
    dJ_dW = (dJ_dz * x_input') 
    dJ_dW ./= batch_size # In-place division

    dJ_db = sum(dJ_dz, dims=2) 
    dJ_db ./= batch_size # In-place division
    dJ_dx_input = layer.W' * dJ_dz 

    return (dJ_dW, dJ_db), dJ_dx_input
end

# For Embedding Layer
function layer_backward(layer::Embedding, dJ_doutput::AbstractArray{Float32, 3}, _, cache)
    x_indices, = cache
    embedding_dim, seq_len, batch_size = size(dJ_doutput)
    vocab_size = size(layer.E, 2)

    dJ_dE = zeros(Float32, embedding_dim, vocab_size)

    for b_idx in 1:batch_size
        for s_idx in 1:seq_len
            word_idx = x_indices[s_idx, b_idx]
            if 1 <= word_idx <= vocab_size
                dJ_dE[:, word_idx] .+= dJ_doutput[:, s_idx, b_idx]
            end
        end
    end

    dJ_dE ./= batch_size
    return (dJ_dE,), nothing
end

# For SimpleRNN Layer
function layer_backward(layer::SimpleRNN, dJ_dhs::AbstractArray{Float32, 3}, _, cache)
    x_seq, h_prev_cache, zs, _ = cache 
    
    hidden_size, seq_len, batch_size = size(dJ_dhs)
    input_dim = size(x_seq, 1)

    dJ_dWx = zeros(Float32, size(layer.Wx))
    dJ_dWh = zeros(Float32, size(layer.Wh))
    dJ_db = zeros(Float32, size(layer.b))
    dJ_dx_seq = zeros(Float32, size(x_seq)) 

    dh_total_buffer = zeros(Float32, hidden_size, batch_size)
    dh_next_t = zeros(Float32, hidden_size, batch_size) 

    @diffunction d_rnn_activation_func(val) = layer.activation.(val)

    for t in seq_len:-1:1
        dht_total_buffer = similar(dh_next_t)
        dJ_dhs_slice = @view dJ_dhs[:, t, :]
        dht_total_buffer .= dJ_dhs_slice .+ dh_next_t
        
        # Gradient through activation
        zt = zs[:, t, :]
        dJ_dzt = dht_total_buffer .* grad(d_rnn_activation_func, [zt])[1]

        # Gradients for parameters
        xt = @view x_seq[:, t, :]
        h_prev_t = @view h_prev_cache[:, t, :]
        
        dJ_dWx .+= dJ_dzt * xt'
        dJ_dWh .+= dJ_dzt * h_prev_t'
        dJ_db .+= sum(dJ_dzt, dims=2)

        # Gradient to pass to previous layer (input x_t)
        dJ_dx_seq[:, t, :] = layer.Wx' * dJ_dzt
        
        # Gradient for h_{t-1} (to be dh_next_t in the next iteration)
        dh_next_t = layer.Wh' * dJ_dzt
    end
    
    # Average gradients by batch_size
    dJ_dWx ./= batch_size
    dJ_dWh ./= batch_size
    dJ_db ./= batch_size

    # Grads for Wx, Wh, b
    return (dJ_dWx, dJ_dWh, dJ_db), dJ_dx_seq
end

# For Flatten Layer
function layer_backward(layer::Flatten, dJ_doutput::AbstractMatrix{Float32}, _, cache)
    original_size, = cache
    dJ_dx_input = reshape(dJ_doutput, original_size)
    return (), dJ_dx_input # No parameters
end

# For SelectLastTimestep Layer
function layer_backward(layer::SelectLastTimestep, dJ_doutput::AbstractMatrix{Float32}, _, cache)
    original_x_seq_size, = cache 
    
    dJ_dx_input = zeros(Float32, original_x_seq_size)
    dJ_dx_input[:, end, :] = dJ_doutput
    return (), dJ_dx_input # No parameters
end