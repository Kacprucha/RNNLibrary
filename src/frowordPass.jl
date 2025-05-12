function (layer::Dense)(x::AbstractMatrix{Float32}) # Assuming x is features x batch_size
    z = layer.W * x .+ layer.b
    a = layer.activation.(z)
    cache = (x, z) 
    return a, cache
end

function (layer::Embedding)(x_indices::AbstractMatrix{Int}) # x_indices: seq_len x batch_size
    seq_len, batch_size = size(x_indices)
    embedding_dim = size(layer.E, 1)
    
    # Initialize output array
    output = zeros(Float32, embedding_dim, seq_len, batch_size)
    
    for b_idx in 1:batch_size
        for s_idx in 1:seq_len
            word_idx = x_indices[s_idx, b_idx]
            if 1 <= word_idx <= size(layer.E, 2)
                output[:, s_idx, b_idx] = layer.E[:, word_idx]
            end
        end
    end
    cache = (x_indices,) # Cache original indices for backward pass
    return output, cache
end

function (layer::SimpleRNN)(x_seq::AbstractArray{Float32, 3}, h_init::Union{Nothing, AbstractMatrix{Float32}}=nothing)
    input_dim, seq_len, batch_size = size(x_seq)
    hidden_size = size(layer.Wh, 1)

    hs = zeros(Float32, hidden_size, seq_len, batch_size)
    zs = zeros(Float32, hidden_size, seq_len, batch_size)

    h_prev = if isnothing(h_init)
        zeros(Float32, hidden_size, batch_size)
    else
        h_init
    end

    for t in 1:seq_len
        xt_view = @view x_seq[:, t, :] 

        z_t = layer.Wx * xt_view .+ layer.Wh * h_prev .+ layer.b 

        h_curr = layer.activation.(z_t)

        @views hs[:, t, :] .= h_curr 
        @views zs[:, t, :] .= z_t  
        h_prev = h_curr 
    end

    h_prev_cache = similar(hs)
    if seq_len > 0
      @views h_prev_cache[:, 1, :] .= isnothing(h_init) ? zeros(Float32, hidden_size, batch_size) : h_init
      if seq_len > 1
          @views h_prev_cache[:, 2:seq_len, :] .= hs[:, 1:seq_len-1, :]
      end
    end

    cache = (x_seq, h_prev_cache, zs, hs)
    return hs, cache
end

function (layer::Flatten)(x::AbstractArray{Float32})
    original_size = size(x)
    if length(original_size) < 2
        error("Flatten input must have at least 2 dimensions.")
    end
    
    dim1 = prod(original_size[1:end-1])
    batch_size = original_size[end]
    output = reshape(x, dim1, batch_size)
    cache = (original_size,)
    return output, cache
end

function (layer::SelectLastTimestep)(x_seq::AbstractArray{Float32, 3})
    output = x_seq[:, end, :]
    cache = (size(x_seq),) 
    return output, cache
end