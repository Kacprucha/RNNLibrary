include("ADLibrary/BackwordsDiffLibrary.jl")

using Random
using .BackwordsDiffLibrary

function parameters_initialisation(layer_conf::Vector{Dict{String, Any}})
    layers = []
    for i in 2:length(layer_conf)
        Random.seed!(0)
        k = 1/sqrt(layer_conf[i]["hidden"])
        input_weights = rand(layer_conf[i-1]["units"], layer_conf[i]["hidden"]) * 2 * k .- k

        hiden_weights = rand(layer_conf[i]["hidden"], layer_conf[i]["hidden"]) * 2 * k .- k
        hidden_bias = rand(1, layer_conf[i]["hidden"]) * 2 * k .- k

        output_weights = rand(layer_conf[i]["hidden"], layer_conf[i]["units"]) * 2 * k .- k
        output_bias = rand(1, layer_conf[i]["units"]) * 2 * k .- k

        push!(layers, [input_weights, hiden_weights, hidden_bias, output_weights, output_bias])
    end

    return layers
    
end

function forward(x, layers, activation_function)
    hiddens = []
    outputs = []

    for i in 1:length(layers)
        input_weights, hiden_weights, hidden_bias, output_weights, output_bias = layers[i]

        hidden = zeros(size(x,1), size(input_weights, 2))
        output = zeros(size(x,1), size(output_weights, 2))

        for j in 1:size(x, 1)
            input_x = x[j:j,:] * input_weights

            prev_idx = j == 1 ? 1 : j - 1
            hidden_x = input_x .+ hidden[prev_idx:prev_idx,:] * hiden_weights .+ hidden_bias
            hidden_x = activation_function(hidden_x)

            hidden[j,:] = hidden_x[1,:]

            output_x = hidden_x * output_weights + output_bias

            output[j,:] = output_x[1,:]
        end

        push!(hiddens, hidden)
        push!(outputs, output)
    end    

    return hiddens, outputs[end]
end

function backward(layers, x, lr, grad, hiddens, d_activation_function)
    for i in 1:length(layers)
        i_weight, h_weight, h_bias, o_weight, o_bias = layers[i]

        hidden = hiddens[i]
        next_h_grad = nothing 

        i_weight_grad = zeros(size(i_weight))
        h_weight_grad = zeros(size(h_weight))
        h_bias_grad   = zeros(size(h_bias))
        o_weight_grad = zeros(size(o_weight))
        o_bias_grad   = zeros(size(o_bias))

        num_samples = size(x, 1)
        for j in reverse(1:num_samples)
            out_grad = grad[j:j, :]

            hidden_col = reshape(hidden[j, :], (size(hidden, 2), 1))
            o_weight_grad .+= hidden_col * out_grad
            o_bias_grad .+= out_grad

            h_grad = out_grad * o_weight'

            if j < num_samples
                h_grad .+= next_h_grad * h_weight'
            end

            h_grad .= h_grad .* d_activation_function(hidden[j:j, :])

            next_h_grad = copy(h_grad)

            if j > 1
                prev_hidden = reshape(hidden[j-1, :], (size(hidden, 2), 1))
                h_weight_grad .+= prev_hidden * h_grad
                h_bias_grad .+= h_grad
            end

            x_row = reshape(x[j, :], (size(x, 2), 1))
            i_weight_grad .+= x_row * h_grad
        end

        lr_scaled = lr / num_samples
        i_weight .-= i_weight_grad * lr_scaled
        h_weight .-= h_weight_grad * lr_scaled
        h_bias   .-= h_bias_grad   * lr_scaled
        o_weight .-= o_weight_grad * lr_scaled
        o_bias   .-= o_bias_grad   * lr_scaled

        layers[i] = [i_weight, h_weight, h_bias, o_weight, o_bias]
    end
    
    return layers
end

function rnn_print(train_x, train_y, valid_x, valid_y, layer_conf, epochs, lr, sequence_length, activation_function, loss_function)
    if activation_function == "tanh"
        activation_fun = tanh_activation
        d_activation_fun = d_tanh_activation
    elseif activation_function == "softmax"
        activation_fun = softmax
        d_activation_fun = dsoftmax
    end
    
    layers = parameters_initialisation(layer_conf)

    for epoch in 1:epochs
        sequence_len = sequence_length
        epoch_loss = 0.0

        for j in 1:(size(train_x, 1) - sequence_len)
            seq_x = train_x[j:(j+sequence_len-1), :]
            seq_y = train_y[j:(j+sequence_len-1), :]

            hiddens, outputs = forward(seq_x, layers, activation_fun)

            #grad = BackwordsDiffLibrary.grad(loss_function, [seq_y, outputs])[2]
            grad = back_grad(loss_function, [seq_y, outputs])[2]

            layers = backward(layers, seq_x, lr, grad, hiddens, d_activation_fun)

            seq_y_r = lift(seq_y)
            outputs_r = lift(outputs)
            epoch_loss += loss_function(seq_y_r, outputs_r).value
        end

        # Every 10 epochs, compute validation loss.
        if epoch % 10 == 0
            valid_loss = 0.0
            for j in 1:(size(valid_x, 1) - sequence_len)
                seq_x = valid_x[j:(j+sequence_len-1), :]
                seq_y = valid_y[j:(j+sequence_len-1), :]

                _, outputs = forward(seq_x, layers, activation_fun)

                seq_y_r = lift(seq_y)
                outputs_r = lift(outputs)
                valid_loss += loss_function(seq_y_r, outputs_r).value
            end

            println("Epoch: $epoch train loss $(epoch_loss / size(train_x, 1)) valid loss $(valid_loss / size(valid_x, 1))")
        end
    end
end

function rnn(train_x, train_y, layer_conf, epochs, lr, sequence_length, activation_function, loss_function)
    if activation_function == "tanh"
        activation_fun = tanh_activation
        d_activation_fun = d_tanh_activation
    elseif activation_function == "softmax"
        activation_fun = softmax
        d_activation_fun = dsoftmax
    end
    
    layers = parameters_initialisation(layer_conf)

    for epoch in 1:epochs
        sequence_len = sequence_length
        epoch_loss = 0.0

        for j in 1:(size(train_x, 1) - sequence_len)
            seq_x = train_x[j:(j+sequence_len-1), :]
            seq_y = train_y[j:(j+sequence_len-1), :]

            hiddens, outputs = forward(seq_x, layers, activation_fun)

            grad = BackwordsDiffLibrary.grad(loss_function, [seq_y, outputs])[2]

            layers = backward(layers, seq_x, lr, grad, hiddens, d_activation_fun)

            seq_y_r = lift(seq_y)
            outputs_r = lift(outputs)
            epoch_loss += loss_function(seq_y_r, outputs_r).value
        end
    end

    return layers, epoch_loss / size(train_x, 1)
end