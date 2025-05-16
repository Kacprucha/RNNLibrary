include("ADLibrary/dataTyp.jl")
include("ADLibrary/overloadMethods.jl")
include("ADLibrary/gradient.jl")
include("ADLibrary/functions.jl")

include("layersInfo.jl")
include("optymizersInfo.jl")
include("frowordPass.jl")
include("backwardPass.jl")

using Random
using Statistics
using LinearAlgebra

function forward(model::Sequential, x_initial) 
    current_x = x_initial
    layer_inputs = Any[x_initial] 
    layer_caches = []
    
    for layer_idx in 1:length(model.layers)
        layer = model.layers[layer_idx]
        output_l, cache_l = layer(current_x)
        
        push!(layer_caches, cache_l)
        current_x = output_l
        push!(layer_inputs, current_x) # Store output of layer i, which is input to layer i+1
    end

    return layer_inputs[end], layer_inputs[1:end-1], layer_caches
end

function backward(loss_grad_y_pred, model::Sequential, all_inputs_to_layers, all_layer_caches)
    model_param_grads = []
    dJ_dx_downstream = loss_grad_y_pred # Gradient w.r.t. output of the last layer

    for i in length(model.layers):-1:1
        layer = model.layers[i]
        x_input_to_current_layer = all_inputs_to_layers[i]
        cache_current_layer = all_layer_caches[i]
        
        param_grads_layer_tuple, dJ_dx_upstream = layer_backward(layer, dJ_dx_downstream, x_input_to_current_layer, cache_current_layer)

        if !isempty(param_grads_layer_tuple)
            # Convert tuple to vector and prepend
            for grad_p in reverse(param_grads_layer_tuple) # Ensure correct order if tuple was (W,b)
                prepend!(model_param_grads, [grad_p])
            end
        end
        
        dJ_dx_downstream = dJ_dx_upstream 
        
        if isnothing(dJ_dx_downstream) && i > 1
            num_remaining_params = 0
            for j in 1:i-1
                num_remaining_params += length(get_params(model.layers[j]))
            end
            prepend!(model_param_grads, [nothing for _ in 1:num_remaining_params]) # Placeholder
            break 
        end
    end

    return model_param_grads
end

function train!(model::Sequential, loss_fun, X_train, y_train, X_test, y_test;
    epochs=5, lr=0.001, batchsize=64, optimizer=:SGD, clip_norm=1.0f0, decay_factor=0.5f0, decay_epochs=4, print_learning_data=true)

    ps = get_params(model)
    opt = nothing
    if optimizer == :SGD
        opt = SGD(lr)
    elseif optimizer == :Adam
        opt = Adam(convert(Float32, lr), ps)
    else
        error("Unsupported optimizer type. Choose :SGD or :Adam.")
    end

    n_samples = size(X_train, 2)  

    for epoch in 1:epochs

        if epoch > 1 && (epoch -1) % decay_epochs == 0
            opt.lr[1] *= decay_factor
        end

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        t = @elapsed begin
            for i in 1:batchsize:n_samples
                last = min(i + batchsize - 1, n_samples)
                x_batch = X_train[:, i:last]
                y_batch = y_train[:, i:last]

                # FORWARD PASS
                y_pred, all_inputs_to_layers, all_layer_caches = forward(model, x_batch)
            
                batch_loss = loss_fun(y_pred, y_batch)
                batch_acc = batch_accuracy(y_pred, y_batch)

                # Compute Loss Gradient 
                @diffunction loss_wrapper(a, y) = loss_fun(a, y)
                delta = grad(loss_wrapper, [y_pred, y_batch])[1]
            
                # BACKWARD PASS
                grads = backward(delta, model, all_inputs_to_layers, all_layer_caches)

                # Calculate total norm of gradients
                valid_grads = filter(g -> !isnothing(g), grads)
                if !isempty(valid_grads)
                    total_norm_sq = sum(sum(g.^2f0) for g in valid_grads)
                    total_norm = sqrt(total_norm_sq)

                    if total_norm > clip_norm
                        for i in eachindex(grads)
                            if !isnothing(grads[i])
                                grads[i] .*= (clip_norm / total_norm)
                            end
                        end
                    end
                end

                update!(opt, ps, grads)

                # accumulate
                total_loss += batch_loss
                total_acc  += batch_acc
                num_batches += 1
            end

            # average over all batches
            avg_train_loss = total_loss / num_batches
            avg_train_acc  = total_acc  / num_batches

            y_test_pred, _, _ = forward(model, Matrix(X_test)) 
            test_loss = loss_fun(y_test_pred, y_test)
            test_acc  = batch_accuracy(y_test_pred, y_test)
        end

        if print_learning_data
            println(
                "Epoch $epoch ▶ ",
                "Train Loss=$(round(avg_train_loss, digits=4)), ",
                "Train Acc=$(round(100*avg_train_acc, digits=2))%\t│   ",
                "Test Loss=$(round(test_loss, digits=4)), ",
                "Test Acc=$(round(100*test_acc, digits=2))%\t|   ",
                "Time=$(round(t, digits=2))"
            )
        end
    end
end

# --- Helpers ---
function batch_accuracy(y_pred::Array, y_true::Array)
    if size(y_pred,1)==1
      # binary: predict “1” if p≥0.5, else “0”
      preds = vec(y_pred .>= 0.5f0)
      trues = vec(y_true .== 1f0)
    else
      # multi‑class: usual argmax
      preds = vec(argmax(y_pred, dims=1))
      trues = vec(argmax(y_true, dims=1))
    end
    return mean(preds .== trues)
end