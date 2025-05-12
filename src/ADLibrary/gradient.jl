# Backward pass
function backward!(node::ReverseNode)
    for (child, local_grad) in node.children
        grad_update = local_grad(node.grad[1])
        if isa(child.grad[1], AbstractArray)
            if size(child.grad[1]) != size(grad_update)
                grad_update = reshape(grad_update, size(child.grad[1]))
            end
            child.grad[1] .+= grad_update
        else
            child.grad[1] += grad_update
        end
        
        # Recursively propagate the gradients.
        backward!(child)
    end
end

function grad(f, input_values::Vector)
    nodes = [ReverseNode(x) for x in input_values]

    output = f(nodes...)
    output.grad[1] = seed(output.value)

    backward!(output)

    return [n.grad[1] for n in nodes]
end

seed(x) = x isa Number ? one(x) : ones(size(x))
