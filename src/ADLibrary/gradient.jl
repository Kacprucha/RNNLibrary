# Backward pass
function backward!(node::ReverseNode)
    for (child, local_grad) in node.children
        child.grad[1] += local_grad(node.grad[1])
        backward!(child)
    end
end

function back_grad(f, input_values::Vector)
    nodes = [ReverseNode(x) for x in input_values]

    output = f(nodes...)
    output.grad[1] = seed(output.value)

    backward!(output)

    return [n.grad[1] for n in nodes]
end

seed(x) = x isa Number ? one(x) : ones(size(x))
