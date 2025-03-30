struct ReverseNode{T}
    value::T
    grad::Vector{T}
    children::Vector{Tuple{ReverseNode, Function}}
end

ReverseNode(x) = ReverseNode(float(x), [zero(float(x))], Tuple{ReverseNode, Function}[])

lift(x) = ReverseNode(x)