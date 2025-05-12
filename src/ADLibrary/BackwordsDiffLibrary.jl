module BackwordsDiffLibrary

include("dataTyp.jl")
include("overloadMethods.jl")
include("gradient.jl")
include("functions.jl")

export ReverseNode, lift, @diffunction
export grad
export ReLU, Sigmoid

end
