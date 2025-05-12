module RNNLibrary

include("layersInfo.jl")
include("optymizersInfo.jl")
include("frowordPass.jl")
include("backwardPass.jl")
include("networkFunctions.jl")

export Dense, Embedding, SimpleRNN, SelectLastTimestep, Flatten, Sequential
export train!

end
