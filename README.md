# RNNLibrary

## Overview

RNNLibrary is a Julia library that provides simple sceletion for making your own neural network. 

## Features

- **Build in layers**: Library provieds layers like RNN, Dence, Embeding. 
- **Performance**: Optimized for speed and low memory usage.
- **Flexibility**: Suitable for crating network from your dreams..

## Installation

Ensure you have Julia installed (version 1.11.3). Install Julia's package manager. In the Julia REPL, run:

```julia
using Pkg
Pkg.add("RNNLibrary")
```

## Usage

Import the library in your Julia script:

```julia
using RNNLibrary
```

Define your model:

```julia
model = Sequential([
  Embedding(vocab_size, embedding_dim),
  SimpleRNN(embedding_dim, hidden_size, ReLU), 
  SelectLastTimestep(),
  Flatten(), 
  Dense(hidden_size, 1, Sigmoid)
])
```

and the loss function:

```julia
loss_fun(y_pred, y_true) = -mean(y_true .* log.(y_pred .+ 1e-7) .+ (1 .- y_true) .* log.(1 .- y_pred .+ 1e-7))
```

Import the data you want to train your network on and use train function to make the magic happen:

```julia
train!(model, loss_fun, X_train, y_train, X_test, y_test; epochs=12, lr=0.001, batchsize=128, optimizer=:Adam)
```

## Examples

In folder examples there is an notbook ilustrating full proces of crating your own network and training it on sumple data.

## License

This project is licensed under the MIT License. See the LICENSE file for details.