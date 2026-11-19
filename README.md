# NeuroEvolution.jl

NeuroEvolution.jl is a tutorial project for learning how genetic algorithms,
neural networks, and neuroevolution can be applied in Julia. It provides small,
easy-to-follow implementations and examples that introduce the core ideas
without requiring a large framework or extensive background in machine learning.

The project is based on the concepts presented in *Practical Artificial
Intelligence in Julia: Implementing Neural Networks, Genetic Algorithms, and
Neuroevolution in Julia* by Alexandre Bergel.

## What is included

- Individual neurons with step or sigmoid activation functions
- Perceptron and neuron training
- Feed-forward neural network layers
- Configurable one-layer and two-layer neural networks
- Digital logic examples, including AND, OR, NOR, NOT, and a digital comparator
- Small runnable examples for training and visualizing perceptrons
- Tests demonstrating neuron behavior and network construction

## Requirements

- Julia 1.9 or newer

## Getting started

From the Julia REPL, activate the project from its root directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using NeuroEvolution
```

A simple neuron can then be created and evaluated:

```julia
neuron = Neuron([1.0, 1.0], -1.5)
NeuroEvolution.feed(neuron, [1.0, 1.0]) # 1
NeuroEvolution.feed(neuron, [1.0, 0.0]) # 0
```

For sigmoid neurons and training:

```julia
neuron = NeuroEvolution.NNets.sigmoid_neuron!([-1.0, -1.0], 2.0)

training_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
desired_outputs = [0.0, 1.0, 1.0, 1.0]

NeuroEvolution.NNets.train_neuron!(
    neuron,
    training_inputs,
    desired_outputs,
    10000,
    0.1,
)
```

## Running the examples

The perceptron example is in `examples/perceptron_examples.jl`. Run it from the
project directory with:

```sh
julia --project=. examples/perceptron_examples.jl
```

The example trains sigmoid perceptrons and plots the training error. Plotting
support requires the `Plots` package.

## Running the tests

Run the test suite from the Julia REPL:

```julia
using Pkg
Pkg.test()
```

Or from the project directory:

```sh
julia --project=. test/runtests.jl
```

## Project structure

```text
src/
  NeuroEvolution.jl   Package entry point and public exports
  NNets/
    NNets.jl          Neural-network module
    neuron.jl         Neuron behavior and training
    neural_net.jl     Layers and feed-forward networks
examples/             Runnable tutorial examples
test/                 Package tests
```

## Learning goals

This repository is intended as a practical starting point. It can be used to
explore how weighted inputs and biases affect neuron output, how error-driven
training updates parameters, how neurons combine into layers, and how the
principles of genetic algorithms can guide future neuroevolution experiments.
