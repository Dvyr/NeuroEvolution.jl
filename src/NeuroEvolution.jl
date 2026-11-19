module NeuroEvolution

include("NNets/NNets.jl")

using .NNets

export Neuron, feed, digital_comparator, train!, train_neuron!, sigmoid_neuron!, step_neuron!, isclose

end