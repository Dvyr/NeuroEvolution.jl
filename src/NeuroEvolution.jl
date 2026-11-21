module NeuroEvolution

include("NNets/NNets.jl")
include("GeneticAlgorithm/GeneticAlgorithm.jl")

using .NNets
using .GeneticAlgorithm


export Neuron,
    feed,
    digital_comparator,
    train!,
    train_neuron!,
    sigmoid_neuron!,
    step_neuron!,
    isclose,
    predict,
    int_to_one_hot,
    nn_plot

export 
    Ent,
    build_ent,
    compute_fitness

    
end