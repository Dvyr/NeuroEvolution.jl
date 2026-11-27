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
    nn_plot,

    Ent,
    GenomeFactory,
    generate_genome,
    generate_population,
    generate_ent,
    compute_fitness!,
    best_worst_ent,
    generate_population,
    TournamentSelection,
    select_parent,
    mutate_uint8!,
    crossover_uint8

    
end