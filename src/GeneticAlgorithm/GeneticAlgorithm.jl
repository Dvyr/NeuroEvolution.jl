module GeneticAlgorithm

using Random

include("entity.jl")
include("genetic_operations.jl")
include("population.jl")
include("evolution.jl")

export Ent,
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

    #= ,
    build_new_population,
    GACrossover,
    GAMutation,
    OnePointCrossover,
    RandomPointCrossover,
    MutationOperation,
    apply_crossover,
    apply_mutation,
    mutate,
    mutate!,
    GALog,
    evolve =#

end