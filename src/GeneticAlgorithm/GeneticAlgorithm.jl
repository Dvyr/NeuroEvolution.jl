module GeneticAlgorithm

include("entity.jl")
include("genetic_operations.jl")
include("population.jl")

export Ent,
    build_ent,
    compute_fitness,
    is_better,
    generate_population,
    TournamentSelection,
    select_parent,
    build_new_population,
    GACrossover,
    GAMutation,
    OnePointCrossover,
    RandomPointCrossover,
    MutationOperation,
    apply_crossover,
    apply_mutation,
    mutate,
    mutate!

end