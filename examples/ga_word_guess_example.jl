# Example usage of the GeneticAlgorithm module for a word guessing problem
include(joinpath(@__DIR__, "..", "src", "GeneticAlgorithm", "GeneticAlgorithm.jl"))
const GA = GeneticAlgorithm
using Random
using StatsPlots

## Helper functions for the word guessing problem
the_word = "szczerzuja"
fact_ascii = (i::Int, ent::GA.Ent) -> rand('a':'z')
fit_ascii = (ent::GA.Ent) -> count(p -> p[1] == p[2], zip(ent.genome, the_word))
terminate_criteria_100(logs) = logs[end].generation >= 100
gene_size = length(the_word)

## Evolution start
logs = GA.evolve(
    fit_ascii,
    fact_ascii,
    gene_size;
    population_size = 100,
    termination_criteria = terminate_criteria_100,
    compare_fitness = GA.is_better,
    selection = GA.TournamentSelection(),
    crossover = GA.RandomPointCrossover(),
    mutation =GA.MutationOperation(0.1),
    elitism=true,
    verbose=true
)