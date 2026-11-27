## Genetic Algorithm Operations Examples
include(joinpath(@__DIR__, "..", "src", "GeneticAlgorithm", "GeneticAlgorithm.jl"))
const GA = GeneticAlgorithm
using Random
using StatsPlots

## Crossover Operation Examples
p1 = GeneticAlgorithm.Ent([1, 2, 3, 4, 11, 12, 13, 14])
p2 = GeneticAlgorithm.Ent([9, 8, 7, 6, 15, 16, 17, 18])

child = GeneticAlgorithm.apply_crossover(
    GeneticAlgorithm.OnePointCrossover(4),
    p1,
    p2,
)
println("Child: ", child.genome)

## Random point crossover examples
random_child = GeneticAlgorithm.apply_crossover(
            GeneticAlgorithm.RandomPointCrossover(),
            p1,
            p2,
        )
println("Random Child: ", random_child.genome)

## ASCII genes
# Example factory function for generating individuals
function fact1()
    return rand(['a', 'b', 'c', 'd', 'e'])
end
# Generate genome as a String of ASCII from the factory function
function build_ascii_gene(length::Int, gene_factory::Function)
    return String([gene_factory() for i in 1:length])
end

# Example individuals for crossover
ent1 = GA.Ent(build_ascii_gene( 5, fact1))
ent2 = GA.Ent(build_ascii_gene( 5, fact1))
println("Parent 1: ", ent1.genome)
println("Parent 1: ", join(ent1.genome))
println("Parent 2: ", join(ent2.genome))

# Perform crossover (assuming GA.crossover exists)
child1 = GA.apply_crossover(GA.RandomPointCrossover(), ent1, ent2)
println("Child 1: ", join(child1.genome))

## Mutation Operation Examples
op = GA.MutationOperation(0.1)
mutated_child = GA.apply_mutation(op, child1, fact1)
println("Mutated Child: ", join(mutated_child.genome))

println("Original Ent 1: ", join(ent1.genome))
GA.mutate!(op, ent1, fact1)
println("Entity 1 after in-place mutation: ", join(ent1.genome))
