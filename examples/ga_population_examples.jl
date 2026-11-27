include(joinpath(@__DIR__, "..", "src", "GeneticAlgorithm", "GeneticAlgorithm.jl"))
const GA = GeneticAlgorithm

using Random
using StatsPlots

fact = (i::Int, ent::GA.Ent) -> rand(['a', 'b','e'])
br_line() = println("-"^min(displaysize(stdout)[2], 80))

print_ascii_gene = (ent::GA.Ent) -> println(join(ent.genome))

""" Note: This snippet finds the indices where letters differ between two strings.
# Indices where letters differ
[i for (i, (c1, c2)) in enumerate(zip(s1, s2)) if c1 != c2]
# Output: [4, 5]
"""

## Basic individual is represented by GeneticAlgorithm.Ent with a genome array
ent1 = GA.Ent([1, 2, 3]);  # Example of creating an individual with a genome array [1, 2, 3]
println("Ent created by hand: " * string(ent1));  # Print the created individual
br_line()

# Ent from factory function
ent2 = GA.build_ent(5, fact);  # Example of creating an individual with a genome array [4, 5, 6]
println("Ent created from factory: " * string(ent2));
br_line()

## Creating a population of individuals using the factory function
pop_ascii = GA.generate_population(10, 4, fact);  
for ent in pop_ascii
    print_ascii_gene(ent)
end 
br_line()

## Compute fitness for each individual in the population
fitness = (ent::GA.Ent) -> sum(ent.genome .== 'a')  # Example fitness function counting 'a' in the genome
for ent in pop_ascii
    println("Fitness of ", join(ent.genome), ": ", fitness(ent))
end
br_line()

## Fitness histogram example

the_word = "szczerzuja"
fact_ascii = (i::Int, ent::GA.Ent) -> rand('a':'z')
fit_ascii = (ent::GA.Ent) -> count(p -> p[1] == p[2], zip(ent.genome, the_word))
pop_ascii = GA.generate_population(100000, 10, fact_ascii);
for ent in pop_ascii
    # verbose print for debugging
    # print_ascii_gene(ent)
    ent.fitness = fit_ascii(ent)
end

fitness_values = [ent.fitness for ent in pop_ascii]
histogram(
    fitness_values, 
    bins=0:maximum(fitness_values), 
    xlabel="Fitness", 
    ylabel="Count", 
    title="Fitness Histogram")

