using Test
using Random
using NeuroEvolution
#=
include(joinpath(@__DIR__, "..", "src", "GeneticAlgorithm", "GeneticAlgorithm.jl"))
const GA = GeneticAlgorithm =#
rng = Xoshiro(0x5518fb57531da017)

@testset "Ents and population" begin
    @testset "Ent constructor" begin
        genome = [1, 2, 3]
        ent = Ent(genome)

        @test ent.genome === genome
        @test isnan(ent.fitness)
    end

    @testset "Ent fitness" begin
        genome = [1, 2, 3]
        ent = Ent(genome, 42.0)

        @test ent.genome === genome
        @test ent.fitness == 42.0
    end

    @testset "Ent comparison" begin
        ent1 = Ent([1, 2, 3], 10.0)
        ent2 = Ent([4, 5, 6], 20.0)

        @test ent1 < ent2
        @test ent2 > ent1
        @test ent1 != ent2
        @test ent1 == Ent([1, 2, 3], 10.0)
    end

    @testset "Ent cloning" begin
        ent = Ent([1, 2, 3], 10.0)
        ent_copy = copy(ent)

        @test ent_copy == ent
        @test ent_copy !== ent
        @test ent_copy.genome !== ent.genome
    end

    @testset "GenomeFactory" begin
        factory = GenomeFactory([1, 2, 3], 5)
        genome = generate_genome(factory)

        @test length(genome) == 5
        @test all(x -> x in [1, 2, 3], genome)
    end 

    @testset "Compute fitness" begin
        ent = Ent([1, 2, 3])
        fitness_fn(genome) = sum(genome)

        @test isnan(ent.fitness)
        compute_fitness!(ent, fitness_fn)
        @test ent.fitness == 6.0
    end

    @testset "Validate sort for Ent vector" begin
        ent1 = Ent([1, 2, 3], 10.0)
        ent2 = Ent([4, 5, 6], 20.0)
        ent3 = Ent([7, 8, 9], 5.0)

        population = [ent1, ent2, ent3]
        sorted_population = sort(population)

        @test sorted_population[1] == ent3
        @test sorted_population[2] == ent1
        @test sorted_population[3] == ent2
    end

    @testset "Best Ent in population" begin
        ent1 = Ent([1, 2, 3], 10.0)
        ent2 = Ent([4, 5, 6], 20.0)
        ent3 = Ent([7, 8, 9], 5.0)

        population = [ent1, ent2, ent3]
        best, worst = best_worst_ent(population)

        @test population[best] == ent2
        @test population[worst] == ent3
    end

    @testset "ASCII genome Ent" begin
        factory = GenomeFactory(['A', 'B', 'C'], 4)
        ent = generate_ent(factory)

        @test length(ent.genome) == 4
        @test all(x -> x in ['A', 'B', 'C'], ent.genome)
    end
end