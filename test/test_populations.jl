using Test
using Random
using NeuroEvolution

rng = Xoshiro(0x5518fb57531da017)

@testset "Population tests" begin
    @testset "Generate population" begin
        factory = GenomeFactory(['A', 'B', 'C'], 5)
        population = generate_population(20, factory)

        @test length(population) == 20
        @test all(x -> all(y -> y in ['A', 'B', 'C'], x.genome), population)
    end

    @testset "Tournament selection" begin
        factory = GenomeFactory(['A', 'B', 'C'], 5)
        population = generate_population(20, factory)
        selection = TournamentSelection(3)
        fitness = x -> count(y -> y == 'C', x)
        parent = select_parent(selection, population, fitness)
        @test parent isa Ent
        @test parent.fitness >= 0
    end

    @testset "codeunits Char representation" begin
        s = "ACDEFGHIKLMNPQRSTVWY"
        cu = collect(codeunits(s))
        @test all(x -> x isa UInt8, cu)
    end

    @testset "UInt8 used for genomes" begin
        Random.seed!(42)
        factory = GenomeFactory(collect(codeunits("ACDEFGHIKLMNPQRSTVWY")), 10)
        population = generate_population(20, factory)
        selection = TournamentSelection(5)
        HYDROPHOBIC = Set(UInt8.(collect("AVILMFWY"))
        )
        function fitness(genome::Vector{UInt8})
            score = 0
            for aa in genome
                score += aa in HYDROPHOBIC
            end
            return score
        end
        parent = select_parent(selection, population, fitness)
        # print the UInt8 genome as String
        # println(parent.genome)
        # println("Parent genome: ", String(parent.genome))
        # println("Parent fitness: ", parent.fitness)
        # ATTENTION: String(parent.genome) consumes the UInt8 vector and converts it to a String
        @test String(parent.genome) == "MHYFEMEEWA"
        @test parent.fitness == 6.0
    end
    @testset "Mutate for UInt8 genomes" begin
        Random.seed!(42)
        alphabet = collect(codeunits("ACDEFGHIKLMNPQRSTVWY"))
        genome = collect(codeunits("ALAMAKATA"))
        mutation_rate = 0.5
        mutate_uint8!(genome, alphabet, mutation_rate)
        # println("Mutated genome: ", String(genome))
        @test String(genome) == "ALAMPKGTA"
    end

    @testset "Crossover for UInt8 genomes" begin
        Random.seed!(42)
        parent1 = collect(codeunits("ALAMAKATAR"))
        parent2 = collect(codeunits("ACDEFGHIKL"))
        child = crossover_uint8(parent1, parent2)
        # println("Child genome: ", String(child))
        @test String(copy(child)) == "ALAMAKHIKL"
        @test length(child) == length(parent1)
        @test all(x -> x in parent1 || x in parent2, child)
    end
end