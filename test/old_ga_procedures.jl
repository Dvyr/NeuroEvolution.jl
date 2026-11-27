using Test
using Random

include(joinpath(@__DIR__, "..", "src", "GeneticAlgorithm", "GeneticAlgorithm.jl"))

@testset "GeneticAlgorithm" begin
    @testset "Ent constructor" begin
        genome = [1, 2, 3]
        ent = GeneticAlgorithm.Ent(genome)

        @test ent.genome === genome
        @test ent.fitness === nothing
    end

    @testset "build_ent builds genome from factory" begin
        calls = Int[]
        ent = GeneticAlgorithm.build_ent(4, (i, _) -> begin
            push!(calls, i)
            return i * 2
        end)

        @test calls == [1, 2, 3, 4]
        @test ent.genome == [2, 4, 6, 8]
        @test ent.fitness === nothing
    end

    @testset "build_ent can use partially built entity state" begin
        ent = GeneticAlgorithm.build_ent(3, (i, entity) -> begin
            prefix_sum = foldl(
                +,
                (Int(v) for v in entity.genome[1:(i - 1)] if v !== nothing);
                init = 0,
            )
            return prefix_sum + i
        end)

        @test ent.genome == [1, 3, 7]
    end

    @testset "build_ent edge cases" begin
        ent = GeneticAlgorithm.build_ent(0, (_, _) -> begin
            return 0
        end)

        @test isempty(ent.genome)
        @test_throws ArgumentError GeneticAlgorithm.build_ent(-1, (i, entity) -> i)
    end

    @testset "compute_fitness" begin
        ent = GeneticAlgorithm.Ent([1, 2, 3])
        calls = Ref(0)
        f = genome -> begin
            calls[] += 1
            return sum(genome)
        end

        @test GeneticAlgorithm.compute_fitness(ent, f) == 6
        @test GeneticAlgorithm.compute_fitness(ent, f) == 6
        @test calls[] == 1
        @test ent.fitness == 6

        bad_ent = GeneticAlgorithm.Ent([1, 2, 3])
        @test_throws ArgumentError GeneticAlgorithm.compute_fitness(bad_ent, _ -> "not-a-number")
    end

    @testset "generate_population" begin
        calls = Ref(0)
        population = GeneticAlgorithm.generate_population(3, 2, (i, ent) -> begin
            calls[] += 1
            return i + length(ent.genome)
        end)

        @test length(population) == 3
        @test all(ent -> ent.genome == [3, 4], population)
        @test calls[] == 6
        @test all(ent -> ent.fitness === nothing, population)
        @test all(ent -> ent !== population[1], population[2:end])

        empty_population = GeneticAlgorithm.generate_population(0, 2, (i, ent) -> i)
        @test isempty(empty_population)

        @test_throws ArgumentError GeneticAlgorithm.generate_population(-1, 2, (i, ent) -> i)
        @test_throws ArgumentError GeneticAlgorithm.generate_population(1, -1, (i, ent) -> i)
    end

    @testset "TournamentSelection and parent selection" begin
        @test_throws ArgumentError GeneticAlgorithm.TournamentSelection(0)

        selection = GeneticAlgorithm.TournamentSelection(3)
        population = [
            GeneticAlgorithm.Ent([1, 1]),
            GeneticAlgorithm.Ent([3, 3]),
            GeneticAlgorithm.Ent([2, 2]),
        ]
        fitness = genome -> sum(genome)

        Random.seed!(42)
        parent = GeneticAlgorithm.select_parent(selection, population, fitness, >)
        @test parent in population

        @test_throws ArgumentError GeneticAlgorithm.select_parent(
            selection,
            GeneticAlgorithm.Ent{Int}[],
            fitness,
            >,
        )
    end

    @testset "crossover operations" begin
        p1 = GeneticAlgorithm.Ent([1, 2, 3, 4])
        p2 = GeneticAlgorithm.Ent([9, 8, 7, 6])

        child = GeneticAlgorithm.apply_crossover(
            GeneticAlgorithm.OnePointCrossover(2),
            p1,
            p2,
        )
        @test child.genome == [1, 2, 7, 6]

        Random.seed!(7)
        random_child = GeneticAlgorithm.apply_crossover(
            GeneticAlgorithm.RandomPointCrossover(),
            p1,
            p2,
        )
        @test length(random_child.genome) == 4
        @test random_child.genome != p1.genome

        @test_throws ArgumentError GeneticAlgorithm.apply_crossover(
            GeneticAlgorithm.OnePointCrossover(0),
            p1,
            p2,
        )
        @test_throws ArgumentError GeneticAlgorithm.apply_crossover(
            GeneticAlgorithm.RandomPointCrossover(),
            GeneticAlgorithm.Ent([1]),
            GeneticAlgorithm.Ent([2]),
        )

        struct DummyCrossover <: GeneticAlgorithm.GACrossover end
        @test_throws MethodError GeneticAlgorithm.apply_crossover(
            DummyCrossover(),
            p1,
            p2,
        )
    end

    @testset "mutation operations" begin
        ent = GeneticAlgorithm.Ent([1, 2, 3])
        op = GeneticAlgorithm.MutationOperation(1.0)

        mutated = GeneticAlgorithm.mutate(op, ent, (i, _) -> i * 10)
        @test mutated.genome == [10, 20, 30]
        @test ent.genome == [1, 2, 3]

        mutated2 = GeneticAlgorithm.apply_mutation(op, ent, i -> i + 100)
        @test mutated2.genome == [101, 102, 103]

        mutated3 = GeneticAlgorithm.apply_mutation(op, ent, () -> 999)
        @test mutated3.genome == [999, 999, 999]

        @test_throws ArgumentError GeneticAlgorithm.apply_mutation(
            op,
            ent,
            (a, b, c) -> a + b + c,
        )

        struct DummyMutation <: GeneticAlgorithm.GAMutation end
        @test_throws MethodError GeneticAlgorithm.apply_mutation(DummyMutation(), ent)
        @test_throws MethodError GeneticAlgorithm.apply_mutation(
            DummyMutation(),
            ent,
            () -> 0,
        )
    end

    @testset "build_new_population" begin
        selection = GeneticAlgorithm.TournamentSelection(2)
        current_population = [
            GeneticAlgorithm.Ent([1, 1, 1]),
            GeneticAlgorithm.Ent([5, 5, 5]),
            GeneticAlgorithm.Ent([2, 2, 2]),
        ]
        fitness = genome -> sum(genome)
        compare_fitness = >
        crossover = GeneticAlgorithm.OnePointCrossover(1)
        mutation = GeneticAlgorithm.MutationOperation(1.0)
        gene_factory = () -> 9

        next_population_elitism = GeneticAlgorithm.build_new_population(
            selection,
            current_population,
            fitness,
            compare_fitness,
            crossover,
            mutation,
            gene_factory,
            true,
        )
        @test length(next_population_elitism) == length(current_population)
        @test next_population_elitism[1] === current_population[2]
        @test all(ent -> ent.genome == [9, 9, 9], next_population_elitism[2:end])

        next_population_no_elitism = GeneticAlgorithm.build_new_population(
            selection,
            current_population,
            fitness,
            compare_fitness,
            crossover,
            mutation,
            gene_factory,
            false,
        )
        @test length(next_population_no_elitism) == length(current_population)
        @test all(ent -> ent.genome == [9, 9, 9], next_population_no_elitism)

        @test_throws ArgumentError GeneticAlgorithm.build_new_population(
            selection,
            GeneticAlgorithm.Ent{Int}[],
            fitness,
            compare_fitness,
            crossover,
            mutation,
            gene_factory,
            false,
        )

        next_population_percent = GeneticAlgorithm.build_new_population(
            selection,
            current_population,
            fitness,
            compare_fitness,
            crossover,
            mutation,
            gene_factory,
            50.0,
        )
        @test next_population_percent[1] === current_population[2]
        @test next_population_percent[2] === current_population[3]
        @test next_population_percent[3].genome == [9, 9, 9]

        @test_throws ArgumentError GeneticAlgorithm.build_new_population(
            selection,
            current_population,
            fitness,
            compare_fitness,
            crossover,
            mutation,
            gene_factory,
            -1.0,
        )
        @test_throws ArgumentError GeneticAlgorithm.build_new_population(
            selection,
            current_population,
            fitness,
            compare_fitness,
            crossover,
            mutation,
            gene_factory,
            120.0,
        )
    end
end