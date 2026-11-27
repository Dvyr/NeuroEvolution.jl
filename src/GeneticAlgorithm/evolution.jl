## Evolution of the population using genetic algorithm operations

# Monitoring of evolution and termination 
struct GALog
    generation::Int
    best_fitness::Float64
    worst_fitness::Float64
    average_fitness::Float64
    median_fitness::Float64
    best::Ent
end

function terminate(
    frame::Int=5,
    delta_fitness::Float64=1e-3,
    max_generations = 100
)
    function f(logs)
        current_generation = logs[end].generation

        # dor not terminate before reaching max generations
        current_generation < max_generations && return false

        # terminate if max generations reached
        current_generation >= max_generations && return true

        previous_fitness = logs[end-frame-1].best_fitness
        current_fitness = logs[end].best_fitness

        # measure the variation and act accordingly
        return abs((current_fitness - previous_fitness)/previous_fitness) <= delta_fitness
    end
    return f
end

terminate_criteria_100(logs) = logs[end].generation >= 100

# The entry point of the evolution process
function evolve(
    fitness::Function,
    gene_factory::Function,
    gene_size::Int
    ;
    population_size::Int = 100,
    termination_criteria::Function = terminate_criteria_100,
    compare_fitness::Function = >,
    selection = TournamentSelection(),
    crossover::GACrossover = RandomPointCrossover(),
    mutation::GAMutation = MutationOperation(0.1),
    elitism::Bool=true,
    verbose::Bool=true,
)
    
    logs = GALog[]
    population = generate_population(population_size, gene_size, gene_factory)
    generation = 0
    while(isempty(logs) || !termination_criteria(logs))
        new_population = build_new_population(
                    selection, 
                    population, 
                    fitness, 
                    compare_fitness, 
                    crossover, 
                    mutation, 
                    gene_factory, 
                    elitism
                )
        best = best_individual(new_population, fitness, compare_fitness)
        worst = worst_individual(new_population, fitness, compare_fitness)
        fitness_table = compute_fitness.(new_population, Ref(fitness))
        average = mean(fitness_table)
        median = median(fitness_table)
        log = GALog(
            generation, 
            best.fitness, 
            worst.fitness, 
            average, 
            median, 
            best
        )
        push!(logs, log)
        terminate_now = termination_criteria(logs)
        if terminate_now
            break
        end
        
        population = new_population
        generation += 1
    end
    return logs
end