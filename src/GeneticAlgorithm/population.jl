
"""
    generate_population(pop_size, gene_factory)

Generate a population of entities by repeatedly calling [`generate_ent`] with
`gene_factory`.
"""
function generate_population(pop_size::Integer, gene_factory::GenomeFactory)
    pop_size >= 0 || throw(ArgumentError("pop_size must be non-negative"))

    return [generate_ent(gene_factory) for _ in 1:pop_size]
end

# Parent selection

struct TournamentSelection
    tournament_size::Int
    function TournamentSelection(tournament_size::Int = 5)
        tournament_size > 0 ||
            throw(ArgumentError("tournament_size must be positive"))
        new(tournament_size)
    end
end

# select entity from population using tournament selection
# Select an individual from a `population` using the `fitness function`.
# Individuals are compared using the `compare_fitness` and the parameters of the selection
# are carried out in `selection`.
"""
    select_parent(
        selection::TournamentSelection, 
        population::AbstractVector{<:Ent}, 
        fitness::Function, 
        compare_fitness::Function
)

TBW
"""
function select_parent(
    selection::TournamentSelection, 
    population::AbstractVector{<:Ent}, 
    fitness::Function
)
    isempty(population) &&
        throw(ArgumentError("population must not be empty"))

    best = nothing
    population_size = length(population)
    for _ in 1:selection.tournament_size
        candidate = population[rand(1:population_size)]
        if best === nothing ||
            compute_fitness!(candidate, fitness) > compute_fitness!(best, fitness)
            best = candidate
        end
    end
    return best
end

function build_new_population(
    selection::TournamentSelection,
    current_population::AbstractVector{<:Ent},
    fitness::Function,
    compare_fitness::Function,
    crossover::GACrossover,
    mutation::GAMutation,
    gene_factory,
    elitism::Bool
)
    elitism_percent = elitism ? (100 / max(length(current_population), 1)) : 0.0
    return build_new_population(
        selection,
        current_population,
        fitness,
        compare_fitness,
        crossover,
        mutation,
        gene_factory,
        elitism_percent,
    )
end

function build_new_population(
    selection::TournamentSelection,
    current_population::AbstractVector{<:Ent},
    fitness::Function,
    compare_fitness::Function,
    crossover::GACrossover,
    mutation::GAMutation,
    gene_factory,
    elitism_percent::Real
)
    isempty(current_population) &&
        throw(ArgumentError("current_population must not be empty"))
    0 <= elitism_percent <= 100 ||
        throw(ArgumentError("elitism_percent must be in [0, 100]"))

    new_population = Vector{Ent}(undef, length(current_population))

    # step 1: handle elitism (keep the best percentage)
    elite_count = ceil(Int, length(current_population) * (elitism_percent / 100))
    elite_count = min(elite_count, length(current_population))

    if elite_count > 0
        sorted_population = sort(
            collect(current_population);
            lt = (a, b) -> compare_fitness(
                compute_fitness(a, fitness),
                compute_fitness(b, fitness),
            ),
        )
        new_population[1:elite_count] = sorted_population[1:elite_count]
    end

    # step 2: fill the rest of the new population
    for i in (elite_count + 1):length(new_population)
        parent1 = select_parent(selection, current_population, fitness)
        parent2 = select_parent(selection, current_population, fitness)
        child = apply_crossover(crossover, parent1, parent2)
        child = apply_mutation(mutation, child, gene_factory)
        new_population[i] = child
    end
    return new_population
end

function best_individual(
    population::AbstractVector{<:Ent},
    fitness::Function,
    compare_fitness::Function
)
    isempty(population) &&
        throw(ArgumentError("population must not be empty"))

    best = population[1]
    for candidate in population[2:end]
        if compare_fitness(
            compute_fitness(candidate, fitness),
            compute_fitness(best, fitness),
        )
            best = candidate
        end
    end
    return best
end

function worst_individual(
    population::AbstractVector{<:Ent},
    fitness::Function,
    compare_fitness::Function
)
    isempty(population) &&
        throw(ArgumentError("population must not be empty"))

    worst = population[1]
    for candidate in population[2:end]
        if !compare_fitness(
            compute_fitness(candidate, fitness),
            compute_fitness(worst, fitness),
        )
            worst = candidate
        end
    end
    return worst
end

