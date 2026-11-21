# Individual entity for the genetic algorithm.
# Ent{T} is parametric for clearer typing
mutable struct Ent{T}
    genome::Vector{T}
    fitness::Union{Number,Nothing}

    Ent(genome::Vector{T}) where {T} = new{T}(genome, nothing)
end

Ent(genome::AbstractVector{T}) where {T} = Ent(collect(genome))

"""
    build_ent(genes_count, gene_factory)

Build an [`Ent`] where each gene is produced by `gene_factory(index, entity)`.
The factory can inspect already generated genes through `entity.genome`.
"""
function build_ent(genes_count::Integer, gene_factory)
    genes_count >= 0 || throw(ArgumentError("genes_count must be non-negative"))

    genome = Vector{Any}(undef, genes_count)
    fill!(genome, nothing)
    ent = Ent(genome)

    for i in 1:genes_count
        @inbounds ent.genome[i] = gene_factory(i, ent)
    end
    return ent
end

function compute_fitness(ent::Ent, fitness_fn)
    !isnothing(ent.fitness) && return ent.fitness

    computed_fitness = fitness_fn(ent.genome)
    computed_fitness isa Number ||
        throw(ArgumentError("fitness function must return a Number"))
    ent.fitness = computed_fitness
    return ent.fitness
end

function is_better(
    fitness::Function,
    ent1::Ent,
    ent2::Ent
)
    compute_fitness(ent1, fitness)
    compute_fitness(ent2, fitness)
    return ent1.fitness > ent2.fitness
end
