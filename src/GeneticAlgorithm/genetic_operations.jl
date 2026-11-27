
abstract type GACrossover end

abstract type GAMutation end

# return a new Ent after applying the crossover operation
function apply_crossover(crossover::GACrossover, parent1::Ent, parent2::Ent)::Ent
    throw(MethodError(apply_crossover, (crossover, parent1, parent2)))
end

# return a new Ent after applying the mutation operation
function apply_mutation(mutation::GAMutation, ent::Ent)::Ent
    throw(MethodError(apply_mutation, (mutation, ent)))
end

function apply_mutation(mutation::GAMutation, ent::Ent, gene_factory)::Ent
    throw(MethodError(apply_mutation, (mutation, ent, gene_factory)))
end

struct OnePointCrossover <: GACrossover
    point::Int
end

struct RandomPointCrossover <: GACrossover
end

function apply_crossover(c::RandomPointCrossover, parent1::Ent, parent2::Ent)::Ent
    length(parent1.genome) == length(parent2.genome) ||
        throw(ArgumentError("Parents must have genomes of equal length"))

    n = length(parent1.genome)
    n >= 2 || throw(ArgumentError("Genomes must have at least 2 genes for crossover"))
    point = rand(1:(n-1))

    child_genome = vcat(
        parent1.genome[1:point],
        parent2.genome[(point + 1):end],
    )

    return Ent(child_genome)
end

function apply_crossover(c::OnePointCrossover, parent1::Ent, parent2::Ent)::Ent
    length(parent1.genome) == length(parent2.genome) ||
        throw(ArgumentError("Parents must have genomes of equal length"))

    n = length(parent1.genome)
    n >= 2 || throw(ArgumentError("Genomes must have at least 2 genes for crossover"))
    1 <= c.point < n ||
        throw(ArgumentError("Crossover point must be in 1:(genome_length-1)"))

    child_genome = vcat(
        parent1.genome[1:c.point],
        parent2.genome[(c.point + 1):end],
    )

    return Ent(child_genome)
end

function _call_gene_factory(gene_factory, i::Int, ent::Ent)
    if applicable(gene_factory, i, ent)
        return gene_factory(i, ent)
    elseif applicable(gene_factory, i)
        return gene_factory(i)
    elseif applicable(gene_factory)
        return gene_factory()
    else
        throw(ArgumentError("gene_factory must accept (), (i), or (i, ent)"))
    end
end

function mutate(op::T, ent::Ent, gene_factory) where T <: GAMutation
    new_ent = Ent(copy(ent.genome))
    mutate!(op, new_ent, gene_factory)
    return new_ent
end

struct MutationOperation <: GAMutation
    mutation_rate::Float64

    function MutationOperation(mutation_rate::Float64)
        0.0 <= mutation_rate <= 1.0 ||
            throw(ArgumentError("mutation_rate must be between 0 and 1"))
        new(mutation_rate)
    end
end

function mutate!(op::MutationOperation, ent::Ent, gene_factory)
    for i in eachindex(ent.genome)
        if rand() < op.mutation_rate
            ent.genome[i] = _call_gene_factory(gene_factory, i, ent)
        end
    end
end

# UInt8 version of population type for efficiency 
"""
function mutate_uint8!(
    genome,
    alphabet,
    mutation_rate
    )
    alphabet created by collect(codeunits(string))
Mutate a UInt8 genome in-place using the given alphabet and mutation rate.
"""
function mutate_uint8!(
    genome::Vector{UInt8},
    alphabet::Vector{UInt8},
    mutation_rate::Float64
)
    for i in eachindex(genome)
        if rand() < mutation_rate
            genome[i] = rand(alphabet)
        end
    end
end

function crossover_uint8(
    parent1::Vector{UInt8},
    parent2::Vector{UInt8}
)
    length(parent1) == length(parent2) ||
        throw(ArgumentError("Parents must have genomes of equal length"))
    n = length(parent1)
    point = rand(1:n-1)

    child = similar(parent1)

    child[1:point] .= parent1[1:point]
    child[point+1:end] .= parent2[point+1:end]

    return child
end

function apply_mutation(op::MutationOperation, ent::Ent, gene_factory)::Ent
    return mutate(op, ent, gene_factory)
end

