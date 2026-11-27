# Individual entity for the genetic algorithm.
# Ent{T} is parametric for clearer typing

# eq_byte = collect(codeunits("ACDEFGHIKLMNPQRSTVWY"))

"""
    Ent{T}

Individual entity for a genetic algorithm with a `Vector{T}` genome
and a `Float64` fitness score (`NaN` when unevaluated).
"""
mutable struct Ent{T}
    genome::Vector{T}
    fitness::Float64

    Ent(genome::Vector{T}, fitness::Float64=NaN) where {T} = new{T}(genome, fitness)
end

# Convenience outer constructors
Ent(genome::AbstractVector{T}, fitness::Real=NaN) where {T} = 
        Ent(collect(genome), Float64(fitness))
Ent(genome::AbstractString, fitness::Real=NaN) = Ent(collect(genome), Float64(fitness))

# State and cloning helpers
isevaluated(e::Ent) = !isnan(e.fitness)
Base.copy(e::Ent{T}) where {T} = Ent(copy(e.genome), e.fitness)

# Defining Base.isless and Base.isequal enables native sorting
# (sort!(population), partialsort!, maximum, minimum) with zero closures or overhead.

# Comparison operators for sorting (treats NaN as strictly worst/largest so unevaluated
# individuals sort to the end during minimization, or adjust for maximization)
@inline function Base.isless(a::Ent, b::Ent)
    isnan(b.fitness) && return !isnan(a.fitness)
    isnan(a.fitness) && return false
    return a.fitness < b.fitness
end

@inline Base.:(<)(a::Ent, b::Ent) = isless(a, b)

# 2. Structural equality (keeps unique(population) and == accurate to the genome)
@inline Base.:(==)(a::Ent, b::Ent) = (a.fitness == b.fitness) && (a.genome == b.genome)
@inline Base.isequal(a::Ent, b::Ent) = isequal(a.fitness, b.fitness) && isequal(a.genome, b.genome)
@inline Base.hash(a::Ent, h::UInt) = hash(a.genome, hash(a.fitness, h))

# Genome factory
struct GenomeFactory{T}
    alphabet::Vector{T}
    length::Int
end

function GenomeFactory(alphabet, n::Int)
    return GenomeFactory(collect(alphabet), n)
end

"""
    generate_genome(factory::GenomeFactory)

Generate a random genome using the factory's alphabet and length.
"""
function generate_genome(factory::GenomeFactory)
    return rand(factory.alphabet, factory.length)
end

"""
    generate_ent(factory::GenomeFactory)

Generate a new `Ent` with a random genome from the factory and an unevaluated fitness.
"""
function generate_ent(factory::GenomeFactory)
    return Ent(generate_genome(factory))
end
# end of generate_ent function

"""
    compute_fitness!(ent::Ent, fitness_fn) -> Float64

Evaluate `fitness_fn(ent.genome)` and cache the result in `ent.fitness` if not yet evaluated.
Returns the fitness score.
Remember that whenever a mutation or crossover operator modifies ent.genome, 
it must reset ent.fitness = NaN so compute_fitness! knows to re-evaluate the new genome
"""
@inline function compute_fitness!(ent::Ent, fitness_fn::F) where {F}
    isnan(ent.fitness) || return ent.fitness
    ent.fitness = Float64(fitness_fn(ent.genome))
    return ent.fitness
end

#=
function validate_fitness_fn(fitness_fn, sample_genome)
    val = fitness_fn(sample_genome)
    val isa Real || throw(ArgumentError("fitness_fn must return a Real, got $(typeof(val))"))
    return nothing
end
=#

# Find the best or worst Ent in a population (maximization by default)
# works for any concrete Ent{T} and any vector/view
function best_worst_ent(population::AbstractVector{<:Ent}, maximize::Bool=true)
    isempty(population) && throw(ArgumentError("Population is empty"))
    if maximize
        return findmax(population)[2], findmin(population)[2]
    else
        return findmin(population)[2], findmax(population)[2]
    end
end
