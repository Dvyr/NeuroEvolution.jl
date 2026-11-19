
mutable struct Neuron{T<:Number}
    weights::Vector{T}
    bias::T
    activation_function::Union{Nothing,Function}
    derivative_activation_function::Union{Nothing,Function}
    delta::T
    output::T
end

@inline function isclose(x::Real, y::Real; atol::Real = 0.01)
    x1, y1, atol1 = promote(float(x), float(y), float(atol))
    return abs(x1 - y1) <= atol1
end

function Neuron(
    weights::AbstractVector{T},
    bias::T,
) where {T<:Number}
    return Neuron{T}(
        Vector{T}(weights),
        bias,
        nothing,
        nothing,
        zero(T),
        zero(T),
    )
end

function Neuron(
    weights::AbstractVector{T},
    bias::T,
    activation_function::Function,
    derivative_activation_function::Function,
) where {T<:Number}
    return Neuron{T}(
        Vector{T}(weights),
        bias,
        activation_function,
        derivative_activation_function,
        zero(T),
        zero(T),
    )
end

sigmoid(x) = 1 / (1 + exp(-x))

function sigmoid_derivative(x)
    return x * (1 - x)
end

function sigmoid_neuron!(weights::AbstractVector{T}, bias::T) where {T<:Number}
    return Neuron(
        weights,
        bias,
        sigmoid,
        sigmoid_derivative,
    )
end

function step_neuron!(weights::AbstractVector{T}, bias::T) where {T<:Number}
    return Neuron(
        weights,
        bias,
        x -> x > 0 ? one(T) : zero(T),
        x -> zero(T),
    )
end

function feed(
    n::Neuron{T}, 
    inputs::AbstractVector{T}
) where {T<:Number}
    @assert length(n.weights) == length(inputs) "Inputs and weights should have same size"
    z = sum(w * x for (w, x) in zip(n.weights, inputs)) + n.bias
    if n.activation_function === nothing
        n.output = z > 0 ? one(T) : zero(T)
        return n.output
    end
    n.output = n.activation_function(z)
    return n.output
end

function digital_comparator(a::Float64, b::Float64)
    @assert (a in [0.0, 1.0]) && (b in [0.0, 1.0]) "Each argument must be either 0 or 1"
    and = Neuron([1.0, 1.0], -1.5)
    not = Neuron([-1.0], 0.5)
    nor = Neuron([-1.0, -1.0], 0.5)

    not_a = feed(not, [a])
    not_b = feed(not, [b])

    a_less_than_b = feed(and, [not_a, b])
    a_greater_than_b = feed(and, [a, not_b])
    a_equals_to_b = feed(nor, [a_greater_than_b, a_less_than_b])
    return [a_greater_than_b, a_equals_to_b, a_less_than_b]
end

function train!(
	n::Neuron,
	inputs::Vector{T},
	desired_output::K,
    learning_rate = 0.01
) where T <: Number where K <: Number
    output = feed(n, inputs)
    computed_error = desired_output - output
    if n.derivative_activation_function === nothing
        delta = computed_error
    else
        delta = computed_error * n.derivative_activation_function(output)
    end
    for (index, input) in enumerate(inputs)
        n.weights[index] += learning_rate * delta * input
    end
    n.bias += learning_rate * delta
end

"""
    train_neuron!(
        neuron,
        inputs,
        desired_outputs,
        no_epochs,
        learning_rate;
        return_average_errors_per_epoch=false,
    )

    Trains a perceptron for a specified number of epochs using the provided training data.
    Arguments:
    - neuron: The perceptron neuron to be trained.
    - inputs: A vector of input vectors for training.
    - desired_outputs: A vector of desired outputs corresponding to the input vectors.
    - no_epochs: The number of training epochs (default: 100).
    - learning_rate: The learning rate for weight updates (default: 0.01).
    - return_average_errors_per_epoch: If true, also returns a vector of average
      absolute errors (one value per epoch).
"""
function train_neuron!(
    neuron::Neuron{Float64},
    inputs::Vector{Vector{Float64}}, 
    desired_outputs::Vector{Float64},
    no_epochs::Int = 100,
    learning_rate::Float64 = 0.1;
    return_average_errors_per_epoch::Bool = false,
    )
    @assert length(inputs) == length(desired_outputs) 
    "Number of input sets must match number of desired outputs"
    average_errors_per_epoch = Float64[]
    for _ in 1:no_epochs
        total_error = 0.0
        for (input_set, desired_output) in zip(inputs, desired_outputs)
            output = feed(neuron, input_set)
            total_error += abs(desired_output - output)
            train!(neuron, input_set, desired_output, learning_rate)
        end
        push!(average_errors_per_epoch, total_error / length(desired_outputs))
    end
    if return_average_errors_per_epoch
        return neuron, average_errors_per_epoch
    end
    return neuron
end