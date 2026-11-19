
mutable struct Layer{T<:Real}
    neurons::Vector{Neuron{T}}
    previous_layer::Union{Nothing,Layer{T}}
    next_layer::Union{Nothing,Layer{T}}
end

function Layer(::Type{T}, neurons_count::Int, input_count::Int) where {T<:Real}
    @assert neurons_count > 0 "neurons_count must be positive"
    @assert input_count > 0 "input_count must be positive"
    neurons = [sigmoid_neuron!(randn(T, input_count), zero(T)) for _ in 1:neurons_count]
    return Layer{T}(neurons, nothing, nothing)
end

function Layer(neurons_count::Int, input_count::Int)
    return Layer(Float64, neurons_count, input_count)
end

function Layer(neurons_count::Int, input_count::Int, next_layer::Layer{T}) where {T<:Real}
    new_layer = Layer(T, neurons_count, input_count)
    next_layer.previous_layer = new_layer
    new_layer.next_layer = next_layer
    return new_layer
end

function feed(layer::Layer{T}, inputs::AbstractVector{T}) where {T<:Real}
    outputs = [feed(neuron, inputs) for neuron in layer.neurons]
    if layer.next_layer === nothing
        return outputs
    end
    return feed(layer.next_layer, outputs)
end

mutable struct NeuralNet{T<:Real}
    root_layer::Layer{T}
    errors::Vector{Float64}
    accuracies::Vector{Float64}
    NeuralNet(root_layer::Layer{T}) where {T<:Real} = new{T}(root_layer, Float64[], Float64[])
end

function feed(net::NeuralNet{T}, inputs::AbstractVector{T}) where {T<:Real}
    return feed(net.root_layer, inputs)
end

"""
    NeuralNet(::Type{T}, input_count, hidden_neurons_per_layer; output_count=1) where T<:Real

Builds a feed-forward neural net with sigmoid neurons, a configurable list of hidden
layer sizes, and a configurable output layer size.
"""
function NeuralNet(
    ::Type{T},
    input_count::Int,
    hidden_neurons_per_layer::AbstractVector{Int};
    output_count::Int = 1,
) where {T<:Real}
    @assert input_count > 0 "input_count must be positive"
    @assert !isempty(hidden_neurons_per_layer) "At least one hidden layer is required"
    @assert all(n -> n > 0, hidden_neurons_per_layer) "All hidden layer sizes must be positive"
    @assert output_count > 0 "output_count must be positive"

    layers = Layer{T}[]
    push!(layers, Layer(T, hidden_neurons_per_layer[1], input_count))
    for i in 2:length(hidden_neurons_per_layer)
        push!(layers, Layer(T, hidden_neurons_per_layer[i], hidden_neurons_per_layer[i - 1]))
    end
    push!(layers, Layer(T, output_count, hidden_neurons_per_layer[end]))

    for i in 1:(length(layers) - 1)
        layers[i].next_layer = layers[i + 1]
        layers[i + 1].previous_layer = layers[i]
    end

    return NeuralNet(layers[1])
end

function NeuralNet(
    input_count::Int,
    hidden_neurons_per_layer::AbstractVector{Int};
    output_count::Int = 1,
)
    return NeuralNet(Float64, input_count, hidden_neurons_per_layer; output_count = output_count)
end

function one_hidden_layer_net(
    ::Type{T},
    input_count::Int,
    hidden_count::Int;
    output_count::Int = 1,
) where {T<:Real}
    return NeuralNet(T, input_count, [hidden_count]; output_count = output_count)
end

function one_hidden_layer_net(input_count::Int, hidden_count::Int; output_count::Int = 1)
    return one_hidden_layer_net(Float64, input_count, hidden_count; output_count = output_count)
end

function two_hidden_layer_net(
    ::Type{T},
    input_count::Int,
    hidden1_count::Int,
    hidden2_count::Int;
    output_count::Int = 1,
) where {T<:Real}
    return NeuralNet(T, input_count, [hidden1_count, hidden2_count]; output_count = output_count)
end

function two_hidden_layer_net(
    input_count::Int,
    hidden1_count::Int,
    hidden2_count::Int;
    output_count::Int = 1,
)
    return two_hidden_layer_net(
        Float64,
        input_count,
        hidden1_count,
        hidden2_count;
        output_count = output_count,
    )
end

