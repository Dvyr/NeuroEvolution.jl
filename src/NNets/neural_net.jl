
using JSON
using Plots: plot, plot!, twinx
using Serialization: deserialize, serialize

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
    return NeuralNet(
        Float64,
        input_count,
        hidden_neurons_per_layer;
        output_count = output_count,
    )
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
    return one_hidden_layer_net(
        Float64,
        input_count,
        hidden_count;
        output_count = output_count,
    )
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

# Backpropagation

function adjust_delta_with(neuron::Neuron{T}, error::T) where {T<:Real}
    if neuron.derivative_activation_function === nothing
        neuron.delta = error
    else
        neuron.delta = error * neuron.derivative_activation_function(neuron.output)
    end
    return neuron.delta
end

function adjust_weight_with_inputs(
    neuron::Neuron{T},
    inputs::AbstractVector{T},
    learning_rate::T,
) where {T<:Real}
    @assert length(neuron.weights) == length(inputs) "Inputs and weights should have same size"
    for (index, input) in enumerate(inputs)
        neuron.weights[index] += learning_rate * neuron.delta * input
    end
    neuron.bias += learning_rate * neuron.delta
    return neuron
end

function output_layer(net::NeuralNet{T}) where {T<:Real}
    layer = net.root_layer
    while layer.next_layer !== nothing
        layer = layer.next_layer
    end
    return layer
end

function _input_size(net::NeuralNet)
    return length(net.root_layer.neurons[1].weights)
end

function _output_size(net::NeuralNet)
    return length(output_layer(net).neurons)
end

function _hidden_layers(net::NeuralNet)
    hidden_sizes = Int[]
    layer = net.root_layer
    while layer.next_layer !== nothing
        push!(hidden_sizes, length(layer.neurons))
        layer = layer.next_layer
    end
    return hidden_sizes
end

function _layer_sequence(net::NeuralNet)
    sequence = typeof(net.root_layer)[]
    layer = net.root_layer
    while true
        push!(sequence, layer)
        layer.next_layer === nothing && break
        layer = layer.next_layer
    end
    return sequence
end

function _to_typed_vector(::Type{T}, values::AbstractVector) where {T<:Real}
    return values isa Vector{T} ? values : Vector{T}(values)
end

function _prediction_from_outputs(outputs::AbstractVector{T}) where {T<:Real}
    if length(outputs) == 1
        return outputs[1] >= T(0.5) ? 1 : 0
    end
    return findmax(outputs)[2] - 1
end

function _validate_training_sample(
    net::NeuralNet{T},
    inputs::AbstractVector,
    expected_outputs::AbstractVector,
) where {T<:Real}
    length(inputs) == _input_size(net) ||
        throw(ArgumentError("Input size must match network input size"))
    length(expected_outputs) == length(output_layer(net).neurons) ||
        throw(ArgumentError("Expected output count must match network output count"))
end

function _train_step!(
    net::NeuralNet{T},
    converted_inputs::Vector{T},
    converted_outputs::Vector{T},
    learning_rate::T,
) where {T<:Real}
    outputs = feed(net, converted_inputs)
    backpropagate(net, converted_outputs)
    update_weights(net, converted_inputs, learning_rate)
    return outputs
end

function update_weights(
    net::NeuralNet{T},
    initial_inputs::AbstractVector{T},
    learning_rate::T,
) where {T<:Real}
    update_weights(net.root_layer, initial_inputs, learning_rate)
    return net
end

function update_weights(
    layer::Layer{T},
    inputs::AbstractVector{T},
    learning_rate::T,
) where {T<:Real}
    for neuron in layer.neurons
        adjust_weight_with_inputs(neuron, inputs, learning_rate)
    end

    if layer.next_layer !== nothing
        next_inputs = T[n.output for n in layer.neurons]
        update_weights(layer.next_layer, next_inputs, learning_rate)
    end
    return layer
end

function update_weights(layer::Layer{T}, learning_rate::T) where {T<:Real}
    @assert layer.previous_layer !== nothing "The root layer does not store its inputs"
    inputs = T[neuron.output for neuron in layer.previous_layer.neurons]
    return update_weights(layer, inputs, learning_rate)
end

function backpropagate(
    net::NeuralNet{T},
    expected_outputs::AbstractVector{T},
) where {T<:Real}
    @assert length(expected_outputs) == length(output_layer(net).neurons) "Expected output count must match network output count"
    backpropagate(output_layer(net), expected_outputs)
    return net
end

function backpropagate(
    layer::Layer{T},
    expected_outputs::AbstractVector{T},
) where {T<:Real}
    @assert length(layer.neurons) == length(expected_outputs) "Expected output count must match layer neuron count"
    for (neuron, expected_output) in zip(layer.neurons, expected_outputs)
        adjust_delta_with(neuron, expected_output - neuron.output)
    end

    if layer.previous_layer !== nothing
        backpropagate(layer.previous_layer)
    end
    return layer
end

# Propagate previously computed errors through the network.
function backpropagate(layer::Layer{T}) where {T<:Real}
    @assert layer.next_layer !== nothing "Only hidden layers can propagate from a next layer"

    for (index, neuron) in enumerate(layer.neurons)
        propagated_error = zero(T)
        for next_neuron in layer.next_layer.neurons
            propagated_error += next_neuron.weights[index] * next_neuron.delta
        end
        adjust_delta_with(neuron, propagated_error)
    end

    if layer.previous_layer !== nothing
        backpropagate(layer.previous_layer)
    end
    return layer
end

function train!(
    net::NeuralNet{T},
    inputs::AbstractVector,
    expected_outputs::AbstractVector,
    learning_rate::Real = 0.1,
) where {T<:Real}
    isfinite(learning_rate) ||
        throw(ArgumentError("learning_rate must be finite"))
    learning_rate >= zero(learning_rate) ||
        throw(ArgumentError("learning_rate must be non-negative"))
    _validate_training_sample(net, inputs, expected_outputs)

    converted_inputs = _to_typed_vector(T, inputs)
    converted_outputs = _to_typed_vector(T, expected_outputs)
    _train_step!(net, converted_inputs, converted_outputs, convert(T, learning_rate))
    return net
end

function train!(
    net::NeuralNet{T},
    dataset_inputs::AbstractVector{<:AbstractVector},
    dataset_expected_outputs::AbstractVector{<:AbstractVector},
    epochs_count::Integer,
    learning_rate::Real = 0.1,
) where {T<:Real}
    isfinite(learning_rate) ||
        throw(ArgumentError("learning_rate must be finite"))
    learning_rate >= zero(learning_rate) ||
        throw(ArgumentError("learning_rate must be non-negative"))
    epochs_count >= 0 || throw(ArgumentError("epochs_count must be non-negative"))
    length(dataset_inputs) == length(dataset_expected_outputs) ||
        throw(ArgumentError("Number of input sets must match number of expected output sets"))
    !isempty(dataset_inputs) || throw(ArgumentError("Training dataset must not be empty"))
    length(net.errors) == length(net.accuracies) ||
        throw(ArgumentError("net.errors and net.accuracies must have the same length"))

    sample_count = length(dataset_inputs)
    typed_learning_rate = convert(T, learning_rate)

    for _ in 1:epochs_count
        epoch_mse = 0.0
        epoch_hits = 0

        for (sample_inputs, sample_expected_outputs) in zip(dataset_inputs, dataset_expected_outputs)
            _validate_training_sample(net, sample_inputs, sample_expected_outputs)
            converted_inputs = _to_typed_vector(T, sample_inputs)
            converted_outputs = _to_typed_vector(T, sample_expected_outputs)
            outputs = _train_step!(net, converted_inputs, converted_outputs, typed_learning_rate)

            squared_error = sum((expected - output)^2 for (expected, output) in zip(converted_outputs, outputs))
            epoch_mse += Float64(squared_error) / length(converted_outputs)

            predicted_class = _prediction_from_outputs(outputs)
            expected_class = _prediction_from_outputs(converted_outputs)
            epoch_hits += predicted_class == expected_class ? 1 : 0
        end

        push!(net.errors, epoch_mse / sample_count)
        push!(net.accuracies, epoch_hits / sample_count)
    end

    return net
end

function train!(
    net::NeuralNet{T},
    dataset_inputs::AbstractVector{<:AbstractVector},
    dataset_expected_outputs::AbstractVector{<:Real},
    epochs_count::Integer,
    learning_rate::Real = 0.1,
) where {T<:Real}
    wrapped_outputs = [[output] for output in dataset_expected_outputs]
    return train!(net, dataset_inputs, wrapped_outputs, epochs_count, learning_rate)
end

function predict(net::NeuralNet{T}, inputs::AbstractVector) where {T<:Real}
    length(inputs) == _input_size(net) ||
        throw(ArgumentError("Input size must match network input size"))

    outputs = feed(net, _to_typed_vector(T, inputs))
    return _prediction_from_outputs(outputs)
end

# a helper function for converting integers to one-hot encoded vectors
function int_to_one_hot(value::Integer, length::Integer)
    length > 0 || throw(ArgumentError("length must be positive"))
    0 <= value < length || throw(ArgumentError("value must be in range [0, length-1]"))
    vec = zeros(Int, length)
    vec[value + 1] = 1
    return vec
end

# Visualization of learning errors and accuracy.
function nn_plot(net::NeuralNet)
    isempty(net.errors) &&
        throw(ArgumentError("net.errors is empty; collect error values before plotting"))
    isempty(net.accuracies) &&
        throw(ArgumentError("net.accuracies is empty; collect accuracy values before plotting"))
    length(net.errors) == length(net.accuracies) ||
        throw(ArgumentError("net.errors and net.accuracies must have the same length"))

    plot(
        net.errors,
        label = "error",
        xlabel = "epoch",
        legend = :topleft,
        y_foreground_color_border = :blue,
        y_foreground_color_text = :blue,
        y_guidefontcolor = :blue,
        ylims = (0, :auto),
    )

    plot!(
        twinx(),
        net.accuracies,
        color = :red,
        xticks = :none,
        label = "accuracy",
        legend = :topright,
        y_foreground_color_border = :red,
        y_foreground_color_text = :red,
        y_guidefontcolor = :red,
        ylims = (0, 1),
    )
end

# A function for saving the trained neural network to a file
function save_nn(net::NeuralNet, filename::AbstractString)
    open(filename, "w") do io
        serialize(io, net)
    end
end

# A function for loading a trained neural network from a file
function load_nn(filename::AbstractString)
    open(filename, "r") do io
        return deserialize(io)
    end
end

# A function for saving the trained neural network to a JSON file.
function save_nn_json(net::NeuralNet, filename::AbstractString)
    layers = [
        Dict(
            "weights" => [Float64.(neuron.weights) for neuron in layer.neurons],
            "biases" => Float64[neuron.bias for neuron in layer.neurons],
        ) for layer in _layer_sequence(net)
    ]

    data = Dict{String,Any}(
        "input_size" => _input_size(net),
        "hidden_layers" => _hidden_layers(net),
        "output_size" => _output_size(net),
        "errors" => copy(net.errors),
        "accuracies" => copy(net.accuracies),
        "layers" => layers,
    )

    open(filename, "w") do io
        JSON.print(io, data, 2)
    end
    return nothing
end

# A function for loading a trained neural network from a JSON file.
function load_nn_json(filename::AbstractString)
    data = JSON.parsefile(filename)
    input_size = Int(data["input_size"])
    hidden_layers = Int.(data["hidden_layers"])
    output_size = Int(data["output_size"])
    layers_data = data["layers"]

    net = NeuralNet(Float64, input_size, hidden_layers; output_count = output_size)
    layers = _layer_sequence(net)
    length(layers_data) == length(layers) ||
        throw(ArgumentError("Layer count in file does not match reconstructed network"))

    for (layer_data, layer) in zip(layers_data, layers)
        weights_data = layer_data["weights"]
        biases_data = layer_data["biases"]
        length(weights_data) == length(layer.neurons) ||
            throw(ArgumentError("Neuron count does not match layer definition"))
        length(biases_data) == length(layer.neurons) ||
            throw(ArgumentError("Bias count does not match layer definition"))

        for (neuron, serialized_weights, serialized_bias) in zip(layer.neurons, weights_data, biases_data)
            length(serialized_weights) == length(neuron.weights) ||
                throw(ArgumentError("Weight count does not match neuron definition"))
            neuron.weights .= Float64.(serialized_weights)
            neuron.bias = Float64(serialized_bias)
        end
    end

    net.errors = haskey(data, "errors") ? Float64.(data["errors"]) : Float64[]
    net.accuracies = haskey(data, "accuracies") ? Float64.(data["accuracies"]) : Float64[]
    length(net.errors) == length(net.accuracies) ||
        throw(ArgumentError("errors and accuracies in JSON must have the same length"))
    return net
end

# A function for printing the structure and parameters of a neural network
function print_nn(net::NeuralNet)
    println("Neural Network Structure:")
    println("Input size: ", _input_size(net))
    println("Output size: ", _output_size(net))
    println("Hidden layers: ", _hidden_layers(net))

    layer = net.root_layer
    layer_index = 1
    while true
        weights = [copy(neuron.weights) for neuron in layer.neurons]
        biases = [neuron.bias for neuron in layer.neurons]
        println("Layer ", layer_index, " size: ", length(layer.neurons))
        println("  Weights: ", weights)
        println("  Biases: ", biases)

        layer.next_layer === nothing && break
        layer = layer.next_layer
        layer_index += 1
    end
end

# Print a short summary of the neural network structure
function print_nn_summary(net::NeuralNet)
    hidden_layer_sizes = _hidden_layers(net)
    println("Neural Network Summary:")
    println("Input size: ", _input_size(net))
    println("Output size: ", _output_size(net))
    println("Hidden layers: ", length(hidden_layer_sizes))
    println("Hidden layer sizes: ", hidden_layer_sizes)
    println("Total layers: ", length(_layer_sequence(net)))
end