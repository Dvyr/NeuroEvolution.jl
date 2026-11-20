include(joinpath(@__DIR__, "..", "src", "NeuroEvolution.jl"))

const NN = NeuroEvolution.NNets

using Plots: plot, plot!, scatter

## Sigmoid neuron training
p = NN.sigmoid_neuron!([-1.0, -1.0], 2.0)
NN.feed(p, [0.0, 1.0])  # Return 1
for _ in 1:1000
    NN.train_neuron!(p, [[0.0, 1.0]], [0.0], 1, 0.1)
end
println("After training: ", NN.feed(p, [0.0, 1.0]))  # Return 0

## AND perceptron training
p2 = NN.sigmoid_neuron!([-1.0, -1.0], 2.0)
train_set = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
desired_outputs = [0.0, 0.0, 0.0, 1.0]
p2, train_errors = NN.train_neuron!(
    p2,
    train_set,
    desired_outputs,
    5000,
    0.05;
    return_average_errors_per_epoch = true,
)
println("After training with train_neuron!: ", NN.feed(p2, [0.0, 1.0]))  # Return 0
println("Final training error: ", train_errors[end])
plot(train_errors, title="Training Errors per Epoch", xlabel="Epoch", ylabel="Error")

## XOR neural network training
n = NN.one_hidden_layer_net(2, 3; output_count = 1)
xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_outputs = [0.0, 1.0, 1.0, 0.0]
NN.train!(n, xor_inputs, xor_outputs, 5000, 0.1)

println("After training: ", NN.feed(n, [0.0, 1.0]))  # Return 1 (XOR output)
println("After training: ", NN.feed(n, [1.0, 1.0]))  # Return 0 (XOR output)
println("After training: ", NN.feed(n, [0.0, 0.0]))  # Return 0 (XOR output)
println("After training: ", NN.feed(n, [1.0, 0.0]))  # Return 1 (XOR output)

## Binary to Decimal neural network training
b2d_inputs = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
b2d_outputs = NN.int_to_one_hot.(0:7, 8)

b2d_net = NN.one_hidden_layer_net(3, 8; output_count = 8);
NN.train!(b2d_net, b2d_inputs, b2d_outputs, 2000, 0.1)

println("After training, prediction for [1,0,1] is: ", NN.predict(b2d_net, [1.0, 0.0, 1.0]))  # Return 5 (Binary to Decimal output)
println("After training, prediction for [0,1,1] is: ", NN.predict(b2d_net, [0.0, 1.0, 1.0]))  # Return 3 (Binary to Decimal output)
println("After training, prediction for [1,1,1] is: ", NN.predict(b2d_net, [1.0, 1.0, 1.0]))  # Return 7 (Binary to Decimal output)
println("After training, prediction for [0,0,0] is: ", NN.predict(b2d_net, [0.0, 0.0, 0.0]))  # Return 0 (Binary to Decimal output)

NN.nn_plot(b2d_net)