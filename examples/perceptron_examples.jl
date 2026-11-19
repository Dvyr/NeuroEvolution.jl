include(joinpath(@__DIR__, "..", "src", "NeuroEvolution.jl"))

const NN = NeuroEvolution.NNets

using Plots: plot, plot!, scatter

p = NN.sigmoid_neuron!([-1.0, -1.0], 2.0)
NN.feed(p, [0.0, 1.0])  # Return 1
for _ in 1:1000
    NN.train_neuron!(p, [[0.0, 1.0]], [0.0], 1, 0.1)
end
println("After training: ", NN.feed(p, [0.0, 1.0]))  # Return 0

# Train an "AND" perceptron using train_neuron!
p2 = NN.sigmoid_neuron!([-1.0, -1.0], 2.0)
train_set = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
desired_outputs = [0.0, 0.0, 0.0, 1.0]
p2, train_errors = NN.train_neuron!(p2, train_set, desired_outputs, 5000, 0.05; 
        return_average_errors_per_epoch = true);
println("After training with train_neuron!: ", NN.feed(p2, [0.0, 1.0]))  # Return 0("Training errors per epoch: ", train_errors)
plot(train_errors, title="Training Errors per Epoch", xlabel="Epoch", ylabel="Error")