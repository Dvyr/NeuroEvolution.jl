## Initialization
include(joinpath(@__DIR__, "..", "src", "NeuroEvolution.jl"))

const NN = NeuroEvolution.NNets

using CSV
using DataFrames
using Random
using Downloads: download

## Download, save locally to ./data/ and load the Iris dataset
df = CSV.read(download("https://agileartificialintelligence.github.io/Datasets/iris.csv"), DataFrame)
# Save locally to ./data/iris.csv
mkpath("data")
CSV.write("data/iris.csv", df)

df = CSV.read("data/iris.csv", DataFrame)

## Generate one-hot encoded labels for the Iris dataset
species = unique(df.:species)
species_to_index = Dict(s => i - 1 for (i, s) in enumerate(species))
labels = [species_to_index[s] for s in df.:species]
one_hot_labels = NN.int_to_one_hot.(labels, length(species))

## Prepare input features
features = Matrix(df[:, 1:4])  # Assuming the first 4 columns are features

## Split the dataset into training and testing sets
Random.seed!(1234)  # For reproducibility
indices = shuffle(1:size(features, 1))
train_ratio = 0.8
train_size = Int(floor(train_ratio * length(indices)))
train_indices = indices[1:train_size]
test_indices = indices[train_size+1:end]

train_features = features[train_indices, :]
train_labels = one_hot_labels[train_indices]
test_features = features[test_indices, :]
test_labels = one_hot_labels[test_indices]

## Data normalization (fit on train, apply to train and test)
function minmax_params(features)
    mins = minimum(features, dims = 1)
    maxs = maximum(features, dims = 1)
    spans = maxs .- mins
    spans[spans .== 0.0] .= 1.0
    return mins, spans
end

function normalize_with(features, mins, spans)
    return (features .- mins) ./ spans
end

train_mins, train_spans = minmax_params(train_features)
train_features = normalize_with(train_features, train_mins, train_spans)
test_features = normalize_with(test_features, train_mins, train_spans)

train_inputs = [collect(row) for row in eachrow(train_features)]

## Create and train the neural network
n_inputs = size(train_features, 2)
n_hidden = 5  # Number of neurons in the hidden layer
n_outputs = length(species)

iris_net = NN.one_hidden_layer_net(n_inputs, n_hidden; output_count = n_outputs)
NN.train!(iris_net, train_inputs, train_labels, 5000, 0.1)

## Evaluate the neural network on the test set
predictions = NN.predict.(Ref(iris_net), eachrow(test_features))
expected_labels = [findfirst(x -> x == 1, lbl) - 1 for lbl in test_labels]
accuracy = sum(predictions .== expected_labels) / length(test_labels)
println("Test set accuracy: ", accuracy)    