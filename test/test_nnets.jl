using Test

include(joinpath(@__DIR__, "..", "src", "NNets", "NNets.jl"))

@testset "NNets" begin
    @testset "perceptron feed" begin
        n1 = NNets.Neuron([1.0, 1.0], 0.0)
        @test NNets.feed(n1, [1.0, 1.0]) == 1
        @test NNets.feed(n1, [-1.0, -1.0]) == 0
        @test NNets.feed(n1, [1.0, -1.0]) == 0

        n2 = NNets.Neuron([2.0, -1.0], 1.0)
        @test NNets.feed(n2, [1.0, 1.0]) == 1
        @test NNets.feed(n2, [0.0, 0.0]) == 1
        @test NNets.feed(n2, [0.0, 3.0]) == 0

        n3 = NNets.Neuron([1.0, 2.0], -2.0)
        @test NNets.feed(n3, [5.0, 2.0])  == 1
	    @test NNets.feed(n3, [-5.0, 2.0]) == 0
    end

    @testset "AND gate" begin
        n_and = NNets.Neuron([1.0, 1.0], -1.5)
        @test NNets.feed(n_and, [0.0, 0.0]) == 0
        @test NNets.feed(n_and, [0.0, 1.0]) == 0
        @test NNets.feed(n_and, [1.0, 0.0]) == 0
        @test NNets.feed(n_and, [1.0, 1.0]) == 1
    end

    @testset "OR gate" begin
        n_or = NNets.Neuron([1.0, 1.0], -0.5)
        @test NNets.feed(n_or, [0.0, 0.0]) == 0
        @test NNets.feed(n_or, [0.0, 1.0]) == 1
        @test NNets.feed(n_or, [1.0, 0.0]) == 1
        @test NNets.feed(n_or, [1.0, 1.0]) == 1
    end

    @testset "NOR gate" begin
        n_nor = NNets.Neuron([-1.0, -1.0], 0.5)
        @test NNets.feed(n_nor, [0.0, 0.0]) == 1
        @test NNets.feed(n_nor, [0.0, 1.0]) == 0
        @test NNets.feed(n_nor, [1.0, 0.0]) == 0
        @test NNets.feed(n_nor, [1.0, 1.0]) == 0
    end

    @testset "NOT gate" begin
        n_not = NNets.Neuron([-1.0], 0.5)
        @test NNets.feed(n_not, [0.0]) == 1
        @test NNets.feed(n_not, [1.0]) == 0
    end

    @testset "Digital comparator" begin
        @test NNets.digital_comparator(0.0, 0.0) == [0, 1, 0]
        @test NNets.digital_comparator(0.0, 1.0) == [0, 0, 1]
        @test NNets.digital_comparator(1.0, 0.0) == [1, 0, 0]
        @test NNets.digital_comparator(1.0, 1.0) == [0, 1, 0]
    end

    @testset "train! updates parameters" begin
        n = NNets.Neuron([0.2, -0.1], -0.5)
        trained = NNets.train!(n, [1.0, 0.0], 1.0, 0.1)
        @test trained === n
        @test n.weights[1] ≈ 0.3
        @test n.weights[2] ≈ -0.1
        @test n.bias ≈ -0.4
    end

    @testset "train! validates neuron training arguments" begin
        n = NNets.Neuron([1.0, 1.0], 0.0)
        @test_throws ArgumentError NNets.train!(n, [1.0], 1.0)
        @test_throws ArgumentError NNets.train!(n, [1.0, 1.0], 1.0, -0.1)
    end

    @testset "train! keeps parameters when error is zero" begin
        n = NNets.Neuron([1.0, 1.0], -0.5)
        initial_weights = copy(n.weights)
        initial_bias = n.bias
        NNets.train!(n, [1.0, 0.0], 1.0, 0.1)
        @test n.weights == initial_weights
        @test n.bias == initial_bias
    end

    @testset "feed stores neuron output" begin
        n_step = NNets.Neuron([1.0, 1.0], -0.5)
        y_step = NNets.feed(n_step, [1.0, 0.0])
        @test n_step.output == y_step

        n_sig = NNets.sigmoid_neuron!([1.0, 1.0], -0.5)
        y_sig = NNets.feed(n_sig, [1.0, 0.0])
        @test n_sig.output == y_sig
    end

    @testset "train_neuron! returns average error history" begin
        n = NNets.Neuron([0.0, 0.0], -1.0)
        trained, avg_errors = NNets.train_neuron!(
            n,
            [[0.0, 0.0]],
            [1.0],
            3,
            0.6;
            return_average_errors_per_epoch = true,
        )
        @test trained === n
        @test avg_errors == [1.0, 1.0, 0.0]
    end

    @testset "train_neuron! default return stays neuron-only" begin
        n = NNets.Neuron([0.0, 0.0], -1.0)
        result = NNets.train_neuron!(n, [[0.0, 0.0]], [1.0], 1, 0.1)
        @test result === n
    end

    @testset "input validation" begin
        n = NNets.Neuron([1.0, 2.0], 0.0)
        @test_throws AssertionError NNets.feed(n, [1.0])
    end

    @testset "OR gate training with sigmoid neuron" begin
        n_or = NNets.sigmoid_neuron!([-1.0, -1.0], 2.0)
        NNets.train_neuron!(
            n_or,
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            [0.0, 1.0, 1.0, 1.0],
            10_000,
            0.1,
        )
        @test NNets.feed(n_or, [0.0, 0.0]) < 0.1
        @test NNets.feed(n_or, [0.0, 1.0]) > 0.9
        @test NNets.feed(n_or, [1.0, 0.0]) > 0.9
        @test NNets.feed(n_or, [1.0, 1.0]) > 0.9
    end

    @testset "Layer constructor links adjacent layers" begin
        output_layer = NNets.Layer(1, 2)
        hidden_layer = NNets.Layer(2, 2, output_layer)
        @test hidden_layer.next_layer === output_layer
        @test output_layer.previous_layer === hidden_layer
        @test length(hidden_layer.neurons) == 2
    end

    @testset "Layer feed propagates to next layer" begin
        first = NNets.Neuron([1.0, 0.0], 0.0)
        second = NNets.Neuron([0.0, 1.0], 0.0)
        output_neuron = NNets.Neuron([1.0, 1.0], -1.5)

        output_layer = NNets.Layer{Float64}([output_neuron], nothing, nothing)
        input_layer = NNets.Layer{Float64}([first, second], nothing, output_layer)
        output_layer.previous_layer = input_layer

        @test NNets.feed(input_layer, [0.0, 0.0]) == [0.0]
        @test NNets.feed(input_layer, [1.0, 0.0]) == [0.0]
        @test NNets.feed(input_layer, [0.0, 1.0]) == [0.0]
        @test NNets.feed(input_layer, [1.0, 1.0]) == [1.0]
    end

    @testset "Layer constructor validates dimensions" begin
        @test_throws AssertionError NNets.Layer(0, 2)
        @test_throws AssertionError NNets.Layer(2, 0)
    end

    @testset "NeuralNet constructor builds linked layers" begin
        net = NNets.NeuralNet(2, [3, 2]; output_count = 1)
        l1 = net.root_layer
        l2 = l1.next_layer
        l3 = l2.next_layer

        @test length(l1.neurons) == 3
        @test length(l2.neurons) == 2
        @test length(l3.neurons) == 1
        @test l1.previous_layer === nothing
        @test l2.previous_layer === l1
        @test l3.previous_layer === l2
        @test l3.next_layer === nothing
        @test length(l1.neurons[1].weights) == 2
        @test length(l2.neurons[1].weights) == 3
        @test length(l3.neurons[1].weights) == 2
    end

    @testset "NeuralNet utility constructors" begin
        net1 = NNets.one_hidden_layer_net(2, 4; output_count = 1)
        @test length(net1.root_layer.neurons) == 4
        @test length(net1.root_layer.next_layer.neurons) == 1

        net2 = NNets.two_hidden_layer_net(2, 5, 3; output_count = 1)
        @test length(net2.root_layer.neurons) == 5
        @test length(net2.root_layer.next_layer.neurons) == 3
        @test length(net2.root_layer.next_layer.next_layer.neurons) == 1
    end

    @testset "NeuralNet input validation" begin
        @test_throws AssertionError NNets.NeuralNet(0, [2])
        @test_throws AssertionError NNets.NeuralNet(2, Int[])
        @test_throws AssertionError NNets.NeuralNet(2, [2, 0])
        @test_throws AssertionError NNets.NeuralNet(2, [2]; output_count = 0)
    end

    @testset "NeuralNet backpropagation" begin
        net = NNets.NeuralNet(1, [1]; output_count = 1)
        hidden_neuron = net.root_layer.neurons[1]
        output_neuron = net.root_layer.next_layer.neurons[1]

        hidden_neuron.weights[1] = 1.0
        hidden_neuron.bias = 0.0
        output_neuron.weights[1] = 1.0
        output_neuron.bias = 0.0

        NNets.feed(net, [1.0])
        NNets.backpropagate(net, [0.0])

        @test output_neuron.delta < 0.0
        @test hidden_neuron.delta < 0.0

        initial_hidden_weight = hidden_neuron.weights[1]
        initial_output_weight = output_neuron.weights[1]
        NNets.update_weights(net, [1.0], 0.1)

        @test hidden_neuron.weights[1] != initial_hidden_weight
        @test output_neuron.weights[1] != initial_output_weight

        @test NNets.train!(net, [1.0], [0.0], 0.1) === net
        @test isempty(net.errors)
        @test isempty(net.accuracies)
        @test_throws ArgumentError NNets.train!(net, [1.0], [0.0], -0.1)
        @test_throws ArgumentError NNets.train!(net, [1.0], [0.0], Inf)
        @test_throws ArgumentError NNets.train!(net, [1.0], Float64[])
        @test_throws ArgumentError NNets.train!(net, Float64[], [0.0], 0.1)
    end

    @testset "NeuralNet prediction" begin
        net = NNets.NeuralNet(1, [1]; output_count = 1)
        hidden_neuron = net.root_layer.neurons[1]
        output_neuron = net.root_layer.next_layer.neurons[1]
        hidden_neuron.weights[1] = 2.0
        hidden_neuron.bias = -0.5
        output_neuron.weights[1] = 4.0
        output_neuron.bias = -2.0

        @test NNets.predict(net, [1.0]) == 1
        @test NNets.predict(net, [0.0]) == 0

        multi = NNets.NeuralNet(1, [1]; output_count = 2)
        multi_hidden = multi.root_layer.neurons[1]
        multi_hidden.weights[1] = 10.0
        multi_hidden.bias = 0.0
        first_output = multi.root_layer.next_layer.neurons[1]
        second_output = multi.root_layer.next_layer.neurons[2]
        first_output.weights[1] = -10.0
        first_output.bias = 0.0
        second_output.weights[1] = 10.0
        second_output.bias = 0.0

        @test NNets.predict(multi, [1.0]) == 1
        @test_throws ArgumentError NNets.predict(multi, [1.0, 1.0])
    end

    @testset "NeuralNet plot validation" begin
        net = NNets.NeuralNet(1, [1]; output_count = 1)
        @test_throws ArgumentError NNets.nn_plot(net)

        push!(net.errors, 0.2)
        @test_throws ArgumentError NNets.nn_plot(net)

        push!(net.accuracies, 0.8)
        push!(net.accuracies, 0.9)
        @test_throws ArgumentError NNets.nn_plot(net)

        trained_net = NNets.NeuralNet(1, [1]; output_count = 1)
        NNets.train!(trained_net, [[1.0]], [[0.0]], 1, 0.1)
        @test_nowarn NNets.nn_plot(trained_net)
    end

    @testset "NeuralNet train! stores one metric per epoch" begin
        net = NNets.NeuralNet(1, [1]; output_count = 1)
        output = NNets.feed(net, [1.0])[1]
        squared_error_1 = (0.0 - output)^2
        squared_error_2 = (1.0 - output)^2
        predicted_class = output >= 0.5 ? 1.0 : 0.0
        accuracy_1 = predicted_class == 0.0 ? 1.0 : 0.0
        accuracy_2 = predicted_class == 1.0 ? 1.0 : 0.0

        NNets.train!(net, [[1.0]], [[0.0]], 1, 0.0)
        @test net.errors[1] ≈ squared_error_1
        @test net.accuracies[1] ≈ accuracy_1

        NNets.train!(net, [[1.0], [1.0]], [[0.0], [1.0]], 1, 0.0)
        @test net.errors[2] ≈ (squared_error_1 + squared_error_2) / 2
        @test net.accuracies[2] ≈ (accuracy_1 + accuracy_2) / 2

        scalar_net = NNets.NeuralNet(1, [1]; output_count = 1)
        @test NNets.train!(scalar_net, [[0.0], [1.0]], [0.0, 1.0], 3, 0.1) === scalar_net
        @test length(scalar_net.errors) == 3
        @test length(scalar_net.accuracies) == 3
        @test_throws ArgumentError NNets.train!(scalar_net, [[0.0]], [0.0], -1, 0.1)
        @test_throws ArgumentError NNets.train!(scalar_net, Vector{Vector{Float64}}(), Float64[], 1, 0.1)
    end

    @testset "NeuralNet save/load" begin
        net = NNets.NeuralNet(2, [2]; output_count = 2)
        hidden_1 = net.root_layer.neurons[1]
        hidden_2 = net.root_layer.neurons[2]
        output_1 = net.root_layer.next_layer.neurons[1]
        output_2 = net.root_layer.next_layer.neurons[2]

        hidden_1.weights .= [1.0, -1.0]
        hidden_1.bias = 0.25
        hidden_2.weights .= [0.5, 0.5]
        hidden_2.bias = -0.25
        output_1.weights .= [2.0, -2.0]
        output_1.bias = 0.1
        output_2.weights .= [-2.0, 2.0]
        output_2.bias = -0.1

        NNets.train!(net, [[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], 2, 0.05)
        probe_input = [1.0, 0.0]
        expected_feed = NNets.feed(net, probe_input)
        expected_predict = NNets.predict(net, probe_input)

        mktempdir() do dir
            filename = joinpath(dir, "saved_network.bin")
            NNets.save_nn(net, filename)
            loaded = NNets.load_nn(filename)

            @test loaded isa NNets.NeuralNet
            @test loaded !== net
            @test NNets.feed(loaded, probe_input) == expected_feed
            @test NNets.predict(loaded, probe_input) == expected_predict
            @test loaded.errors == net.errors
            @test loaded.accuracies == net.accuracies

            loaded.root_layer.neurons[1].weights[1] += 1.0
            @test loaded.root_layer.neurons[1].weights[1] != net.root_layer.neurons[1].weights[1]
        end

        @test_throws SystemError NNets.load_nn(joinpath(pwd(), "does_not_exist.nn"))
    end

    @testset "NeuralNet save/load JSON" begin
        net = NNets.NeuralNet(2, [2]; output_count = 2)
        hidden_1 = net.root_layer.neurons[1]
        hidden_2 = net.root_layer.neurons[2]
        output_1 = net.root_layer.next_layer.neurons[1]
        output_2 = net.root_layer.next_layer.neurons[2]

        hidden_1.weights .= [0.3, -0.4]
        hidden_1.bias = 0.15
        hidden_2.weights .= [0.7, 0.2]
        hidden_2.bias = -0.35
        output_1.weights .= [1.2, -0.7]
        output_1.bias = 0.05
        output_2.weights .= [-1.2, 0.7]
        output_2.bias = -0.05
        net.errors = [0.5, 0.3]
        net.accuracies = [0.25, 0.75]

        probe_input = [1.0, 0.0]
        expected_feed = NNets.feed(net, probe_input)
        expected_predict = NNets.predict(net, probe_input)

        mktempdir() do dir
            filename = joinpath(dir, "saved_network.json")
            NNets.save_nn_json(net, filename)
            loaded = NNets.load_nn_json(filename)

            @test loaded isa NNets.NeuralNet
            @test loaded !== net
            @test NNets.feed(loaded, probe_input) == expected_feed
            @test NNets.predict(loaded, probe_input) == expected_predict
            @test loaded.errors == net.errors
            @test loaded.accuracies == net.accuracies

            json_text = read(filename, String)
            @test occursin("\"input_size\": 2", json_text)
            @test occursin("\"layers\"", json_text)
        end

        @test_throws SystemError NNets.load_nn_json(joinpath(pwd(), "does_not_exist.json"))
    end

    @testset "NeuralNet print function" begin
        net = NNets.NeuralNet(2, [3, 2]; output_count = 1)
        text = mktemp() do path, io
            redirect_stdout(io) do
                NNets.print_nn(net)
            end
            close(io)
            read(path, String)
        end

        @test occursin("Neural Network Structure:", text)
        @test occursin("Input size: 2", text)
        @test occursin("Output size: 1", text)
        @test occursin("Hidden layers: [3, 2]", text)
        @test occursin("Layer 1 size: 3", text)
        @test occursin("Layer 2 size: 2", text)
        @test occursin("Layer 3 size: 1", text)
    end

    @testset "NeuralNet print summary function" begin
        net = NNets.NeuralNet(2, [3, 2]; output_count = 1)
        text = mktemp() do path, io
            redirect_stdout(io) do
                NNets.print_nn_summary(net)
            end
            close(io)
            read(path, String)
        end

        @test occursin("Neural Network Summary:", text)
        @test occursin("Input size: 2", text)
        @test occursin("Output size: 1", text)
        @test occursin("Hidden layers: 2", text)
        @test occursin("Hidden layer sizes: [3, 2]", text)
        @test occursin("Total layers: 3", text)
    end
end

