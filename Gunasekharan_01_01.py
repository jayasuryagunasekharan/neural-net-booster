# Gunasekharan, Jayasurya
# 1002_060_437
# 2023_09_24
# Assignment_01_01

import numpy as np


def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))


def mse(y_true, y_pred):
    # Mean squared error calculation
    return np.mean((y_true - y_pred)**2)


def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):
    # This function creates and trains a multi-layer neural Network
    # X_train: Array of input for training [input_dimensions,nof_train_samples]

    # Y_train: Array of desired outputs for training samples [output_dimensions,nof_train_samples]
    # X_test: Array of input for testing [input_dimensions,nof_test_samples]
    # Y_test: Array of desired outputs for test samples [output_dimensions,nof_test_samples]
    # layers: array of integers representing number of nodes in each layer
    # alpha: learning rate
    # epochs: number of epochs for training.
    # h: step size
    # seed: random number generator seed for initializing the weights.

    # return: This function should return a list containing 3 elements:

    # The first element of the return list should be a list of weight matrices.
    # Each element of the list corresponds to the weight matrix of the corresponding layer.

    # The second element should be a one dimensional array of numbers
    # representing the average mse error after each epoch. Each error should
    # be calculated by using the X_test array while the network is frozen.
    # This means that the weights should not be adjusted while calculating the error.

    # The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
    # representing the actual output of network when X_test is used as input.

    # Notes:
    # DO NOT use any other package other than numpy
    # Bias should be included in the weight matrix in the first column.
    # Assume that the activation functions for all the layers are sigmoid.
    # Use MSE to calculate error.
    # Use gradient descent for adjusting the weights.
    # use centered difference approximation to calculate partial derivatives.
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()

    # Initialize a list to store weight matrices
    weights = []

    # Initialize a list to store average mse error after each epoch
    error_history = []

    # Initialize a list to store actual outputs when X_test is used as input
    actual_outputs = []

    np.random.seed(seed)

    # Initialize weights for the input layer
    input_dim = X_train.shape[0]
    hidden_layers = layers
    output_dim = Y_train.shape[0]

    for layer_index in range(len(hidden_layers) + 1):
        if layer_index == 0:
            # Input layer to the first hidden layer
            num_nodes_in_layer = hidden_layers[layer_index]
            weights.append(np.random.randn(num_nodes_in_layer, input_dim + 1))
        elif layer_index == len(hidden_layers):
            # Last hidden layer to the output layer
            num_nodes_in_layer = output_dim
            weights.append(np.random.randn(output_dim, hidden_layers[-1] + 1))
        else:
            # Hidden layers
            num_nodes_in_layer = hidden_layers[layer_index]
            weights.append(np.random.randn(num_nodes_in_layer,
                           hidden_layers[layer_index - 1] + 1))

    for epoch in range(epochs):
        mse_error = 0

        # Forward propagation
        activations = [X_train]
        for layer_weights in weights:
            input_data = np.vstack(
                (np.ones(X_train.shape[1]), activations[-1]))
            z = np.dot(layer_weights, input_data)
            activation = sigmoid(z)
            activations.append(activation)

        # Calculate error using X_test without updating weights (frozen network)
        test_activations = [X_test]
        for layer_weights in weights:
            test_input_data = np.vstack(
                (np.ones(X_test.shape[1]), test_activations[-1]))
            test_z = np.dot(layer_weights, test_input_data)
            test_activation = sigmoid(test_z)
            test_activations.append(test_activation)

        actual_outputs.append(test_activations[-1])
        mse_error = mse(Y_test, test_activations[-1])
        error_history.append(mse_error)

        # print("error_history: ", error_history)

    return weights, error_history, np.array(actual_outputs)
