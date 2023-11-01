import numpy as np

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid function derivative
def sigmoid_prime(x):
    return x * (1 - x)

# Input data 
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Expected output
output = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(0)
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

# seights and biases for the hidden layer
weights_hidden_layer = np.random.uniform(size=(input_layer_size, hidden_layer_size))
bias_hidden_layer = np.random.uniform(size=(1, hidden_layer_size))

# seights and biases for the output layer
weights_output_layer = np.random.uniform(size=(hidden_layer_size, output_layer_size))
bias_output_layer = np.random.uniform(size=(1, output_layer_size))

# learning rate
learning_rate = 1

# artificail neural network
for epoch in range(10000):
    # forward propagation
    hidden_layer_input = np.dot(input, weights_hidden_layer) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_output_layer) + bias_output_layer
    estimated_output = sigmoid(output_layer_input)
    
    # backpropagation
    error = output - estimated_output
    predicted_output_prime = error * sigmoid_prime(estimated_output)
    
    error_hidden_layer = predicted_output_prime.dot(weights_output_layer.T)
    hidden_layer_prime = error_hidden_layer * sigmoid_prime(hidden_layer_output)
    
    # update weights & biases
    weights_output_layer += hidden_layer_output.T.dot(predicted_output_prime) * learning_rate
    bias_output_layer += np.sum(predicted_output_prime, axis=0, keepdims=True) * learning_rate
    
    weights_hidden_layer += input.T.dot(hidden_layer_prime) * learning_rate
    bias_hidden_layer += np.sum(hidden_layer_prime, axis=0, keepdims=True) * learning_rate

# Testing the trained network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    hidden_layer_activation = sigmoid(np.dot(data, weights_hidden_layer) + bias_hidden_layer)
    predicted_output = sigmoid(np.dot(hidden_layer_activation, weights_output_layer) + bias_output_layer)
    print(f"Input: {data}: Estimation: {predicted_output[0]}")
