import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_predictions(weights, x):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(x, weights))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Binary cross-entropy loss function
def binary_cross_entropy_loss(predictions, targets):
    epsilon = 1e-15  # To prevent log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

# Accuracy function
def compute_accuracy(predictions, targets):
    pred_labels = (predictions >= 0.5).astype(int)
    return np.mean(pred_labels == targets)

# Neural Network class
class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layer_sizes = layer_sizes  # List of layer sizes [input_size, hidden1_size, ..., output_size]
        self.activations = activations  # List of activation functions per layer
        self.weights = []  # Weights for each layer
        self.biases = []   # Biases for each layer
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights and biases for each layer
        np.random.seed(0)  # For reproducibility
        for i in range(len(self.layer_sizes) - 1):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]
            if self.activations[i] == 'sigmoid':
                # Xavier Initialization
                limit = np.sqrt(6 / (input_dim + output_dim))
                W = np.random.uniform(-limit, limit, (input_dim, output_dim))
            elif self.activations[i] == 'relu':
                # He Initialization
                W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
            else:
                W = np.random.randn(input_dim, output_dim) * 0.01  # Default small weights
            b = np.zeros((1, output_dim))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        # Forward pass through the network
        self.z_values = []  # Linear combinations (Z) for each layer
        self.a_values = [X]  # Activations for each layer (A_0 = X)

        for i in range(len(self.weights)):
            Z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(Z)

            # Apply activation function
            if self.activations[i] == 'sigmoid':
                A = sigmoid(Z)
            elif self.activations[i] == 'relu':
                A = relu(Z)
            else:
                raise ValueError(f"Invalid activation function: {self.activations[i]}")

            self.a_values.append(A)

        return self.a_values[-1]  # Return the final output layer (A_L)

    def backward(self, y, learning_rate):
        # Backward pass (compute gradients and update weights)
        m = y.shape[0]
        L = len(self.weights)
        dA = -(np.divide(y, self.a_values[-1]) - np.divide(1 - y, 1 - self.a_values[-1]))

        for i in reversed(range(L)):
            # Compute derivative of activation function
            if self.activations[i] == 'sigmoid':
                dZ = dA * sigmoid_derivative(self.a_values[i + 1])
            elif self.activations[i] == 'relu':
                dZ = dA * relu_derivative(self.z_values[i])
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]}")

            dW = np.dot(self.a_values[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)

            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        # Train the neural network using gradient descent
        losses = []
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute loss
            loss = binary_cross_entropy_loss(predictions, y)
            losses.append(loss)

            # Backward pass and update weights
            self.backward(y, learning_rate)

            # Optionally print loss and accuracy
            if epoch % 100 == 0 or epoch == epochs - 1:
                accuracy = compute_accuracy(predictions, y)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        return losses

    def predict(self, X):
        # Make predictions using the trained network
        predictions = self.forward(X)
        return (predictions >= 0.5).astype(int)



def log_reg(X_train, y_train, X_test, y_test, n_epochs, minibatch_size, eta, lmbd):
    
    npoints = len(X)
    
    #initialize weights
    weights = np.zeros(X_train.shape[1])

    M = minibatch_size 
    m = int(npoints/M) 
    
    #set momentum and change values for SGD
    momentum = 0.8
    change = 0.1
    
    # intialize empty list for predictions
    predictions = []
    
    #iterate over each epoch
    for epoch in range(n_epochs):
        
        for i in range(m):
            #iterate over each minibatch and select data
            random_index = M*npr.randint(m)
            xi = X_train[random_index:random_index+M]
            yi = y_train[random_index:random_index+M]
            
            # calculate gradient
            z = xi@weights
            p = sigmoid(z)
            gradient = (xi.T @ (p - yi) - 2*lmbd*weights)/npoints

            new_change = eta * gradient + momentum * change

            weights -= new_change

            change = new_change
        
        # make predictions using logisitic_predictions (which feeds through a sigmoid function)
        ypred = logistic_predictions(weights, X_test)
        predictions.append(ypred)
    
    # calculate accuracies on test data based on threshold of 0.5
    accuracies = []
    for i in range(len(predictions)):
        ypred_ints = []
        for j in predictions[i]:
            if j < 0.5:
                ypred_ints.append(0)
            else:
                ypred_ints.append(1)
        
        accuracies.append(accuracy_score(y_test, ypred_ints))

        
    return accuracies