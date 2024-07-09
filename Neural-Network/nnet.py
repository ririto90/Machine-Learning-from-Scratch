import numpy as np

def relu(input_data):
    return np.maximum(0, input_data)

def relu_derivative(input_data):
    return np.where(input_data > 0, 1, 0)

def tanh(input_data):
    return np.tanh(input_data)

def tanh_derivative(input_data):
    return 1 - np.tanh(input_data)**2

def sigmoid(input_data):
    return 1 / (1 + np.exp(-input_data))

def sigmoid_derivative(input_data):
    return sigmoid(input_data) * (1 - sigmoid(input_data))

def initialize_weights_bias(input_size, output_size):
    weights = np.random.rand(input_size, output_size) - 0.5
    bias = np.random.rand(1, output_size) - 0.5
    return weights, bias

class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward_propagation(self):
        raise NotImplementedError

    def backward_propagation(self):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)

    def forward_propagation(self, input):
        self.input = input
        
        # Calculate linear transformation and add bias
        return np.dot(self.input, self.weights) + self.bias

    def backward_propagation(self, gradient_loss, learning_rate):

        # Calculate gradients for updating parameters
        self.weights_gradient = np.dot(self.input.T, gradient_loss)
        self.bias_gradient = np.sum(gradient_loss, axis=0, keepdims=True)

        # Calculate error gradient
        error_gradient = np.dot(gradient_loss, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * gradient_loss
        
        return error_gradient
    
    def __repr__(self):
        return f"{self.__class__.__name__}"


class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_derivative):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_propagation(self, input):
        self.input = input

        # Apply activation function to input data
        output = self.activation_function(self.input)
        return output

    def backward_propagation(self, gradient_loss, learning_rate):

        # Calculate gradient of the error with activation derivative
        return gradient_loss * self.activation_derivative(self.input)
    
    def __repr__(self):
        return f"{self.__class__.__name__}"


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):

        # Apply forward propagation to each layer
        for layer in self.layers:
            input_data = layer.forward_propagation(input_data)
        return input_data

    def fit(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        self.validation_losses = []
        
        for epoch in range(epochs):
            err = 0
            val_loss = 0

            # Iterate over each training sample
            for x, y in zip(X_train, y_train):

                # Reshape x to [1, x_feature]
                x = x.reshape(1, -1)

                # Calculate the models prediciton with forward propagation
                output = self.predict(x)

                # Accumulate loss between predictions
                err += self.loss(y, output)

                # Compute the derivitive of the loss function
                error = self.loss_derivative(y, output)

                # Traverse each layer and calcuate backpropagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Iterate over each training sample
            for x, y in zip(X_val, y_val):

                # Reshape x to [1, x_feature]
                x = x.reshape(1, -1)

                # Calculate the models prediciton with forward propagation
                y_pred = self.predict(x)

                # Accumulate loss between predictions on validation set
                val_loss += self.loss(y, y_pred)

            # Calculate average validation for the epoch
            val_loss /= len(X_val)
            self.validation_losses.append(val_loss)

            if epochs <= 50:
                print(f'Epoch {epoch+1}/{epochs}, Training Error: {err / len(X_train)}, Validation Loss: {val_loss}')
            elif epochs > 50 and epochs <= 100:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Training Error: {err / len(X_train)}, Validation Loss: {val_loss}')
            elif epochs > 100:
                if epoch % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Training Error: {err / len(X_train)}, Validation Loss: {val_loss}')

    @staticmethod
    def loss(y_true, y_pred):
        # Calculate MSE loss function
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def loss_derivative(y_true, y_pred):
        # Calculate derivative of MSE for backpropagation
        return 2 * (y_pred - y_true) / y_true.size