import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    y = np.zeros(shape=x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                y[i][j] = 0
            else:
                y[i][j] = x[i][j]
    return y


def relu_derivative(x):
    y = np.zeros(shape=x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                y[i][j] = 0
            else:
                y[i][j] = 1
    return y


def loss(predict_y, y):
    return 0.5 * (predict_y - y) * (predict_y - y)


class NeuralNetwork:
    # Initialization of neural network
    def __init__(self, inp_count, layers):
        self.inp_count = inp_count
        self.lay_count = len(layers)
        self.weights = []
        self.bias = []

        self.weights.append(np.random.uniform(size=(inp_count, layers[0])))
        for i in range(1, self.lay_count):
            self.weights.append(np.random.uniform(size=(layers[i - 1], layers[i])))

        for i in range(0, self.lay_count):
            self.bias.append(np.random.uniform(size=(1, layers[i])))

    def get_info(self):
        for i in range(self.lay_count):
            print("weights : ", *self.weights[i])
            print("bias : ", *self.bias[i])

    # Output for a input vector
    def get_output(self, inputs):
        output = inputs
        for i in range(self.lay_count - 1):
            output = relu(np.dot(output, self.weights[i]) + self.bias[i])
        output = sigmoid(np.dot(output, self.weights[self.lay_count - 1]) + self.bias[self.lay_count - 1])
        return output

    def learn(self, inputs, expected_output, lr):
        y = [inputs]
        for i in range(0, self.lay_count):
            y.append(relu(np.dot(y[i], self.weights[i]) + self.bias[i]))

        n = self.lay_count
        d_weight = []
        d_activation = []

        d_error_activation = (y[n] - expected_output) * relu_derivative(y[n - 1])
        d_error_weight = np.dot(y[n - 1].T, d_error_activation)
        d_weight.append(d_error_weight)
        d_activation.append(d_error_activation)

        for i in range(n - 2, -1, -1):
            d_error_activation = np.dot(d_error_activation, self.weights[i + 1].T) * relu_derivative(y[i])
            d_error_weight = np.dot(y[i].T, d_error_activation)
            d_weight.append(d_error_weight)
            d_activation.append(d_error_activation)

        for i in range(n):
            self.weights[n - i - 1] -= d_weight[i] * lr
            self.bias[n - i - 1] -= np.sum(d_activation[i]) * lr
