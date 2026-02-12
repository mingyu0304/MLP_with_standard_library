import random
import math
import pickle


class NeuralNetwork:
    def __init__(self, layer_sizes, isSigmoid=False):
        if isSigmoid and layer_sizes[-1] != 1:
            raise IndexError("Output layer size is not 1")
        self.isSigmoid = isSigmoid
        self.former_inputs = None
        self.former_outputs = None
        self.layer_sizes = layer_sizes
        self.layer_num = len(layer_sizes)
        self.layers = []
        self.weights = []
        self.bias = []
        self.errors = []
        for i in range(self.layer_num):
            temp_layer = [0 for _ in range(self.layer_sizes[i])]
            self.layers.append(temp_layer)
        for i in range(self.layer_num - 1):
            a = math.sqrt(6 / self.layer_sizes[i])
            print(a)
            temp_weight = [
                [random.uniform(-a, a) for _ in range(self.layer_sizes[i])]
                for _ in range(self.layer_sizes[i + 1])
            ]
            self.weights.append(temp_weight)
        for i in range(self.layer_num - 1):
            temp_bias = [0 for _ in range(self.layer_sizes[i + 1])]
            self.bias.append(temp_bias)
        for i in range(self.layer_num - 1):
            temp_error = [0 for _ in range(self.layer_sizes[i + 1])]
            self.errors.append(temp_error)

    def weight_bias_copy(self, model_addr):
        target_model = None
        with open(model_addr, mode="rb") as f:
            target_model = pickle.load(f)
        if self.layer_sizes != target_model.layer_sizes:
            print("Unmatching Model")
            return
        for i in range(len(target_model.weights)):
            for j in range(len(target_model.weights[i])):
                self.weights[i][j] = target_model.weights[i][j].copy()
        for i in range(len(target_model.bias)):
            self.bias[i] = target_model.bias[i].copy()

    def relu(self, x):
        return max(0, x)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0

    def sigmoid(self, x):
        result = [1 / (1 + math.exp(-x))]
        return result

    def dot_product1(self, matrix, vector):
        n = len(matrix)
        m = len(matrix[0])
        if len(vector) != m:
            print("Size Error")
            return
        result = [0] * n
        for i in range(n):
            result[i] = sum(vector[j] * matrix[i][j] for j in range(m))
        return result

    def dot_product2(self, vector1, vector2):
        result = [[0 for _ in range(len(vector2))] for _ in range(len(vector1))]
        for i in range(len(vector1)):
            for j in range(len(vector2)):
                result[i][j] = vector1[i] * vector2[j]
        return result

    def t_dot_product(self, matrix, vector):
        n = len(matrix)
        m = len(matrix[0])
        if len(vector) != n:
            print("Size Error")
            return
        result = [0] * m
        for i in range(m):
            result[i] = sum(vector[j] * matrix[j][i] for j in range(n))
        return result

    def mul_scala_matrix(self, matrix, scala):
        result = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[i][j] = matrix[i][j] * scala
        return result

    def minus_matrix(self, matrix1, matrix2):
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                matrix1[i][j] -= matrix2[i][j]

    def restrict_matrix(self, matrix):
        result = [[0 for _ in range(len(matrix[0]))] for _ in range(matrix)]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[i][j] = max(-1, min(matrix[i][j], 1))
        return result

    def add_vector(self, v1, v2):
        temp = len(v1)
        if temp != len(v2):
            print("Size Error")
            return
        result = [0] * temp
        for i in range(temp):
            result[i] = v1[i] + v2[i]
        return result

    def minus_vector(self, v1, v2):
        temp = len(v1)
        if temp != len(v2):
            print("Size Error")
            return
        result = [0] * temp
        for i in range(temp):
            result[i] = v1[i] - v2[i]
        return result

    def mul_vector(self, v1, v2):
        temp = len(v1)
        if temp != len(v2):
            print("Size Error")
            return
        result = [0] * temp
        for i in range(temp):
            result[i] = v1[i] * v2[i]
        return result

    def mul_scala_vector(self, v, s):
        result = [0 for _ in range(len(v))]
        for i in range(len(v)):
            result[i] = s * v[i]
        return result

    def vector_relu(self, v):
        result = [0] * len(v)
        for i in range(len(v)):
            result[i] = self.relu(v[i])
        return result

    def vector_relu_derivative(self, v):
        result = [0] * len(v)
        for i in range(len(v)):
            result[i] = self.relu_derivative(v[i])
        return result

    def forward(self, inputs):
        if len(inputs) != len(self.layers[0]):
            raise IndexError("Input Size Error")
        else:
            self.layers[0] = inputs
        for i in range(self.layer_num - 1):
            temp1 = self.dot_product1(self.weights[i], self.layers[i])
            temp2 = self.add_vector(temp1, self.bias[i])
            if len(self.layers[i + 1]) != len(temp2):
                raise IndexError(f"{i+2} layer size error")
            if i == (self.layer_num - 2):
                if not self.isSigmoid:
                    self.layers[i + 1] = temp2
                else:
                    self.layers[i + 1] = self.sigmoid(temp2[0])
                break
            self.layers[i + 1] = self.vector_relu(temp2)
        self.former_inputs = inputs
        self.former_outputs = self.layers[-1].copy()
        return self.former_outputs

    def backward(self, inputs, targets, learning_rate):
        if self.former_inputs != inputs:
            print("Forward와 Backward inputs 이 다름")
        if self.isSigmoid:
            self.errors[-1] = [self.layers[-1][0] - targets]
        else:
            self.errors[-1] = self.minus_vector(self.layers[-1], targets)
        for i in range(self.layer_num - 2):
            temp1 = self.t_dot_product(self.weights[-(i + 1)], self.errors[-(i + 1)])
            temp2 = self.vector_relu_derivative(self.layers[-(i + 2)])
            self.errors[-(i + 2)] = self.mul_vector(temp1, temp2)
        for i in range(self.layer_num - 1):
            temp1 = self.dot_product2(self.errors[-(i + 1)], self.layers[-(i + 2)])
            temp2 = self.mul_scala_matrix(temp1, learning_rate)
            self.minus_matrix(self.weights[-(i + 1)], temp2)
        for i in range(self.layer_num - 1):
            temp1 = self.mul_scala_vector(self.errors[-(i + 1)], learning_rate)
            self.bias[-(i + 1)] = self.minus_vector(self.bias[-(i + 1)], temp1)
        return self.former_outputs
