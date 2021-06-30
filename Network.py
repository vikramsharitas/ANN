import copy
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
def one_hot_encoder(Y):
    uniq = np.unique(Y)
    num_unique = uniq.size
    one_hot_Y = np.zeros((Y.size, num_unique))
    for i in range(Y.size):
        one_hot_Y[i][Y[i]-1] = 1
    return one_hot_Y
def test_train_split(X_split, y_split, train_split=0.7):
    """
    Input: X and y as array of features and target attributes
    Output: X_train, X_test, y_train, y_test
    Split dataset by train_split value and suffle it
    """
    temp = np.concatenate((X_split, y_split), axis=1)
    np.random.shuffle(temp)
    s = int(len(temp) * train_split)
    
    X_train = temp[:s, :len(X_split[0])]
    X_test = temp[s:, :len(X_split[0])]
    y_train = temp[:s, len(X_split[0]):]
    y_test = temp[s:, len(X_split[0]):]
    return X_train, X_test, y_train, y_test
def preprocess(X, normal=False):
    """
    Input: X and y for both training and testing and normal is set if we want normalized vadates and is set false if we want standardized dataset\n
    Output: Preprocessed X and y for both training and testing
        1) a) Normalization is carried out (Scale the variables between 0 and 1)
                                     OR
           b) Standardization is carried out (Subtract mean and divide by standard deviation)
        2) Insert 1 for bias in the first index
    """
    #Normalization
    if normal:
        x_1 = np.amax(X, 0)
        x_2 = np.amin(X, 0)
        
        X -= x_2
        X /= (x_1 - x_2)
    
    #Standardization
    if not normal:
        x_1 = np.mean(X, 0)
        x_2 = np.std(X, 0)
        
        X -= x_1
        X /= x_2
    

    return X
def binary_cross_entropy(y_true, y_pred):
    lo1 = np.log(y_pred)
    lo2 = np.log(1-y_pred)
    ret = np.sum(np.multiply(-y_true, lo1) - np.multiply((1-y_true), lo2), axis=1)
    return np.mean(ret)
def mse(y_true, y_pred):
    # print((y_pred-y_true).shape)
    return np.mean(np.sum(np.power((y_pred - y_true), 2), axis=1))
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
def sigmoid(X):
    # print("X: ", X.shape)
    ret = 1 / (1 + np.exp(-X))
    # print("Sig: ", ret)
    return ret
def prime_sigmoid(X):
    sig = sigmoid(X)
    prime_sig = np.multiply(sig, (np.ones((X.shape[0], X.shape[1])) - sig))
    return prime_sig
def relu(X):
    X_rel = copy.deepcopy(X)
    X_rel[X_rel <= 0] = 0
    return X_rel
def relu_prime(X):
    X_rel = copy.deepcopy(X)
    X_rel[X_rel <= 0] = 0
    X_rel[X_rel > 0] = 1
    return X_rel
def leaky_relu(X):
    X_rel = copy.deepcopy(X)
    X_rel[X_rel <= 0] = 0.2*X_rel[X_rel <= 0]
    return X_rel
def leaky_relu_prime(X):
    X_rel = copy.deepcopy(X)
    X_rel[X_rel <= 0] = 0.2
    X_rel[X_rel > 0] = 1
    return X_rel
def softmax(X):
    X_exp = np.exp(X)
    X_exp_sum = np.sum(X_exp, axis=1).reshape(-1, 1)
    return np.divide(X_exp, X_exp_sum)
def softmax_prime(X):
    ones = np.ones((X.shape[0], X.shape[1]))
    sm = softmax(X)
    return np.multiply(sm, ones-sm)
def softplus(X):
    X_exp = copy.deepcopy(X)
    X_exp = np.exp(X_exp)
    return np.log(1-X_exp)
def softplus_prime(X):
    return sigmoid(X)
class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.rand(input_size + 1, output_size) - 0.5
        self.activation = activation
        if activation == sigmoid:
            self.prime_activation = prime_sigmoid
        elif activation == relu:
            self.prime_activation = relu_prime
        elif activation == softmax:
            self.prime_activation = softmax_prime
        elif activation == leaky_relu:
            self.prime_activation = leaky_relu_prime
        elif activation == softplus:
            self.prime_activation = softplus_prime
    def forward_prop(self, X):
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((X, bias), axis=1)
        self.input = X
        # print(self.input.shape, self.weights.shape)
        Z = np.matmul(self.input, self.weights)
        self.Z = Z
        if self.activation != None:
            # print("Z Before: ", Z)
            Z = self.activation(Z)
            # print("Z After: ", Z)
        return Z
    def back_prop(self, output_err, lr=0.001):
        if self.activation != None:
            # print("Z: ", self.Z)
            prime = self.prime_activation(self.Z)
            # print("Prime: ", prime)
            # print("Before: ", output_err)
            output_err = np.multiply(prime, output_err)
            # print("After:", output_err)
        # print(output_err.shape, self.weights.T.shape)
        inp_err = np.matmul(output_err, self.weights.T)
        # print(inp_err.shape)
        self.weights -= lr * np.matmul(self.input.T, output_err)
        inp_err = inp_err[:, :-1]
        return inp_err
class ANN:
    def __init__(self):
        self.layers = []
    def add(self, input_size, output_size, activation=None):
        layer = Layer(input_size=input_size, output_size=output_size, activation=activation)
        self.layers.append(layer)
    def fit(self, X_train, y_train, X_val, y_val, epochs=1000, lr=1, batch_size=None):
        history = {}
        history['val_acc'] = []
        history['val_loss'] = []
        history['trn_acc'] = []
        history['trn_loss'] = []

        if batch_size == None:
            batch_size = X_train.shape[0]

        for epoch in trange(epochs):
            X = copy.deepcopy(X_train)
            Y = copy.deepcopy(y_train)
            if batch_size > 0 and batch_size < X_train.shape[0]:
                X1 = np.concatenate((X_train, y_train), axis=1)
                X1 = X1[np.random.choice(X1.shape[0], batch_size, replace=False)]
                X = copy.deepcopy(X1[:, :X_train.shape[1]])
                Y = copy.deepcopy(X1[:, X_train.shape[1]:])
            
            if len(X) != batch_size:
                print("NOOOOOOOOO!!!!", len(X))
            # else:
            #     print(len(X), left, right)
            # Forward Prop
            for layer in self.layers:
                # print(X.shape, layer.weights.shape)
                X = layer.forward_prop(X)

            # Backward Prop
            derr = mse_prime(Y, X)
            for layer in self.layers[::-1]:
                derr = layer.back_prop(derr, lr)
            
            X_pred = self.predict(X_val)
            history['val_loss'].append(binary_cross_entropy(y_val, X_pred))
            history['val_acc'].append(np.sum(np.argmax(X_pred, axis=1) == np.argmax(y_val, axis=1)) / len(y_val))

            X_pred = self.predict(X_train)
            history['trn_loss'].append(binary_cross_entropy(y_train, X_pred))
            history['trn_acc'].append(np.sum(np.argmax(X_pred, axis=1) == np.argmax(y_train, axis=1)) / len(y_train))
        history['val_loss'] = np.array(history['val_loss'])
        history['val_acc'] = np.array(history['val_acc'])
        history['trn_loss'] = np.array(history['trn_loss'])
        history['trn_acc'] = np.array(history['trn_acc'])
        return history
    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_prop(X)
        return X
    def reset(self):
        new_layers = []
        for layer in self.layers:
            new_layers.append(Layer(layer.weights.shape[0] - 1, layer.weights.shape[1], layer.activation))
        self.layers = new_layers
def plot(history, title='', plot_loss=True):
    if plot_loss:
        fig, ax = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(title, fontsize=25, fontweight='bold')
        ax[0][0].plot(history['val_acc'])
        ax[0][0].title.set_text('Testing Accuracy')
        ax[0][1].plot(history['val_loss'])
        ax[0][1].title.set_text('Testing Loss')
        ax[1][0].plot(history['trn_acc'])
        ax[1][0].title.set_text('Training Accuracy')
        ax[1][1].plot(history['trn_loss'])
        ax[1][1].title.set_text('Training Loss')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        fig.suptitle(title, fontsize=25, fontweight='bold')
        ax[0].plot(history['val_acc'])
        ax[0].title.set_text('Testing Accuracy')
        ax[1].plot(history['trn_acc'])
        ax[1].title.set_text('Training Accuracy')
    plt.show()