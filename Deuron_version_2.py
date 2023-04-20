import copy
from typing import Final
import numpy as np
import time
import os
from matplotlib import pyplot as plt
import zipfile

class Deuron():
    """
    ===== Deuron Class =====
    
    ========================
    
    MADE BY: ÜNAL DALKILIÇ
    
    ========================

    Deuron is the neural network framework that includes:

    There are 5 activation functions available relu, leaky_relu, sigmoid, tanh, softmax; please check Deuron.ACTIVATION...

    --> Regularization methods such as Dropout and L1, L2 regularization is

    (Functions: set_regularization, cancel_regularization)

    --> Feature scaling methods such as Normalization, Mean Normalization and Standardization

    (Functions: normalize, mean_normalize, standardize, transform, redo_transform, cancel_transform)

    --> Optimization Methods such as adam and momentum

    (Functions: set_optimizer, cancel_optimizer)

    --> Learning decay that has ability to optionally be defined the decay time interval

    (Functions: learning_decay)

    --> Mini batches that can increase learning convergence speed

    (Functions: set_mini_batch, cancel_mini_batch)

    ...
    """

    class COST():
        """
        Deuron.COST.SQUARE_ERROR_COST

        Deuron.COST.CROSS_ENTROPY_COST

        Deuron.COST.EXTENDED_CROSS_ENTROPY_COST
        """

        SQUARE_ERROR_COST : Final[str] = "Square_Error_Cost"
        CROSS_ENTROPY_COST : Final[str] = "Cross_Entropy_Cost"
        EXTENDED_CROSS_ENTROPY_COST : Final[str] = "Extended_Cross_Entropy_Cost"
        values : Final[list] = [SQUARE_ERROR_COST, CROSS_ENTROPY_COST, EXTENDED_CROSS_ENTROPY_COST]
    class ACTIVATION():
        """
        Deuron.ACTIVATION.RELU

        Deuron.ACTIVATION.LEAKY_RELU

        Deuron.ACTIVATION.SIGMOID

        Deuron.ACTIVATION.TANH

        Deuron.ACTIVATION.SOFTMAX
        """

        RELU : Final[str] = "Relu"
        LEAKY_RELU : Final[str] = "Leaky_Relu"
        SIGMOID : Final[str] = "Sigmoid"
        TANH : Final[str] = "Tanh"
        SOFTMAX : Final[str] = "Softmax"
        values : Final[list] = [RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX]
    class TRANSFORM():
        """
        Deuron.TRANSFORM.NORMALIZED

        Deuron.TRANSFORM.MEAN_NORMALIZED

        Deuron.TRANSFORM.STANDARDIZED

        Deuron.TRANSFORM.NO_TRANSFORM
        """

        NORMALIZED : Final[str] = "Normalized"
        MEAN_NORMALIZED : Final[str] = "Mean Normalized"
        STANDARDIZED : Final[str] = "Standardized"
        NO_TRANSFORM : Final[str] = "No_Transform"
    class REGULARIZATION():
        """
        Deuron.REGULARIZATION.L1

        Deuron.REGULARIZATION.L2

        Deuron.REGULARIZATION.DROPOUT

        Deuron.REGULARIZATION.NO_REGULARIZATION
        """

        L1 : Final[str] = "L1"
        L2 : Final[str] = "L2"
        DROPOUT : Final[str] = "Dropout"
        NO_REGULARIZATION : Final[str] = "No_Regularization"
        values : Final[list] = [L1, L2, DROPOUT, NO_REGULARIZATION]
    class OPTIMIZER():
        """
        Deuron.OPTIMIZER.MOMENTUM

        Deuron.OPTIMIZER.ADAM

        Deuron.OPTIMIZER.NO_OPTIMIZER
        """

        MOMENTUM : Final[str] = "Momentum"
        ADAM : Final[str] = "Adam"
        NO_OPTIMIZER : Final[str] = "No_Optimizer"
        values: Final[list] = [MOMENTUM, ADAM, NO_OPTIMIZER]

    def __init__(self, X: np.array, Y: np.array, layer_list: list, cost_type: str, epoch_number: int, alpha: float, created_with_state = False, parameter_dict = None):
        """
        === Deuron Constructor ===

        X --> Training input data matrix, (numpy.array mandatory)
        Y --> Trainig actual result data matrix, (numpy.array mandatory)
        layer_list --> Specifies the each layers' node number with their activation type as list of tuples, (list mandatory)
        exp: [(100, Deuron.ACTIVATION.SIGMOID), (30, Deuron.ACTIVATION.TANH), (5, Deuron.ACTIVATION.SOFTMAX)]
        cost_type --> Specifies the type of cost that learning algorithm going to use, (check Deuron.ACTIVATION)
        epoch_number --> Specifies the epoch number (iteration count) of learning algorithm, (int mandatory)
        alpha --> Specifies the learning rate alpha, (float mandatory)
        """

        if self.__prepare_data(X, Y) and self.__create_and_validate_layer_list(layer_list) and alpha > 0 and epoch_number > 0 and (cost_type in Deuron.COST.values):
            self.__alpha = copy.deepcopy(alpha)
            self.__epoch_number = copy.deepcopy(epoch_number)
            self.__cost_type = copy.deepcopy(cost_type)
            self.__J = [None] * self.__epoch_number
            self.__create_data_dicts()
            self.__learn_state = False
            self.__regularization_param = Deuron.REGULARIZATION.NO_REGULARIZATION
            self.__transform_param_X = Deuron.TRANSFORM.NO_TRANSFORM
            self.__optimizer = Deuron.OPTIMIZER.NO_OPTIMIZER
            self.__transform_param_Y = Deuron.TRANSFORM.NO_TRANSFORM
            self.__lambd = 0
            self.__keep_prob = 0
            self.__batch_size = self.__m
            self.__batch_count = 1
            self.__seed = np.random.randint(1000000)
            np.random.seed(self.__seed)
            self.__alpha_decay_rate = 0
            self.__learning_decay_time_interval = 100
            self.__beta_1 = 0
            self.__beta_2 = 0
            self.__epsilon = 1e-10
            if (not created_with_state):
                self.__initialize_params()
            else:
                self.__put_related_parameters(parameter_dict)
        else:
            raise Exception("Error! Invalid value for alpha, cost type or epoch number parameters")

    def __prepare_data(self, X, Y):
        # Prepares the input datas for correct and usable format for algorithm
        # Preferred format is (nx, m) for X and (ny, m) for Y

        # prefer (1,m) shape instead of (m, ) 
        if(X.shape[0] == X.size):
            X = X.reshape(1, X.shape[0])
        if(Y.shape[0] == Y.size):
            Y = Y.reshape(1, Y.shape[0])

        # if nx == ny, choose higher dimension as m and reformat
        if (X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1]) or (X.shape[0] == Y.shape[1] and X.shape[1] == Y.shape[0]):
            if max(Y.shape[0], Y.shape[1]) == max(X.shape[0], X.shape[1]):
                m = max(Y.shape[0], Y.shape[1])
                if X.shape[0] == m:
                    X = X.T
                if Y.shape[0] == m:
                    Y = Y.T
            else:
                raise Exception("Error! Invalid X and Y data pairs!\nTraining example count m not correspond as matrix dimension")

        # Reformat input data
        if X.shape[0] == Y.shape[0]:
            X = X.T
            Y = Y.T
        elif X.shape[1] == Y.shape[0]:
            Y = Y.T
        elif X.shape[0] == Y.shape[1]:
            X = X.T
        elif X.shape[1] == Y.shape[1]:
            pass
        else:
            raise Exception("Error! Invalid X and Y data pairs!\nTraining example count m not correspond as matrix dimension")

        self.__X_raw = copy.deepcopy(X)
        self.__Y_raw = copy.deepcopy(Y)
        self.__m = self.__X_raw.shape[1]
        self.__nx = self.__X_raw.shape[0]

        return True

    def __create_and_validate_layer_list(self, layer_list):
        # Creates necessary data to hold number of nodes for each layer and their types

        # Use X as A0, so add its data to corresponding lists as well
        self.__layer_number_of_nodes_list = [self.__nx]
        self.__layer_type_list = [None]
        for item in layer_list:
            if type(item) == tuple:
                node_count, layer_type = item
                if (type(node_count) == int) and (layer_type in Deuron.ACTIVATION.values):
                    self.__layer_number_of_nodes_list.append(node_count)
                    self.__layer_type_list.append(layer_type)
                else:
                    raise Exception("Error! Tuples must include first node count for layer and then type of that layer\nExp: [(100, Deuron.ACTIVATION.SOGMOID), ...]")
            else:
                raise Exception("Error! Layer list items must include tuples in it\nExp: [(100, Deuron.ACTIVATION.SOGMOID), ...]")

        self.__L = len(self.__layer_number_of_nodes_list)
        return True

    def __create_data_dicts(self):
        self.__W = dict()
        self.__B = dict()
        self.__Z = dict()
        self.__A = dict()   
        self.__dW = dict()
        self.__dB = dict()
        self.__dZ = dict()
        self.__dA = dict()
        self.__dropout_cache = dict()

    def __initialize_params(self):
        # Initializes learning weight parameters (W and B) matrices
        # Uses He initialization

        for i in range(1,self.__L):
            self.__W[str(i)] = np.random.randn(self.__layer_number_of_nodes_list[i-1], self.__layer_number_of_nodes_list[i]) * np.sqrt(2/(self.__layer_number_of_nodes_list[i-1]))
            self.__B[str(i)] = np.zeros((self.__layer_number_of_nodes_list[i], 1))

    def __put_related_parameters(self, parameter_dict):
        # Experimental
        if len(parameter_dict) == 2:
            self.__W, self.__B = copy.deepcopy(parameter_dict)
        else:
            raise Exception("Invalid parameter dictionary, please check the validity of state files")

    def set_mini_batch(self, batch_size: int):
        """
        Sets mini batch to use mini batch or stochastic gradient descent with the help of Momentum or Adam optimizer

        batch_size --> specify the batch size, (int mandatory) [! Batch size must be in interval (1 <= batch size <= m [Training example count])]
        (Hint: It would better to use batch sizes as the power of 2 such as: 16,32,64,128 etc.)
        """

        if batch_size == self.__m:
            self.cancel_mini_batch()
        elif batch_size >= 1 and batch_size < self.__m:
            self.__batch_size = copy.deepcopy(batch_size)
            if self.__m // self.__batch_size == self.__m / self.__batch_size:
                self.__batch_count = self.__m // self.__batch_size
            else:
                self.__batch_count = (self.__m // self.__batch_size) + 1
        else:
            raise Exception("Error! Invalid batch size parameter\nBatch size parameter must be in interval (1 <= batch size <= m [Training example count])")

    def cancel_mini_batch(self):
        """
        Cancels the mini batch settings and use batch gradient descent
        """

        self.__optimizer = Deuron.OPTIMIZER.NO_OPTIMIZER
        self.__batch_size = self.__m
        self.__batch_count = 1

    def __apply_mini_batch(self):
        permutation = list(np.random.permutation(self.__m))
        shuffle_X = self.__X_raw[:, permutation]
        shuffle_Y = self.__Y_raw[:, permutation]
        fitted_mini_batches = self.__m // self.__batch_size
        for i in range(fitted_mini_batches):
            self.__X_list[i] = shuffle_X[:,(i*self.__batch_size):(self.__batch_size + (i*self.__batch_size))]
            self.__Y_list[i] = shuffle_Y[:,(i*self.__batch_size):(self.__batch_size + (i*self.__batch_size))]
        if self.__m % self.__batch_size != 0:
            last_index = self.__m % self.__batch_size
            self.__X_list[-1] = shuffle_X[:,(self.__m-last_index):(self.__m)]
            self.__Y_list[-1] = shuffle_Y[:,(self.__m-last_index):(self.__m)]

    def __prepare_data_matrices(self):
        if self.__batch_count == 1:
            self.__X_list = [self.__X_raw]
            self.__Y_list = [self.__Y_raw]
        else:
            self.__X_list = [None] * self.__batch_count
            self.__Y_list = [None] * self.__batch_count

    def normalize(self, is_Y_transform = False):
        """
        To normalize X or Y matrices, ((matrix - min) / (max - min))

        is_Y_transform (Optional [Default: False]) --> if False apply for X, if True apply for Y
        """

        if is_Y_transform:
            self.__transform_param_Y = Deuron.TRANSFORM.NORMALIZED
            maxx = np.max(self.__Y_raw, axis = 1, keepdims=True)
            minn = np.min(self.__Y_raw, axis = 1, keepdims=True)
            self.__transform_data_Y = {"Max": maxx, "Min": minn}
        else:
            self.__transform_param_X = Deuron.TRANSFORM.NORMALIZED
            maxx = np.max(self.__X_raw, axis = 1, keepdims=True)
            minn = np.min(self.__X_raw, axis = 1, keepdims=True)
            self.__transform_data_X = {"Max": maxx, "Min": minn}

    def mean_normalize(self, is_Y_transform = False):
        """
        Applies mean normalization to X or Y matrices, ((matrix - mean) / (max - min))

        is_Y_transform (Optional [Default: False]) --> if False apply for X, if True apply for Y
        """

        if is_Y_transform:
            self.__transform_param_Y = Deuron.TRANSFORM.NORMALIZED
            maxx = np.max(self.__Y_raw, axis = 1, keepdims=True)
            minn = np.min(self.__Y_raw, axis = 1, keepdims=True)
            meann = np.mean(self.__Y_raw, axis = 1, keepdims=True)
            self.__transform_data_Y = {"Max": maxx, "Min": minn, "Mean": meann}
        else:
            self.__transform_param_X = Deuron.TRANSFORM.NORMALIZED
            maxx = np.max(self.__X_raw, axis = 1, keepdims=True)
            minn = np.min(self.__X_raw, axis = 1, keepdims=True)
            meann = np.mean(self.__X_raw, axis = 1, keepdims=True)
            self.__transform_data_X = {"Max": maxx, "Min": minn, "Mean": meann}

    def standardize(self, is_Y_transform = False):
        """
        To standardize X or Y matrices, ((matrix - mean) / (std))

        is_Y_transform (Optional [Default: False]) --> if False apply for X, if True apply for Y
        """

        if is_Y_transform:
            self.__transform_param_Y = Deuron.TRANSFORM.STANDARDIZED
            meann = np.mean(self.__Y_raw, axis = 1, keepdims=True)
            stdd = np.std(self.__Y_raw, axis = 1, keepdims=True)
            self.__transform_data_Y = {"Mean": meann, "Std": stdd}
        else:
            self.__transform_param_X = Deuron.TRANSFORM.STANDARDIZED
            meann = np.mean(self.__X_raw, axis = 1, keepdims=True)
            stdd = np.std(self.__X_raw, axis = 1, keepdims=True)
            self.__transform_data_X = {"Mean": meann, "Std": stdd}

    def cancel_transform(self, is_Y_cancel = False):
        """
        Cancels and resets the transform settings

        is_Y_cancel (Optional [Default: False]) --> if False apply for X, if True apply for Y
        """

        if is_Y_cancel:
            self.__transform_param_Y = Deuron.TRANSFORM.NO_TRANSFORM
            self.__transform_data_Y = dict()
        else:
            self.__transform_param_X = Deuron.TRANSFORM.NO_TRANSFORM
            self.__transform_data_X = dict()

    def __apply_transform(self):
        if self.__transform_param_X == Deuron.TRANSFORM.NORMALIZED:
            maxx = self.__transform_data_X["Max"]
            minn = self.__transform_data_X["Min"]
            self.__X_raw = (self.__X_raw - minn) / (maxx - minn)
        elif self.__transform_param_X == Deuron.TRANSFORM.STANDARDIZED:
            meann = self.__transform_data_X["Mean"]
            stdd = self.__transform_data_X["Std"]
            self.__X_raw = (self.__X_raw - meann) / stdd
        elif self.__transform_param_X == Deuron.TRANSFORM.MEAN_NORMALIZED:
            maxx = self.__transform_data_X["Max"]
            minn = self.__transform_data_X["Min"]
            meann = self.__transform_data_X["Mean"]
            self.__X_raw = (self.__X_raw - meann) / (maxx - minn)

        if self.__transform_param_Y == Deuron.TRANSFORM.NORMALIZED:
            maxx = self.__transform_data_Y["Max"]
            minn = self.__transform_data_Y["Min"]
            self.__Y_raw = (self.__Y_raw - minn) / (maxx - minn)
        elif self.__transform_param_Y == Deuron.TRANSFORM.STANDARDIZED:
            meann = self.__transform_data_Y["Mean"]
            stdd = self.__transform_data_Y["Std"]
            self.__Y_raw = (self.__Y_raw - meann) / stdd
        elif self.__transform_param_Y == Deuron.TRANSFORM.MEAN_NORMALIZED:
            maxx = self.__transform_data_Y["Max"]
            minn = self.__transform_data_Y["Min"]
            meann = self.__transform_data_Y["Mean"]
            self.__Y_raw = (self.__Y_raw - meann) / (maxx - minn)

    def __reset_applied_transform(self):
        if self.__transform_param_X != Deuron.TRANSFORM.NO_TRANSFORM:
            self.__X_raw = self.redo_transform(self.__X_raw)
        if self.__transform_param_Y != Deuron.TRANSFORM.NO_TRANSFORM:
            self.__Y_raw = self.redo_transform(self.__Y_raw, True)

    def transform(self, data: np.array, is_Y_transform = False):
        """
        Transform target matrix by the rules of X or Y (by selection)

        data --> The target matrix to transform (numpy.array mandatory)
        is_Y_transform (Optional [Default: False]) --> if False apply transform rules of X, if True apply transform rules of Y
        """

        if is_Y_transform:
            if self.__transform_param_Y == Deuron.TRANSFORM.NORMALIZED:
                maxx = self.__transform_data_Y["Max"]
                minn = self.__transform_data_Y["Min"]
                return (data - minn) / (maxx - minn)
            elif self.__transform_param_Y == Deuron.TRANSFORM.STANDARDIZED:
                meann = self.__transform_data_Y["Mean"]
                stdd = self.__transform_data_Y["Std"]
                return (data - meann) / stdd
            elif self.__transform_param_Y == Deuron.TRANSFORM.MEAN_NORMALIZED:
                maxx = self.__transform_data_Y["Max"]
                minn = self.__transform_data_Y["Min"]
                meann = self.__transform_param_Y["Mean"]
                return (data - meann) / (maxx - minn)
            else:
                raise Exception("Error! Y value not transformed before")
        else:
            if self.__transform_param_X == Deuron.TRANSFORM.NORMALIZED:
                maxx = self.__transform_data_X["Max"]
                minn = self.__transform_data_X["Min"]
                return (data - minn) / (maxx - minn)
            elif self.__transform_param_X == Deuron.TRANSFORM.STANDARDIZED:
                meann = self.__transform_data_X["Mean"]
                stdd = self.__transform_data_X["Std"]
                return (data - meann) / stdd
            elif self.__transform_param_X == Deuron.TRANSFORM.MEAN_NORMALIZED:
                maxx = self.__transform_data_X["Max"]
                minn = self.__transform_data_X["Min"]
                meann = self.__transform_data_X["Mean"]
                return (data - meann) / (maxx - minn)
            else:
                raise Exception("Error! X value not transformed before")

    def redo_transform(self, data: np.array, is_Y_transform = False):
        """
        Redo the transform of the target matrix by the rules of X or Y (by selection)

        data --> The target matrix to redo its transformed form (numpy.array mandatory)
        is_Y_transform (Optional [Default: False]) --> if False apply transform rules of X, if True apply transform rules of Y
        """

        if is_Y_transform:
            if self.__transform_param_Y == Deuron.TRANSFORM.NORMALIZED:
                maxx = self.__transform_data_Y["Max"]
                minn = self.__transform_data_Y["Min"]
                return data * (maxx - minn) + minn
            elif self.__transform_param_Y == Deuron.TRANSFORM.STANDARDIZED:
                meann = self.__transform_data_Y["Mean"]
                stdd = self.__transform_data_Y["Std"]
                return data * stdd + meann
            elif self.__transform_param_Y == Deuron.TRANSFORM.MEAN_NORMALIZED:
                maxx = self.__transform_data_Y["Max"]
                minn = self.__transform_data_Y["Min"]
                meann = self.__transform_data_Y["Mean"]
                return data * (maxx - minn) + meann
            else:
                raise Exception("Error! Y value not transformed before")
        else:
            if self.__transform_param_X == Deuron.TRANSFORM.NORMALIZED:
                maxx = self.__transform_data_X["Max"]
                minn = self.__transform_data_X["Min"]
                return data * (maxx - minn) + minn
            elif self.__transform_param_X == Deuron.TRANSFORM.STANDARDIZED:
                meann = self.__transform_data_X["Mean"]
                stdd = self.__transform_data_X["Std"]
                return data * stdd + meann
            elif self.__transform_param_X == Deuron.TRANSFORM.MEAN_NORMALIZED:
                maxx = self.__transform_data_X["Max"]
                minn = self.__transform_data_X["Min"]
                meann = self.__transform_data_X["Mean"]
                return data * (maxx - minn) + meann
            else:
                raise Exception("Error! X value not transformed before")

    def set_regularization(self, regularization_param: str, regularization_factor: float = 0):
        """"
        Sets L1, L2 or Dropout regularization by choice for network

        regularization_param --> Specifies the type of regularization, please check Deuron.REGULARIZATION
        regularization_factor (Optional) --> Specifies the 2nd special parameter for regularization techniques
        If its empty, default values been applied; (L1,L2 --> lambda = 0.01) , (Dropout --> keep probability = 0.8)
        """

        if regularization_param in Deuron.REGULARIZATION.values:
            if regularization_param == Deuron.REGULARIZATION.L1 or regularization_param == Deuron.REGULARIZATION.L2:
                if regularization_factor >= 0:
                    self.__regularization_param = copy.deepcopy(regularization_param)
                    self.__keep_prob = 0
                    if regularization_factor == 0:
                        self.__lambd = 0.01
                    else:
                        self.__lambd = copy.deepcopy(regularization_factor)
                else:
                    raise Exception("Error! Invalid value for lambda, L1 and L2 regularization factor lambda must be in interval, (0 < lambda)")
            elif regularization_param == Deuron.REGULARIZATION.DROPOUT:
                if regularization_factor >= 0 and regularization_factor < 1:
                    self.__regularization_param = Deuron.REGULARIZATION.DROPOUT
                    self.__lambd = 0
                    if regularization_factor == 0:
                        self.__keep_prob = 0.8
                    else:
                        self.__keep_prob = copy.deepcopy(regularization_factor)
                else:
                    raise Exception("Error! Invalid value for keep probability, Dropout regularization factor must be in interval, (0 < keep_prob < 1)")
            else:
                self.cancel_regularization()
        else:
            raise Exception("Error! Invalid regularization parameter, (Please check Deuron.REGULARIZATION)")

    def cancel_regularization(self):
        """
        Cancels the regularization settings for the network
        """

        self.__regularization_param = Deuron.REGULARIZATION.NO_REGULARIZATION
        self.__lambd = 0
        self.__keep_prob = 0

    def learning_decay(self, status: bool, decay_rate = 0.95, time_interval = 100):
        if time_interval < 0 and decay_rate < 0:
            raise Exception("Error! Invalid parameters for decay rate or time interval, (both must be > 0)")
        if status:
            self.__alpha_decay_rate = copy.deepcopy(decay_rate)
            self.__learning_decay_time_interval = copy.deepcopy(time_interval)
        else:
            self.__alpha_decay_rate = 0
            self.__learning_decay_time_interval = 100

    def set_optimizer(self, optimizer: str, beta1 = 0.9, beta2 = 0.99):
        """
        Set an optimization algotihm such as momentum or adam optimization algorithms for learning

        optimizer --> Specifies the optimizer type, (Please check Deuron.OPTIMIZER...)
        beta1 --> Common hyperparameter for momentum and adam algorithm (Default value: 0.9)
        beta2 --> Special additional hyperparameter for adam algorithm (Default value: 0.99)
        """

        if optimizer in Deuron.OPTIMIZER.values:
            if beta1 > 0 and beta2 > 0:
                if optimizer == Deuron.OPTIMIZER.NO_OPTIMIZER:
                    self.cancel_optimizer()
                elif optimizer == Deuron.OPTIMIZER.MOMENTUM:
                    self.__beta_1 = copy.deepcopy(beta1)
                    self.__initialize_momentum_velocity()
                else:
                    self.__beta_1 = copy.deepcopy(beta1)
                    self.__beta_2 = copy.deepcopy(beta2)
                    self.__initialize_adam_optimizer()
                self.__optimizer = optimizer
            else:
                raise Exception("Error! beta1 and beta2 values must be greater than 0")
        else:
            raise Exception("Error! Invalid optimizer parameter, please check (Deuron.OPTIMIZER...)")

    def cancel_optimizer(self):
        """
        Cancels the optimization methods for learning algorithm and resets all previous beta hyperparameters
        """

        self.__optimizer = Deuron.OPTIMIZER.NO_OPTIMIZER
        del self.__beta_1
        del self.__beta_2

    def __initialize_momentum_velocity(self):
        self.__velocity_data = dict()
        for i in range(1, self.__L):
            self.__velocity_data["dW"+str(i)] = np.zeros(self.__W[str(i)].shape)
            self.__velocity_data["dB"+str(i)] = np.zeros(self.__B[str(i)].shape)

    def __initialize_adam_optimizer(self):
        self.__initialize_momentum_velocity()
        self.__adam_data = dict()
        for i in range(1, self.__L):
            self.__adam_data["dW"+str(i)] = np.zeros(self.__W[str(i)].shape)
            self.__adam_data["dB"+str(i)] = np.zeros(self.__B[str(i)].shape)

    def __forward_prop(self):
        for i in range(1,self.__L):
            self.__Z[str(i)] = np.dot(self.__W[str(i)].T, self.__A[str(i - 1)]) + self.__B[str(i)]
            self.__A[str(i)] = self.__activation(self.__Z[str(i)], self.__layer_type_list[i])
        if self.__regularization_param == self.REGULARIZATION.DROPOUT:
            for j in range(1,self.__L - 1):
                D = (np.random.rand(self.__A[str(j)].shape[0], self.__A[str(j)].shape[1]) < self.__keep_prob).astype(int)
                self.__A[str(j)] *= D
                self.__A[str(j)] /= self.__keep_prob
                self.__dropout_cache['D'+str(j)] = D

    def __activation(self, z, type):
        if type == self.ACTIVATION.RELU:
            a = copy.deepcopy(z)
            a[a < 0] = 0
        elif type == self.ACTIVATION.LEAKY_RELU:
            a = copy.deepcopy(z)
            a[a < 0] *= 0.1
        elif type == self.ACTIVATION.SIGMOID:
            return 1/(1 + np.exp(-z))
        elif type == self.ACTIVATION.TANH:
            return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif type == self.ACTIVATION.SOFTMAX:
            return np.exp(z) / np.sum(np.exp(z), axis = 0, keepdims = True)
        else:
            raise Exception("Error!!! Types must be one of the Deuron.ACTIVATION... (RELU, SIGMOID, etc.)")
        return a

    def __cost(self):
        a = self.__A[str(self.__L - 1)]
        if self.__cost_type == self.COST.SQUARE_ERROR_COST:
            return ((1/(2*self.__m_current)) * np.sum((self.__Y - a)**2)) + ((self.__lambd/(2*self.__m_current)) * self.__reg_value())
        elif self.__cost_type == self.COST.EXTENDED_CROSS_ENTROPY_COST:
            return (-1/self.__m_current) * np.sum(self.__Y * np.log(a) + (1 - self.__Y) * np.log(1 - a)) + ((self.__lambd/(2*self.__m_current)) * self.__reg_value())
        elif self.__cost_type == self.COST.CROSS_ENTROPY_COST:
            return (-1/self.__m_current) * np.sum(self.__Y * np.log(a)) + ((self.__lambd/(2*self.__m_current)) * self.__reg_value())

    def __reg_value(self):
        #Applies L1 or L2 regularization for every weight (W) value of the network
        reg = 0
        if self.__regularization_param == self.REGULARIZATION.L1:
            for i in range(1,self.__L):
                reg += np.sum(np.abs(self.__W[str(i)]))
        elif self.__regularization_param == self.REGULARIZATION.L2:
            for i in range(1,self.__L):
                reg += np.sum(self.__W[str(i)]**2)
        return reg

    def __backward_prop(self):
        #Calculate all derivatives before the gradient descent
        self.__dA[str(self.__L - 1)] = self.__derivative_cost()
        for i in range(self.__L - 1, 0, -1):
            self.__dZ[str(i)] = self.__dA[str(i)] * self.__derivative_activation(i)
            self.__dW[str(i)] = (1 / self.__m_current) * (np.dot(self.__A[str(i - 1)], self.__dZ[str(i)].T))
            self.__backward_regularization(i)
            self.__dB[str(i)] = (1 / self.__m_current) * np.sum(self.__dZ[str(i)],axis = 1, keepdims = True)
            if i != 1:
                self.__dA[str(i - 1)] = np.dot(self.__W[str(i)], self.__dZ[str(i)])
                if self.__regularization_param == self.REGULARIZATION.DROPOUT:
                    self.__dA[str(i - 1)] *= self.__dropout_cache['D'+str(i - 1)]
                    self.__dA[str(i - 1)] /= self.__keep_prob

    def __derivative_cost(self):
        a = self.__A[str(self.__L - 1)]
        if self.__cost_type == self.COST.SQUARE_ERROR_COST:
            return (a - self.__Y)
        elif self.__cost_type == self.COST.EXTENDED_CROSS_ENTROPY_COST:
            return (a - self.__Y) / (a * (1 - a))
        elif self.__cost_type == self.COST.CROSS_ENTROPY_COST:
            return -(self.__Y / a)

    def __derivative_activation(self, layer_number):
        A = self.__A[str(layer_number)]
        a = A
        type = self.__layer_type_list[layer_number]
        if type == self.ACTIVATION.RELU:
            a = copy.deepcopy(A)
            a[a > 0] = 1
        elif type == self.ACTIVATION.LEAKY_RELU:
            a = copy.deepcopy(A)
            a[a < 0] = 0.1
            a[a > 0] = 1
        elif type == self.ACTIVATION.SIGMOID:
            return a * (1 - a)
        elif type == self.ACTIVATION.TANH:
            return 1 - a**2
        elif type == self.ACTIVATION.SOFTMAX:
            return a * (1 - a)
        return a

    def __backward_regularization(self, layer_number):
        if self.__regularization_param == self.REGULARIZATION.L1:
            W = copy.deepcopy(self.__W[str(layer_number)])
            W[W > 0] = 1
            W[W < 0] = -1
            self.__dW[str(layer_number)] += (self.__lambd * W)
        elif self.__regularization_param == self.REGULARIZATION.L2:
            self.__dW[str(layer_number)] += (self.__lambd * 2 * self.__W[str(layer_number)])

    def __gradient_desc(self, iter_number):
        alpha = self.__alpha
        if self.__alpha_decay_rate != 0:
            alpha = (self.__alpha / (1 + self.__alpha_decay_rate * np.floor(iter_number / self.__learning_decay_time_interval)))

        if self.__optimizer == Deuron.OPTIMIZER.NO_OPTIMIZER:
            for i in range(1,self.__L):
                self.__W[str(i)] -= alpha * self.__dW[str(i)]
                self.__B[str(i)] -= alpha * self.__dB[str(i)]
        elif self.__optimizer == Deuron.OPTIMIZER.ADAM:
            for i in range(1, self.__L):
                self.__velocity_data['dW'+str(i)] = self.__beta_1 * self.__velocity_data['dW'+str(i)] + (1 - self.__beta_1) * self.__dW[str(i)]
                self.__velocity_data['dB'+str(i)] = self.__beta_1 * self.__velocity_data['dB'+str(i)] + (1 - self.__beta_1) * self.__dB[str(i)]
                adam_counter = iter_number + 1
                velocity_data_corrected_dW = self.__velocity_data["dW"+str(i)] / (1 - self.__beta_1**adam_counter)
                velocity_data_corrected_dB = self.__velocity_data["dB"+str(i)] / (1 - self.__beta_1**adam_counter)
                self.__adam_data["dW"+str(i)] = self.__beta_2 * self.__adam_data["dW"+str(i)] + (1 - self.__beta_2) * self.__dW[str(i)]**2
                self.__adam_data["dB"+str(i)] = self.__beta_2 * self.__adam_data["dB"+str(i)] + (1 - self.__beta_2) * self.__dB[str(i)]**2
                adam_data_corrected_dW = self.__adam_data["dW"+str(i)] / (1 - self.__beta_2**adam_counter)
                adam_data_corrected_dB = self.__adam_data["dB"+str(i)] / (1 - self.__beta_2**adam_counter)
                self.__W[str(i)] = self.__W[str(i)] - alpha * (velocity_data_corrected_dW / ((adam_data_corrected_dW)**0.5 + self.__epsilon))
                self.__B[str(i)] = self.__B[str(i)] - alpha * (velocity_data_corrected_dB / ((adam_data_corrected_dB)**0.5 + self.__epsilon))
        else:
            for i in range(1, self.__L):
                self.__velocity_data['dW'+str(i)] = self.__beta_1 * self.__velocity_data['dW'+str(i)] + (1 - self.__beta_1) * self.__dW[str(i)]
                self.__velocity_data['dB'+str(i)] = self.__beta_1 * self.__velocity_data['dB'+str(i)] + (1 - self.__beta_1) * self.__dB[str(i)]
                self.__W[str(i)] = self.__W[str(i)] - alpha * self.__velocity_data['dW'+str(i)]
                self.__B[str(i)] = self.__B[str(i)] - alpha * self.__velocity_data['dB'+str(i)]

    def __plot_cost(self):
        plt.plot(self.__J, 'r-')
        plt.title('Learning algorithm cost by iteration')
        plt.ylabel('Cost (J)')
        plt.xlabel('Iteration Count #')
        plt.show()

    
    def start(self):
        """
        The start function of Deuron learning algorithm. This function starts learning algorithm, prints and plots cost values of each 100 iteration (epoch). Returns learned weights as a tuple (W,B)
        """

        self.__apply_transform()
        self.__prepare_data_matrices()
        initial_time = time.time()
        total = 0
        for iter in range(self.__epoch_number):
            if self.__batch_count != 1:
                # Shuffle and portition of mini batches for each iteration
                self.__seed += 1
                np.random.seed(self.__seed)
                self.__apply_mini_batch() 
            batch_cost_total = 0
            for batch_index in range(self.__batch_count):
                self.__A["0"] = self.__X_list[batch_index]
                self.__Y = self.__Y_list[batch_index]
                self.__m_current = self.__Y.shape[1]
                self.__forward_prop()
                batch_cost_total += self.__cost()
                self.__backward_prop()
                self.__gradient_desc(iter)   
                self.__progress_bar(iter, self.__epoch_number, batch_index, self.__batch_count, "Total Progress", "Batch Progress", f"Cost (J) : {batch_cost_total / (batch_index + 1)}")
            self.__J[iter] = batch_cost_total / self.__batch_count
        end_time = time.time()
        self.__reset_applied_transform()
        self.__learn_state = True
        self.__plot_cost()
        print(f"\n---> Learning time passed:\t{end_time - initial_time} seconds <---\n")
        return (copy.deepcopy(self.__W), copy.deepcopy(self.__B))

    def get_weights(self):
        """
        Retuns the learned weights as tuple (W,B) after learning algoritm done.

        Deuron.start() function must have been executed first
        """

        if self.__learn_state:
            return (copy.deepcopy(self.__W, copy.deepcopy(self.__B)))
        else:
            raise Exception("Error! There are no learned weights yet, you must initialize Deuron.start() function first")

    def test(self, X: np.array):
        """
        Tests the raw data (needs not trasnformed X, because it transforms inside) and returns the prediction
        (Must complete learning algorithm first)

        X --> test set
        """
        if self.__learn_state:
            if(X.shape[0] == X.size):
                X.reshape(1, X.size)
            if(X.shape[1] == self.__nx):
                X = X.T
            elif(X.shape[0] != self.__nx):
                raise Exception("Error! Test data is inconsistent with the train data (feature count not match, nx)")

            Z_test = dict()
            A_test = dict()
            if self.__transform_param_X != Deuron.TRANSFORM.NO_TRANSFORM:
                A_test["0"] = self.transform(X)
            else:
                A_test["0"] = X

            for i in range(1,self.__L):
                Z_test[str(i)]= np.dot(self.__W[str(i)].T, A_test[str(i - 1)]) + self.__B[str(i)]
                A_test[str(i)] = self.__activation(Z_test[str(i)], self.__layer_type_list[i])

            return A_test[str(self.__L - 1)]
        else:
            raise Exception("Error! Please complete learning algorithm to test a set")

    def print_state(self):
        """
        Prints the current state of the deuron object
        """

        nx, layer_list, cost_type, alpha, learn_state = self.__return_strict_propertities()
        if self.__batch_count == 1:
            batch_size = "Batch Gradient Descent"
        else:
            batch_size = self.__batch_size
        content = f"Deuron Object Current State\n==========\nNx\t{nx}\nLayer list\t{layer_list}\nCost type\t{cost_type}\nLearning rate alpha\t{alpha}\nHas learned\t{learn_state}\n==========\nBatch Size\t{batch_size}\nTransform state for X\t{self.__transform_param_X}\nTransform state for Y\t{self.__transform_param_Y}\n"
        
        content += f"Regularization Type\t{self.__regularization_param}"
        if self.__regularization_param == Deuron.REGULARIZATION.DROPOUT:
            content += f"\tKeep Probability\t{self.__keep_prob}"
        elif self.__regularization_param != Deuron.REGULARIZATION.NO_REGULARIZATION:
            content += f"\tLambda\t{self.__lambd}"

        content += f"\nOptimizer\t{self.__optimizer}"
        if self.__optimizer == Deuron.OPTIMIZER.MOMENTUM:
            content += f"\tBeta 1\t{self.__beta_1}"
        elif self.__optimizer == Deuron.OPTIMIZER.ADAM:
            content += f"\tBeta 1\t{self.__beta_1}\tBeta 2\t{self.__beta_2}"

        content += f"\nLearning Decay\t{self.__alpha_decay_rate != 0}"
        if self.__alpha_decay_rate != 0:
            content += f"\tDecay Rate\t{self.__alpha_decay_rate}\tTime Interval\t{self.__learning_decay_time_interval}"    

        print(content)
        
    def save_state(self):
        """
        (Note: It will be improved in the future [Currently in alpha version function])

        (More flexible propertities such as changing the structure after load state or test before training for recorded states will be added soon)
        
        It saves the current state of the deuron object as zip file that includes parameter datas as W.npy, B.npy and it also includes the state file that has the extension of .deuronstate to record the more detailed
        information about its current state (Exp. layer_list, cost_type, etc.)
        """

        if(input("Are you sure want to save Deuron Object state ? (Yes: Y, No: [Any Key])") == "Y"):
            state_name = input("Please enter a state name")
            with open('W.npy', 'wb') as f:
                np.save(f, self.__W)
            with open('B.npy', 'wb') as f:
                np.save(f, self.__B)

            nx, layer_list, cost_type, alpha, _ = self.__return_strict_propertities()
            content = f"{nx}\n{layer_list}\n{cost_type}\n{alpha}\n"

            if self.__batch_count != 1:
                content += f"{self.__batch_size}\n"
            else:
                content += "0\n"

            if self.__regularization_param == Deuron.REGULARIZATION.L1 or self.__regularization_param == Deuron.REGULARIZATION.L2:
                content += f"{self.__regularization_param}\n{self.__lambd}\n"
            elif self.__regularization_param == Deuron.REGULARIZATION.DROPOUT:
                content += f"{self.__regularization_param}\n{self.__keep_prob}\n"
            else:
                content += f"{self.__regularization_param}\n"

            if self.__optimizer == Deuron.OPTIMIZER.MOMENTUM:
                content += f"{self.__optimizer}\n{self.__beta_1}\n"
            elif self.__optimizer == Deuron.OPTIMIZER.ADAM:
                content += f"{self.__optimizer}\n{self.__beta_2}\n"
            else:
                content += f"{self.__optimizer}\n"

            if self.__alpha_decay_rate != 0:
                content += f"True\n{self.__alpha_decay_rate}\n{self.__learning_decay_time_interval}"
            else:
                content += "False"


            with open(f"{state_name}.deuronstate", "w") as f:
                f.write(content)
                f.flush()
                f.close()

            file_names = ["W.npy", "B.npy", f"{state_name}.deuronstate"]
            with zipfile.ZipFile(f"{state_name}.zip", mode="w") as archive:
                for filename in file_names:
                    archive.write(filename)
            for filename in file_names:
                os.remove(filename)

    def create_with_state(X: np.array, Y: np.array, epoch_number: int):
        """
        (Note: It will be improved in the future [Currently in alpha version function])
        
        (More flexible propertities such as changing the structure after load state or test before training for recorded states will be added soon)

        It creates a new deuron object with new training data pairs X and Y with new epoch number, and with the deuron state that has recorded before

        X --> new trainig input data matrix
        Y --> new training label data matrix
        epoch_number --> new epoch number for new deuron object
        """

        file_name = input("Please enter the name of deuron state zip (enter only name without .zip)")
        file_names = ["W.npy", "B.npy", f"{file_name}.deuronstate"]
        if os.path.exists(f"{file_name}.zip"):
            with zipfile.ZipFile(f"{file_name}.zip", 'r') as zipState:
                zipState.extractall()
            W = np.load('W.npy', allow_pickle = True)
            B = np.load('B.npy', allow_pickle = True)
            W = W.item()
            B = B.item()
            with open(f"{file_name}.deuronstate", "r") as statefile:
                content = statefile.read().split("\n")
            
            nx = int(content[0])
            layer_list = eval(content[1])
            cost_type = content[2]
            alpha = float(content[3])
            batch_size = int(content[4])
            regularization_param = content[5]
            if regularization_param == Deuron.REGULARIZATION.NO_REGULARIZATION:
                index = 6
            else:
                regularization_sub_value = float(content[6])
                index = 7
            optimizer = content[index]
            index += 1
            if optimizer == Deuron.OPTIMIZER.MOMENTUM:
                beta_1 = float(content[index])
                index += 1
            elif optimizer == Deuron.OPTIMIZER.ADAM:
                beta_1 = float(content[index])
                beta_2 = float(content[index + 1])
                index += 2
            learning_decay_state = bool(content[index])
            index += 1
            if learning_decay_state:
                learning_decay_rate = float(content[index])
                learning_decay_time_interval = int(content[index + 1])

            for item in file_names:
                os.remove(item)
            if X.shape[0] == nx or X.shape[1] == nx:
                obj = Deuron(X, Y, layer_list, cost_type, epoch_number, alpha, True, (W,B))
                if batch_size != 0:
                    obj.set_mini_batch(batch_size)
                if regularization_param != Deuron.REGULARIZATION.NO_REGULARIZATION:
                    obj.set_regularization(regularization_param, regularization_sub_value)
                if optimizer == Deuron.OPTIMIZER.MOMENTUM:
                    obj.set_optimizer(optimizer, beta_1)
                elif optimizer == Deuron.OPTIMIZER.ADAM:
                    obj.set_optimizer(optimizer, beta_1, beta_2)
                if learning_decay_state:
                    obj.learning_decay(True, learning_decay_rate, learning_decay_time_interval)

                return obj
            else:
                raise Exception("Invalid X train set matrix for deuron state!")
        else:
            raise Exception("Invalid file name or path, could not found deuron state file")


    def __return_strict_propertities(self):
        nx = self.__nx
        layer_list = list()
        for i in range(1, self.__L):
            layer_list.append((self.__layer_number_of_nodes_list[i], self.__layer_type_list[i]))
        cost_type = self.__cost_type
        alpha = self.__alpha
        learn_state = self.__learn_state
        return [nx, layer_list, cost_type, alpha, learn_state]

    def __progress_bar(self, current_iter1, total_iter1, current_iter2, total_iter2, description1 = "", description2 = "", additional_info = ""):
        progress1 = (current_iter1 / total_iter1)*100
        sign_count1 = int(progress1 / 5)
        sign_str1 = "="*sign_count1
        space_str1 = " "*(20 - sign_count1)
        progress2 = (current_iter2 / total_iter2)*100
        sign_count2 = int(progress2 / 5)
        sign_str2 = "="*sign_count2
        space_str2 = " "*(20 - sign_count2)
        print(f"\r{description1}\t[{current_iter1}/{total_iter1}]\t\t[{sign_str1}{space_str1}]\t{'{:.2f}'.format(progress1)}%\t\t|  {description2}\t[{current_iter2}/{total_iter2}]\t\t[{sign_str2}{space_str2}]\t{'{:.2f}'.format(progress2)}%\t\t|  {additional_info}", end = "")
