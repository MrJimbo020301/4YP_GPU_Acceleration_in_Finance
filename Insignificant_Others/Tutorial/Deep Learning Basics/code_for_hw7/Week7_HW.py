import numpy as np
import modules_disp as disp
from expected_results import * 
import matplotlib.pyplot as plt


class Module:
    def sgd_step(self, lrate): pass # For modules without weights

# Linear Modules 
# Each linear module has a forward method that takes in a batch of activate A (from the previous layer)
# and returns a batch of pre-activations Z. 
# 
# Each linear module has a backward method that takes in dLdZ and returns dLdA. 
# This module also computes and stores dLdW and dLdW0, the gradients with respect to the weights. 

class Linear(Module): 
    def __init__(self, m, n):
        self.m, self.n = (m, n) # (in size, out size)
        self.W0 = np.zeros([self.n,1]) # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-0.5), [m, n]) # (m x n) 

    def forward(self, A):
        self.A = A # (m x b) Hint: make sure you understand what b stands for
        return np.dot(self.W.T, A) + self.W0 # Your code (n x b)
    
    def backward(self, dLdZ): # dLdZ is size (n x b), uses stored self.A
        self.dLdW = np.dot(self.A, dLdZ.T) # *(m x n)
        self.dLdW0 = dLdZ.sum(axis=1).reshape((self.n, 1)) # (n x 1)
        return np.dot(self.W, dLdZ) # (m x b)
    
    def sgd_step(self, lrate): # Gradient descent step
        self.W = self.W - lrate*self.dLdW
        self.W0 = self.W0 - lrate*self.dLdW0

# Activation modules 
# Each activation module has a forward method that takes in a batch of 
# pre-activations Z and returns a batch of activations A.

# Each activation module has a backward method that takes in dLdA and returns dLdZ.
# With the exception of Softmax, each activation module has no parameters, where we assume
# dLdZ is passed in. 

class Tanh(Module):
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        return dLdA * (1 - self.A**2)
    
class ReLU(Module):
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        return dLdA * np.where(self.A > 0, 1, 0)

class Softmax(Module):
    def forward(self, Z):
        self.A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return self.A

    def backward(self, dLdZ):
        return dLdZ

    def class_fun(self, Z):
        return np.argmax(self.A, axis=0)

# Loss modules
# Each loss module has a forward method that takes in a batch of 
# predictions Ypred (from the previous layer) and labels Y and returns 
# a scalar loss value. 

# The NLL module has a backward method that returns dLdZ, the gradient 
# with respect to the preactivation to Softmax.(note: not the activation)
# Since we are always pairing SoftMax activation with NLL loss

class NLL(Module):
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return float(np.sum(-Y * np.log(Ypred)))


    def backward(self):
        return self.Ypred - self.Y
    
# Sequential module
class Sequential: 
    def __init__(self, modules, loss): 
        self.modules = modules
        self.loss = loss
    def sgd(self, X, Y, iters=100, lrate=0.005): 
        D, N = X.shape
        sum_loss = 0
        for it in range(iters):
            i = np.random.randint(N)
            Xt = X[:, i:i+1]
            Yt = Y[:, i:i+1]
            Ypred = self.forward(Xt)
            sum_loss += self.loss.forward(Ypred, Yt)
            err = self.loss.backward()
            self.backward(err)
            self.sgd_step(lrate)
    
    def forward(self, Xt): 
        for m in self.modules: 
            Xt = m.forward(Xt)
        return Xt
    
    def backward(self, delta):
        for m in self.modules[::-1]: 
            delta = m.backward(delta)
    def sgd_step(self, lrate):
        for m in self.modules:
            m.sgd_step(lrate)
    
    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print("Iteration = ", it, "  Accuracy = ", acc, "  Loss = ", cur_loss)


######################################################################
#   Data Sets
######################################################################

def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)


def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, 0, 0]])
    return X, for_softmax(y)


def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    return X, for_softmax(y)


def hard():
    X = np.array([[-0.23390341, 1.18151883, -2.46493986, 1.55322202, 1.27621763,
                   2.39710997, -1.3440304, -0.46903436, -0.64673502, -1.44029872,
                   -1.37537243, 1.05994811, -0.93311512, 1.02735575, -0.84138778,
                   -2.22585412, -0.42591102, 1.03561105, 0.91125595, -2.26550369],
                  [-0.92254932, -1.1030963, -2.41956036, -1.15509002, -1.04805327,
                   0.08717325, 0.8184725, -0.75171045, 0.60664705, 0.80410947,
                   -0.11600488, 1.03747218, -0.67210575, 0.99944446, -0.65559838,
                   -0.40744784, -0.58367642, 1.0597278, -0.95991874, -1.41720255]])
    y = np.array([[1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1.,
                   1., 0., 0., 0., 1., 1., 0.]])
    return X, for_softmax(y)


def for_softmax(y):
    return np.vstack([1 - y, y])


######################################################################
# Tests
######################################################################

def unit_test(name, expected, actual):
    if actual is None:
        print(name + ": unimplemented")
    elif np.allclose(expected, actual):
        print(name + ": OK")
    else:
        print(name + ": FAILED")
        print("expected: " + str(expected))
        print("but was: " + str(actual))


def sgd_test(nn, test_values):
    """Run one step of SGD on a simple dataset with the specified
    network, and with batch size (b) = len(dataset)

    :param nn: A "Sequential" object representing a neural network

    :param test_values: A dictionary containing the expected values
    for the necessary unit tests

    """
    lrate = 0.005
    # data
    X, Y = super_simple_separable()

    # define the modules
    assert len(nn.modules) == 4
    (linear_1, f_1, linear_2, f_2) = nn.modules
    Loss = nn.loss

    unit_test('linear_1.W', test_values['linear_1.W'], linear_1.W)
    unit_test('linear_1.W0', test_values['linear_1.W0'], linear_1.W0)
    unit_test('linear_2.W', test_values['linear_2.W'], linear_2.W)
    unit_test('linear_2.W0', test_values['linear_2.W0'], linear_2.W0)

    z_1 = linear_1.forward(X)
    unit_test('z_1', test_values['z_1'], z_1)
    a_1 = f_1.forward(z_1)
    unit_test('a_1', test_values['a_1'], a_1)
    z_2 = linear_2.forward(a_1)
    unit_test('z_2', test_values['z_2'], z_2)
    a_2 = f_2.forward(z_2)
    unit_test('a_2', test_values['a_2'], a_2)

    Ypred = a_2
    loss = Loss.forward(Ypred, Y)
    unit_test('loss', test_values['loss'], loss)
    dloss = Loss.backward()
    unit_test('dloss', test_values['dloss'], dloss)

    dL_dz2 = f_2.backward(dloss)
    unit_test('dL_dz2', test_values['dL_dz2'], dL_dz2)
    dL_da1 = linear_2.backward(dL_dz2)
    unit_test('dL_da1', test_values['dL_da1'], dL_da1)
    dL_dz1 = f_1.backward(dL_da1)
    unit_test('dL_dz1', test_values['dL_dz1'], dL_dz1)
    dL_dX = linear_1.backward(dL_dz1)
    unit_test('dL_dX', test_values['dL_dX'], dL_dX)

    linear_1.sgd_step(lrate)
    unit_test('updated_linear_1.W', test_values['updated_linear_1.W'], linear_1.W)
    unit_test('updated_linear_1.W0', test_values['updated_linear_1.W0'], linear_1.W0)
    linear_2.sgd_step(lrate)
    unit_test('updated_linear_2.W', test_values['updated_linear_2.W'], linear_2.W)
    unit_test('updated_linear_2.W0', test_values['updated_linear_2.W0'], linear_2.W0)

    
np.random.seed(0)
X, Y = super_simple_separable()  # data
linear_1 = Linear(2, 3)  # module
learning_rate = 0.005  #hyperparameter

z_1 = linear_1.forward(X)
exp_z_1 =  np.array([[10.41750064, 6.91122168, 20.73366505, 22.8912344],
                     [7.16872235, 3.48998746, 10.46996239, 9.9982611],
                     [-2.07105455, 0.69413716, 2.08241149, 4.84966811]])
unit_test("linear_forward", exp_z_1, z_1)

dL_dz1 = np.array([[1.69467553e-09, -1.33530535e-06, 0.00000000e+00, -0.00000000e+00],
                                     [-5.24547376e-07, 5.82459519e-04, -3.84805202e-10, 1.47943038e-09],
                                     [-3.47063705e-02, 2.55611604e-01, -1.83538094e-02, 1.11838432e-04]])
exp_dLdX = np.array([[-2.40194628e-02, 1.77064845e-01, -1.27021626e-02, 7.74006953e-05],
                                    [2.39827939e-02, -1.75870737e-01, 1.26832126e-02, -7.72828555e-05]])
dLdX = linear_1.backward(dL_dz1)
unit_test("linear_backward", exp_dLdX, dLdX)

linear_1.sgd_step(learning_rate)
exp_linear_1_W = np.array([[1.2473734,  0.28294514,  0.68940437],
                           [1.58455079, 1.32055711, -0.69218045]]),
unit_test("linear_sgd_step_W",  exp_linear_1_W,  linear_1.W)

exp_linear_1_W0 = np.array([[6.66805339e-09],
                            [-2.90968033e-06],
                            [-1.01331631e-03]]),
unit_test("linear_sgd_step_W0", exp_linear_1_W0, linear_1.W0)

np.random.seed(0)
sgd_test(Sequential([Linear(2,3), Tanh(), Linear(3,2), Softmax()], NLL()), test_1_values)

np.random.seed(0)
sgd_test(Sequential([Linear(2,3), ReLU(), Linear(3,2), Softmax()], NLL()), test_2_values)

X, Y = hard()
nn = Sequential([Linear(2, 10), ReLU(), Linear(10, 10), ReLU(), Linear(10,2), Softmax()], NLL())
disp.classify(X, Y, nn, it=100000)


# TEST 4: try calling these methods that train with a simple dataset
def nn_tanh_test():
    np.random.seed(0)
    nn = Sequential([Linear(2, 3), Tanh(), Linear(3, 2), Softmax()], NLL())
    X, Y = super_simple_separable()
    nn.sgd(X, Y, iters=1, lrate=0.005)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]


def nn_relu_test():
    np.random.seed(0)
    nn = Sequential([Linear(2, 3), ReLU(), Linear(3, 2), Softmax()], NLL())
    X, Y = super_simple_separable()
    nn.sgd(X, Y, iters=2, lrate=0.005)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]


def nn_pred_test():
    np.random.seed(0)
    nn = Sequential([Linear(2, 3), ReLU(), Linear(3, 2), Softmax()], NLL())
    X, Y = super_simple_separable()
    nn.sgd(X, Y, iters=1, lrate=0.005)
    Ypred = nn.forward(X)
    return nn.modules[-1].class_fun(Ypred).tolist(), [nn.loss.forward(Ypred, Y)]

print(nn_tanh_test())

print(nn_relu_test())

print(nn_pred_test())