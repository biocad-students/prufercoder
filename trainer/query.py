import random

import matplotlib.pyplot as plt
import numpy as np
import lasagne
import theano
import theano.tensor as T
from builtins import range
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh
from scipy.stats import norm

def sample_noise(M):
    z = np.float32(np.linspace(-5.0, 5.0, M) + np.random.random(M) * 0.01)
    return z


def plot_d_boundary(discriminate):
    # p_data
    xs = np.linspace(-5, 5, 1000)
    plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')
    # decision boundary
    r = 1000  # resolution (number of points)
    xs = np.float32(np.linspace(-5, 5, r))
    # process multiple points in parallel in a minibatch
    ds = discriminate(xs.reshape(r, 1))

    plt.plot(xs, ds, label='decision boundary')
    plt.show()


def plot_fig(generate, discriminate):
    # plots pg, pdata, decision boundary
    xs = np.linspace(-5, 5, 1000)
    plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')
    # decision boundary
    r = 5000  # resolution (number of points)
    xs = np.float32(np.linspace(-5, 5, r))
    # process multiple points in parallel in same minibatch
    ds = discriminate(xs.reshape(r, 1))
    plt.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs = sample_noise(r)
    gs = generate(zs.reshape(r, 1))
    plt.hist(gs, bins=10, normed=True)



def prepare(vector):
    res = []
    if len(vector) == 1:
        for i in range(49):
            res.append(-1)
        res.append(vector[0])
        for i in range(50):
            res.append(-1)
        return res
    n = int(len(vector) / 2 - 1)
    for i in range(49):
        if i < n:
            res.append(vector[i])
        else:
            res.append(-1)
    for i in range(51):
        if i < n + 2:
            res.append(vector[n + i])
        else:
            res.append(-1)
    return res

def getData():
    return [list(map(int, x.split())) for x in open("../data/modif100#1500.txt", "r").read().split('\n')]


if __name__ == '__main__':
    dirtyData = getData()
    while [] in dirtyData:
        dirtyData.remove([])
    data = list(map(prepare, dirtyData))

    mu = -2
    sigma = 0.3
    M = 100

    # generator
    G_input = T.matrix('Gx')
    length = 5
    size = 100
    G_l1 = lasagne.layers.InputLayer((None, 1), G_input)
    for i in range(length):
        G_l1 = lasagne.layers.DenseLayer(G_l1, size, nonlinearity=rectify)
    G_l4 = lasagne.layers.DenseLayer(G_l1, 1, nonlinearity=linear)
    G = G_l4

    G_out = lasagne.layers.get_output(G)

    # discriminators
    D1_input = T.matrix('D1x')
    D1_target = T.matrix('D1y')

    D1_l1 = lasagne.layers.InputLayer((None, 1), D1_input)
    D2_l1 = lasagne.layers.InputLayer((None, 1), G_out)
    for i in range(length):
        D1_l1 = lasagne.layers.DenseLayer(D1_l1, size, nonlinearity=tanh)
        D2_l1 = lasagne.layers.DenseLayer(D2_l1, size, nonlinearity=tanh, W=D1_l1.W, b=D1_l1.b)

    D1 = lasagne.layers.DenseLayer(D1_l1, 1, nonlinearity=sigmoid)
    D2 = lasagne.layers.DenseLayer(D2_l1, 1, nonlinearity=sigmoid, W=D1.W, b=D1.b)

    D1_out = lasagne.layers.get_output(D1)
    D2_out = lasagne.layers.get_output(D2)

    # output functions
    discriminate = theano.function([D1_input], D1_out)
    generate = theano.function([G_input], G_out)

    plot_d_boundary(discriminate)

    G_obj = (T.log(D2_out)).mean()
    D_obj = (T.log(D1_out) + T.log(1 - D2_out)).mean()

    # parameters update and training
    G_params = lasagne.layers.get_all_params(G, trainable=True)
    G_lr = theano.shared(np.array(0.01, dtype=theano.config.floatX))
    G_updates = lasagne.updates.nesterov_momentum(1 - G_obj, G_params, learning_rate=G_lr, momentum=0.6)
    G_train = theano.function([G_input], G_obj, updates=G_updates)

    D_params = lasagne.layers.get_all_params(D1, trainable=True)
    D_lr = theano.shared(np.array(0.1, dtype=theano.config.floatX))
    D_updates = lasagne.updates.nesterov_momentum(1 - D_obj, D_params, learning_rate=D_lr, momentum=0.6)
    D_train = theano.function([G_input, D1_input], D_obj, updates=D_updates)

    plot_fig(generate, discriminate)
    plt.title('Before Training')
    plt.show()

    epochs = 400
    histd, histg = np.zeros(epochs), np.zeros(epochs)
    k = 20

    visualize_training = False # set to True to monitor training progress

    plt.ion()

    for i in range(epochs):
        for j in range(k):
            x = np.int32(data[i * k + j]) / 12000  # sampled m-batch from p_data
            z = sample_noise(M)  # sample m-batch from noise prior
            histd[i] = D_train(z.reshape(M, 1), x.reshape(M, 1))
        z = sample_noise(M)
        histg[i] = G_train(z.reshape(M, 1))
        if i % 10 == 0:
            G_lr *= 0.999
            D_lr *= 0.999
        if visualize_training:
            plt.clf()
            plot_fig(generate, discriminate)
            plt.draw()

    plt.ioff()

    plt.clf()
    plt.plot(range(epochs), histd, label='obj_d')
    plt.plot(range(epochs), 1 - histg, label='obj_g')
    plt.legend()
    plt.show()

    plot_fig(generate, discriminate)
    plt.title('After Training')
    plt.show()