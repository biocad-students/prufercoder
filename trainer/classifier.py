import random
import matplotlib.pyplot as plt
import numpy as np
import lasagne
import theano
import theano.tensor as T
from builtins import range
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh
from scipy.stats import norm


def getData():
    return [list(map(int, x.split())) for x in open("../data/modif100#1500.txt", "r").read().split('\n')]


def dataSize():
    return 100


def noise():
    M = dataSize()
    z = np.float32(np.linspace(-5.0, 5.0, M) + np.random.random(M) * 0.01)
    return z


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


"""def forger(len, size):
    forgerInput = T.matrix('forger')
    layer = lasagne.layers.InputLayer(1, dataSize(), forgerInput)
    for i in range(len):
        layer = lasagne.layers.DenseLayer(layer, size, nonlinearity=rectify)
    layer = lasagne.layers.DenseLayer(layer, dataSize(), nonlinearity=linear)
    return (layer, forgerInput)

def banker(forgerOut, len, size):
    bankerInput = T.matrix('banker')
    dataLayer = lasagne.layers.InputLayer(1, dataSize(), bankerInput)
    noiseLayer = lasagne.layers.InputLayer(1, dataSize(), forgerOut)

    for i in range(len):
        dataLayer = lasagne.layers.DenseLayer(dataLayer, size, nonlinearity=tanh)
        noiseLayer = lasagne.layers.DenseLayer(noiseLayer, 10, nonlinearity=tanh, W=dataLayer.W, b=dataLayer.b)

    return (bankerInput, dataLayer, noiseLayer)

def prep():


def train(epochs, k, len, size, data):
"""

print("hello")

k = 20
epochs = 400
size = 50
length = 5

dirtyData = getData()
while [] in dirtyData:
    dirtyData.remove([])
data = list(map(prepare, dirtyData))

print("data ready")


forgerInput = T.matrix('forger')
layer = lasagne.layers.InputLayer((None, 1), forgerInput)
for i in range(length):
    layer = lasagne.layers.DenseLayer(layer, size, nonlinearity=rectify)
forger = lasagne.layers.DenseLayer(layer, dataSize(), nonlinearity=linear)

forgerOutput = lasagne.layers.get_output(forger)

print("forger ready")

bankerInput = T.matrix('banker')
dataLayer = lasagne.layers.InputLayer((None, 1), bankerInput)
noiseLayer = lasagne.layers.InputLayer((None, 1), forgerOutput)

for i in range(length):
    dataLayer = lasagne.layers.DenseLayer(dataLayer, size, nonlinearity=tanh)
    noiseLayer = lasagne.layers.DenseLayer(noiseLayer, size, nonlinearity=tanh, W=dataLayer.W, b=dataLayer.b)

dataBanker = lasagne.layers.DenseLayer(dataLayer, 1, nonlinearity=sigmoid)
noiseBanker = lasagne.layers.DenseLayer(noiseLayer, 1, nonlinearity=sigmoid, W=dataBanker.W, b=dataBanker.b)

dataBankerOutput = lasagne.layers.get_output(dataBanker)
noiseBankerOutput = lasagne.layers.get_output(noiseBanker)

print("banker ready")

forgerFunc = (T.log(noiseBankerOutput)).mean()
bankerFunc = (T.log(dataBankerOutput) + T.log(1 - noiseBankerOutput)).mean()

# parameters update and training
forgerParams = lasagne.layers.get_all_params(forger, trainable=True)
forgerLearningRate = theano.shared(np.array(0.01, dtype=theano.config.floatX))
forgerUpdates = lasagne.updates.nesterov_momentum(1 - forgerFunc, forgerParams, learning_rate=forgerLearningRate,
                                                  momentum=0.6)
forgerTrain = theano.function([forgerInput], forgerFunc, updates=forgerUpdates)

bankerParams = lasagne.layers.get_all_params(dataBanker, trainable=True)
bankerLearningRate = theano.shared(np.array(0.1, dtype=theano.config.floatX))
bankerUpdates = lasagne.updates.nesterov_momentum(1 - bankerFunc, bankerParams, learning_rate=bankerLearningRate,
                                                  momentum=0.6)
bankerTrain = theano.function([forgerInput, bankerInput], bankerFunc, updates=bankerUpdates)
np.random.shuffle(data)

print("train started")

histd, histg = np.zeros(epochs), np.zeros(epochs)
for i in range(epochs):
    print(i)
    for j in range(k):
        x = np.int32(data[i * k + j]) / np.linalg.norm(np.int32(data[i * k + j])) # sampled m-batch from p_data
        z = noise()  # sample m-batch from noise prior
        histd[i] = bankerTrain(z.reshape(dataSize(), 1), x.reshape(dataSize(), 1))
    z = noise()
    histg[i] = forgerTrain(z.reshape(dataSize(), 1))
    # if i % 10 == 0:
    #    G_lr *= 0.999
    #    D_lr *= 0.999


"""def classifier(size, len):
    input1 = T.matrix("input1")
    input2 = T.matrix("input2")
    layer1 = lasagne.layers.InputLayer((1, dataSize()), input1)
    layer2 = lasagne.layers.InputLayer((1, dataSize()), input2)
    for i in range(len):
        layer1 = lasagne.layers.DenseLayer(layer1, size, nonlinearity=tanh)
        layer2 = lasagne.layers.DenseLayer(layer2, size, nonlinearity=tanh, W = layer1.W, b = layer1.b)

    output1 = lasagne.layers.DenseLayer(layer1, 1, nonlinearity=sigmoid)
    output2 = lasagne.layers.DenseLayer(layer2, 1, nonlinearity=sigmoid, W=output1.W, b=output1.b)
    func = (T.log(lasagne.layers.get_output(output1)) + T.log(1 - lasagne.layers.get_output(output2)))

    params = lasagne.layers.get_all_params(output1, trainable=True)
    lr = theano.shared(np.array(0.1, dtype=theano.config.floatX))
    updates = lasagne.updates.nesterov_momentum(1 - func.mean(), params, learning_rate=lr, momentum=0.6)
    train = theano.function([input1, input2], func, updates=updates)
    return train

train = classifier(110, 5)
data = getData()
while [] in data:
    data.remove([])
trainData = list(map(prepare, data))
data = np.random.shuffle(data)

for i in range(400): # sample m-batch from noise prior
    train(np.int32(trainData[i]).reshape(1, dataSize()), np.int32(noise()).reshape(1, dataSize()))"""
