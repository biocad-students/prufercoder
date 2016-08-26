import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import sigmoid, theano, softmax, tanh

from lasagne.layers import Gate

vector_len = 30
batch_size = 20
data_size = 10000

def prepare(vector):
    res = []
    half = int(vector_len / 2)
    if len(vector) == 1:
        for i in range(half - 1):
            res.append(0)
        res.append(vector[0] + 1)
        for i in range(half):
            res.append(0)
        return res
    n = int(len(vector) / 2 - 1)
    for i in range(half - 1):
        if i < n:
            res.append(vector[i] + 1)
        else:
            res.append(0)
    for i in range(half + 1):
        if i < n + 2:
            res.append(vector[n + i])
        else:
            res.append(0)
    try:
        assert len(res) == vector_len
    except:
        print(len(res))
    return res


def m(data):
    dic = {0: 0}
    un_dic = [0]
    i = 1
    for vec in data:
        for elem in vec:
            if elem not in dic:
                dic[elem] = i
                un_dic.append(elem)
                i += 1
    return dic, un_dic


def prepareData(data, dic):
    res = []
    for vec in data:
        cur = []
        for elem in vec:
            el = np.zeros(len(dic))
            el[dic[elem]] = 1
            cur.append(el)
        res.append(cur)
    return res


def get_data(data, i, dic):
    cur = data[batch_size * i: batch_size * (i + 1)]
    res = prepareData(cur, dic)
    return np.array(res)


def noise(n):
    return np.random.random(batch_size * vector_len * n).reshape(batch_size, vector_len, n)


def get_res(arr, map):
    res = []
    for elem in arr:
        ans = []
        for v in elem:
            real = v.tolist().index(max(v))
            ans.append(map[real])
        res.append(ans)
    return res

print("prepearing data")

data = [list(map(int, x.split())) for x in open("../data/modif30#1500.txt", "r").read().split('\n')]

np.random.shuffle(data)

newData = []
for elem in data:
    if len(elem) != 0:
        newData.append(prepare(elem))
data = newData
dict, un_dict = m(data)

print("data ready")

print("creating gan")

learning_rate = .1

forger_in = lasagne.layers.InputLayer(shape=(batch_size, vector_len, len(dict)))

forger_lstm = lasagne.layers.LSTMLayer(forger_in, 2 * len(dict), nonlinearity=tanh)
forger_lstm = lasagne.layers.LSTMLayer(forger_lstm, len(dict), nonlinearity=tanh)

reshape = lasagne.layers.ReshapeLayer(forger_lstm, (batch_size * vector_len, len(dict)))
dense = lasagne.layers.DenseLayer(reshape, len(dict), nonlinearity=softmax)
reshape = lasagne.layers.ReshapeLayer(dense, (batch_size, vector_len, len(dict)))

out = lasagne.layers.get_output(reshape)
banker1_in = lasagne.layers.InputLayer((batch_size, vector_len, len(dict)))
banker2_in = lasagne.layers.InputLayer((batch_size, vector_len, len(dict)), out)

banker1 = lasagne.layers.LSTMLayer(banker1_in, 2 * len(dict), nonlinearity=tanh)

forgetgate = Gate(b=banker1.b_forgetgate, W_in=banker1.W_in_to_forgetgate, W_hid=banker1.W_hid_to_forgetgate, W_cell=banker1.W_cell_to_forgetgate, nonlinearity=banker1.nonlinearity_forgetgate)
outgate    = Gate(b=banker1.b_outgate,    W_in=banker1.W_in_to_outgate,    W_hid=banker1.W_hid_to_outgate,    W_cell=banker1.W_cell_to_outgate,    nonlinearity=banker1.nonlinearity_outgate)
ingate     = Gate(b=banker1.b_ingate,     W_in=banker1.W_in_to_ingate,     W_hid=banker1.W_hid_to_ingate,     W_cell=banker1.W_cell_to_ingate,     nonlinearity=banker1.nonlinearity_ingate)
cell       = Gate(b=banker1.b_cell,       W_in=banker1.W_in_to_cell,       W_hid=banker1.W_hid_to_cell,                                            nonlinearity=banker1.nonlinearity_cell)
banker2 = lasagne.layers.LSTMLayer(banker2_in, 2 * len(dict), forgetgate=forgetgate, ingate=ingate, outgate=outgate, cell=cell)

banker1 = lasagne.layers.LSTMLayer(banker1, len(dict), nonlinearity=tanh)

forgetgate = Gate(b=banker1.b_forgetgate, W_in=banker1.W_in_to_forgetgate, W_hid=banker1.W_hid_to_forgetgate, W_cell=banker1.W_cell_to_forgetgate, nonlinearity=banker1.nonlinearity_forgetgate)
outgate    = Gate(b=banker1.b_outgate,    W_in=banker1.W_in_to_outgate,    W_hid=banker1.W_hid_to_outgate,    W_cell=banker1.W_cell_to_outgate,    nonlinearity=banker1.nonlinearity_outgate)
ingate     = Gate(b=banker1.b_ingate,     W_in=banker1.W_in_to_ingate,     W_hid=banker1.W_hid_to_ingate,     W_cell=banker1.W_cell_to_ingate,     nonlinearity=banker1.nonlinearity_ingate)
cell       = Gate(b=banker1.b_cell,       W_in=banker1.W_in_to_cell,       W_hid=banker1.W_hid_to_cell,                                            nonlinearity=banker1.nonlinearity_cell)
banker2 = lasagne.layers.LSTMLayer(banker2, len(dict), forgetgate=forgetgate, ingate=ingate, outgate=outgate, cell=cell)

banker1 = lasagne.layers.DenseLayer(banker1, 1, nonlinearity=sigmoid)
banker2 = lasagne.layers.DenseLayer(banker2, 1, nonlinearity=sigmoid, W=banker1.W, b=banker1.b)

banker1_out = lasagne.layers.get_output(banker1)
banker2_out = lasagne.layers.get_output(banker2)

forger_cost = (T.log(banker2_out)).mean()
banker_cost = (T.log(banker1_out) + T.log(1 - banker2_out)).mean()

forger_params = lasagne.layers.get_all_params(reshape, trainable=True)
banker_params = lasagne.layers.get_all_params(banker1, trainable=True)

forger_updates = lasagne.updates.adagrad(1 - forger_cost, forger_params, learning_rate)
banker_updates = lasagne.updates.adagrad(1 - banker_cost, banker_params, learning_rate)

forgerTrain = theano.function([forger_in.input_var], forger_cost, updates=forger_updates, allow_input_downcast=True)
bankerTrain = theano.function([forger_in.input_var, banker1_in.input_var], banker_cost, updates=banker_updates,
                              allow_input_downcast=True)

res = theano.function([forger_in.input_var], out, allow_input_downcast=True)

test_func = theano.function([forger_in.input_var], banker2_out.mean(), allow_input_downcast=True)
test_banker = theano.function([banker1_in.input_var], banker1_out.mean(), allow_input_downcast=True)

print("gan ready")
print("train started")
trainlen = 20000
iter = 0
exceptions = 0


def calc(x):
    return 1200 * (x - 0.5) * (x - 0.5)

test_res = []
bank_res = []
banker_train = []
forger_train = []

for k in range(trainlen):
    try:
        dat = get_data(data, iter, dict)
        iter += 1
        val = noise(len(dict))
        bankerTrain(val, dat)
        val = noise(len(dict))
        forgerTrain(val)
        val = noise(len(dict))
        test = test_func(val)
        if test > 0.5:
            test = np.math.floor(calc(test))
            if (k < 500) :
                test = 1
            banker_train.append(test)
            forger_train.append(1)
            for i in range(test):
                dat = get_data(data, iter, dict)
                iter += 1
                val = noise(len(dict))
                bankerTrain(val, dat)
        elif test < 0.5:
            banker_train.append(1)
            test = np.math.floor(calc(test))
            forger_train.append(test)
            for i in range(test):
                val = noise(len(dict))
                forgerTrain(val)
        else:
            banker_train.append(1)
            forger_train.append(1)
        if k % 10 == 0:
            learning_rate *= .99
            dat = get_data(data, iter, dict)
            iter += 1
            val = noise(len(dict))
            test_res.append(test_func(val))
            bank_res.append(test_banker(dat))
            print(k)
            print('test')
            print(test_res[-1])
            print('bank')
            print(bank_res[-1])
            print('forger total')
            print(sum(forger_train))
            print('banker total')
            print(sum(banker_train))
    except Exception as ex:
        exceptions += 1
        print("Wow, you have an exception there!")
        print(ex)
        print("total number of exceptions is:")
        print(exceptions)

print("train ended")
"""
#code for vector generation
val = noise(len(dict))
get_res(res(val), un_dict)"""