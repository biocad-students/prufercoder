import lasagne
import numpy as np
import theano.tensor as T
from lasagne.layers import Gate
from lasagne.nonlinearities import sigmoid, theano, softmax, tanh
from tqdm import tqdm

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
        for i in range(half + 1):
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
    return res

def m(data):
    dic = {0:0}
    un_dic = []
    i = 1
    for vec in tqdm(data):
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


data = [list(map(int, x.split())) for x in open("../data/modif30#1500.txt", "r").read().split('\n')]

np.random.shuffle(data)

newData = []
for elem in tqdm(data):
    if len(elem) != 0:
        newData.append(prepare(elem))
data = newData
dict, un_dict = m(data)


print("prepared")
LEARNING_RATE = .01


"""
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

forgetgate = Gate(b=banker1.b_forgetgate, W_in=banker1.W_in_to_forgetgate, W_hid=banker1.W_hid_to_forgetgate, W_cell=banker1.W_cell_to_forgetgate)
outgate    = Gate(b=banker1.b_outgate,    W_in=banker1.W_in_to_outgate,    W_hid=banker1.W_hid_to_outgate,    W_cell=banker1.W_cell_to_outgate)
ingate     = Gate(b=banker1.b_ingate,     W_in=banker1.W_in_to_ingate,     W_hid=banker1.W_hid_to_ingate,     W_cell=banker1.W_cell_to_ingate)
cell       = Gate(b=banker1.b_cell,       W_in=banker1.W_in_to_cell,       W_hid=banker1.W_hid_to_cell)
banker2 = lasagne.layers.LSTMLayer(banker2_in, 2 * len(dict), forgetgate=forgetgate, outgate=outgate, ingate=ingate, cell=cell)

banker1 = lasagne.layers.LSTMLayer(banker1, len(dict))

forgetgate = Gate(b=banker1.b_forgetgate, W_in=banker1.W_in_to_forgetgate, W_hid=banker1.W_hid_to_forgetgate, W_cell=banker1.W_cell_to_forgetgate)
outgate    = Gate(b=banker1.b_outgate,    W_in=banker1.W_in_to_outgate,    W_hid=banker1.W_hid_to_outgate,    W_cell=banker1.W_cell_to_outgate)
ingate     = Gate(b=banker1.b_ingate,     W_in=banker1.W_in_to_ingate,     W_hid=banker1.W_hid_to_ingate,     W_cell=banker1.W_cell_to_ingate)
cell       = Gate(b=banker1.b_cell,       W_in=banker1.W_in_to_cell,       W_hid=banker1.W_hid_to_cell)
banker2 = lasagne.layers.LSTMLayer(banker2, len(dict), forgetgate=forgetgate, outgate=outgate, ingate=ingate, cell=cell)

banker1 = lasagne.layers.DenseLayer(banker1, 1, nonlinearity=sigmoid)
banker2 = lasagne.layers.DenseLayer(banker1, 1, nonlinearity=sigmoid, W=banker1.W, b=banker1.b)

banker1_out = lasagne.layers.get_output(banker1)
banker2_out = lasagne.layers.get_output(banker2)

forger_cost = (T.log(banker2_out)).mean()
banker_cost = (T.log(banker1_out) + T.log(1 - banker2_out)).mean()

forger_params = lasagne.layers.get_all_params(reshape, trainable=True)
banker_params = lasagne.layers.get_all_params(banker1, trainable=True)

forger_updates = lasagne.updates.adagrad(1 - forger_cost, forger_params, LEARNING_RATE)
banker_updates = lasagne.updates.adagrad(1 - banker_cost, banker_params, LEARNING_RATE)

forgerTrain = theano.function([forger_in.input_var], forger_cost, updates=forger_updates)
bankerTrain = theano.function([forger_in.input_var, banker1_in.input_var], banker_cost, updates=banker_updates)
"""

forger_in = lasagne.layers.InputLayer(shape=(batch_size, vector_len, len(dict)))

forger_lstm = lasagne.layers.LSTMLayer(forger_in, 2 * len(dict), nonlinearity=tanh)
forger_lstm = lasagne.layers.LSTMLayer(forger_lstm, len(dict), nonlinearity=tanh)

reshape = lasagne.layers.ReshapeLayer(forger_lstm, (batch_size * vector_len, len(dict)))
dense = lasagne.layers.DenseLayer(reshape, len(dict), nonlinearity=softmax)
reshape = lasagne.layers.ReshapeLayer(dense, (batch_size, vector_len, len(dict)))

out = lasagne.layers.get_output(reshape)

banker1_in = lasagne.layers.InputLayer((batch_size, vector_len, len(dict)))
banker2_in = lasagne.layers.InputLayer((batch_size, vector_len, len(dict)), out)

banker3 = lasagne.layers.LSTMLayer(banker1_in, 2 * len(dict), nonlinearity=tanh)

banker2 = lasagne.layers.LSTMLayer(banker3, len(dict), nonlinearity=tanh)

#banker2 = lasagne.layers.BatchNormLayer(banker2)

banker1 = lasagne.layers.DenseLayer(banker2, 1, nonlinearity=sigmoid)

banker1_out = lasagne.layers.get_output(banker1)
banker2_out = lasagne.layers.get_output(banker1, out)

forger_cost = (T.log(banker2_out)).mean()
banker_cost = (T.log(banker1_out) + T.log(1 - banker2_out)).mean()

"""
forger_cost = (banker2_out).mean()
banker_cost = (banker1_out.mean() * (1 - banker2_out.mean()))
"""
forger_params = lasagne.layers.get_all_params(reshape, trainable=True)
banker_params = lasagne.layers.get_all_params(banker1, trainable=True)

forger_updates = lasagne.updates.adagrad(1 - forger_cost, forger_params, LEARNING_RATE)
banker_updates = lasagne.updates.adagrad(1 - banker_cost, banker_params, LEARNING_RATE)

forgerTrain = theano.function([forger_in.input_var], forger_cost, updates=forger_updates)
bankerTrain = theano.function([forger_in.input_var, banker1_in.input_var], banker_cost, updates=banker_updates)

res = theano.function([forger_in.input_var], out)

print("gan ready")

import math

f1 = theano.function([forger_in.input_var], banker2_out)
f2 = theano.function([banker1_in.input_var], banker1_out)
f1l1 = theano.function([forger_in.input_var], lasagne.layers.get_output(banker3, out))
f2l1 = theano.function([banker1_in.input_var],lasagne.layers.get_output(banker3))
f1l2 = theano.function([forger_in.input_var], lasagne.layers.get_output(banker2, out))
f2l2 = theano.function([banker1_in.input_var], lasagne.layers.get_output(banker2))
"""
def show(i):
    print("FIRST LSTM")
    print(i)
    print("IN IN")
    print(banker3.W_in_to_ingate.get_value())
    print("CELL IN")
    print(banker3.W_cell_to_ingate.get_value())
    print("HID IN")
    print(banker3.W_hid_to_ingate.get_value())
    print("IN CELL")
    print(banker3.W_in_to_cell.get_value())
    print("HID CELL")
    print(banker3.W_hid_to_cell.get_value())
    print("IN FOR")
    print(banker3.W_in_to_forgetgate.get_value())
    print("CELL FOR")
    print(banker3.W_cell_to_forgetgate.get_value())
    print("HID FOR")
    print(banker3.W_hid_to_forgetgate.get_value())
    print("IN OUT")
    print(banker3.W_in_to_outgate.get_value())
    print("CELL OUT")
    print(banker3.W_cell_to_outgate.get_value())
    print("HID OUT")
    print(banker3.W_hid_to_outgate.get_value())
    print("B CELL")
    print(banker3.b_cell.get_value())
    print("B FOR")
    print(banker3.b_forgetgate.get_value())
    print("B OUT")
    print(banker3.b_outgate.get_value())
    print("B IN")
    print(banker3.b_ingate.get_value())
    print("SECOND LSTM")
    print(i)
    print("IN IN")
    print(banker2.W_in_to_ingate.get_value())
    print("CELL IN")
    print(banker2.W_cell_to_ingate.get_value())
    print("HID IN")
    print(banker2.W_hid_to_ingate.get_value())
    print("IN CELL")
    print(banker2.W_in_to_cell.get_value())
    print("HID CELL")
    print(banker2.W_hid_to_cell.get_value())
    print("IN FOR")
    print(banker2.W_in_to_forgetgate.get_value())
    print("CELL FOR")
    print(banker2.W_cell_to_forgetgate.get_value())
    print("HID FOR")
    print(banker2.W_hid_to_forgetgate.get_value())
    print("IN OUT")
    print(banker2.W_in_to_outgate.get_value())
    print("CELL OUT")
    print(banker2.W_cell_to_outgate.get_value())
    print("HID OUT")
    print(banker2.W_hid_to_outgate.get_value())
    print("B CELL")
    print(banker2.b_cell.get_value())
    print("B FOR")
    print(banker2.b_forgetgate.get_value())
    print("B OUT")
    print(banker2.b_outgate.get_value())
    print("B IN")
    print(banker2.b_ingate.get_value())
val = []
i = 0
for i in range(10):
    print(i)
    val = noise(len(dict))
    print(f1(val))
    print(f2(vect[i]))
    print(f1l1(val))
    print(f2l1(vect[i]))
    print(f1l2(val))
    print(f2l2(vect[i]))
    if math.isnan(bankerTrain(val, vect[i])):
        break
    #show(i)
#bankerTrain(noise(len(dict)), vect[0:batch_size])
"""
epochs = 5000

for k in tqdm(range(epochs)):
    try:
        j = k + 2000
        dat = get_data(data, j, dict)
        val = noise(len(dict))
        res1 = bankerTrain(val, dat)
        for i in tqdm(range(10)):
            val = noise(len(dict))
            res2 = forgerTrain(val)
        if res1 == 0 or res1 == 1 or res2 == 0 or res2 == 1:
            print(f1(val))
            print(f2(dat))
            print(f1l1(val))
            print(f2l1(dat))
            print(f1l2(val))
            print(f2l2(dat))
    except:
        continue

"""
epochs = 5
k = 10

for i in tqdm(range(epochs)):
    print(i)
    for j in tqdm(range(k)):
        print('#')
        print(bankerTrain(noise(len(dict)), vect[(i * k + j) * batch_size: (i * k + j + 1) * batch_size]))
    print('*')
    print(forgerTrain(noise(len(dict))))"""