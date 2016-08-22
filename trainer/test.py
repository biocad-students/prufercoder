import matplotlib.pyplot as plt
import sys

import time
import numpy as np
plt.ion()
for i in range(100):
    plt.clf()
    plt.plot(range(i), np.random.rand(i))
    plt.pause(1)