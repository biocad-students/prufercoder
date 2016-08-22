from collections import Counter



"""out = open("../data/modif1.txt", "w")
data = [list(map(int, x.split())) for x in open("../data/vectors.txt", "r").read().split('\n')]

for line in data:
    x = int(len(line) / 2 - 1)
    for i in range(len(line)):
        if i >= x:
            if line[i] >= 1000:
                line[i] = int(line[i] / 1000)
            else:
                line[i] += 900
        out.write(str(line[i]))
        out.write(' ')
    out.write('\n')
out.close()
"""
"""
data = [list(map(int, x.split())) for x in open("../data/modif1.txt", "r").read().split('\n')]
print(len(data))
words = list(map(int, open("../data/modif1.txt").read().split()))
c = Counter(words)
count = 0
res = []
for line in data:
    min = 10000000
    for word in line:
        if c[word] < min:
            min = c[word]
    res.append(min)
"""
data = list(map(int, open("../data/modif1.txt").read().split()))


#Импортируем библиотеку Math
import math
#Импортируем один из пакетов Matplotlib
import pylab
#Импортируем пакет со вспомогательными функциями
from matplotlib import mlab
c = Counter(data)
#Рисуем график функции y = sin(x)
def func (x):
    """
    sin (x)
    """
    return c[x]

#Указываем X наименьее и наибольшее
xmin = 0
xmax = 10000

# Шаг между точками
dx = 1

#Создадим список координат по оси
#X на отрезке [-xmin; xmax], включая концы
xlist = mlab.frange (xmin, xmax, dx)

# Вычислим значение функции в заданных точках
ylist = [func (x) for x in xlist]

i = 0
for k in ylist:
    if k > 1500:
        i += 1

print(i)