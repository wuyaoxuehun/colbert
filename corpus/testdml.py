import numpy as np
from matplotlib import pyplot

import math
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def testdml():
    x = list(range(10))
    mean = 3
    std = 2
    y = [normpdf(i, mean, std) for i in x]
    y = [i/sum(y) for i in y]
    pyplot.plot(x, y)
    pyplot.show()

if __name__ == '__main__':
    testdml()