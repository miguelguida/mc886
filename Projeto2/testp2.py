import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from random import *
import pandas as pd
import sys

from mpl_toolkits.mplot3d import Axes3D


# with open('kick2.dat') as binary_file:
   
#     for d in binary_file:
#         print(d)
        

def import_dataset(filename):
    with open(filename) as binary_file:
        data = []
        x = []
        y = []
        z = []
        for d in binary_file:
            string = d.split()
            x.append(float(string[0]))
            y.append(float(string[1]))
            z.append(float(string[2]))
            
    array = np.ndarray(shape=(len(x), 3), dtype=float)

    for i in range(len(x)):
        array[i][0] = x[i]
        array[i][1] = y[i]
        array[i][2] = z[i]

    return array

kick1 = import_dataset('kick1.dat')
# kick2 = import_dataset('kick2.dat')

print(kick1)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(kick1[:,0], kick1[:,1], kick1[:,2]) # plot the point (2,3,4) on the figure

plt.show()