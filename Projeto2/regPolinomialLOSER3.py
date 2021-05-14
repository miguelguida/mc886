import numpy as np
import matplotlib.pyplot as plt

import sys

from mpl_toolkits.mplot3d import Axes3D


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

# kick1 = import_dataset('kick1.dat')
kick2 = import_dataset('kick2.dat')

# print(kick1)

# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(kick2[:,0], kick2[:,1], kick2[:,2]) # plot the point (2,3,4) on the figure

# plt.show()

def regressaoPolinomial(X_original, Y):
    print(X_original.shape)
    
    # X = np.array(X_original, copy=True)
    Y2 = X_original[:,:]**2
    Y3 = X_original[:,:]**3
    X = np.hstack((X_original, Y2.reshape((X_original.shape[0],2)),Y3.reshape((X_original.shape[0],2))))
    print(X)
    n = X.shape[1]
    m = X.shape[0]
    W = np.random.rand(1,n+1)*0.001
    print("Init W: ", W)           
    
    learning_rate = 0.003
    costs = []
    iterations = 60

    for it in range(0,iterations):
        W, j = gradient_desc(W,X,Y,m,n,learning_rate)
        costs.append(j)
    
    h = calc_h(W, X)
    plotRegression(W,X,Y,h,iterations,costs, X_original)


    

def cost(h, Y):
    m = Y.shape[0]
    j = (1/(2*m))*np.sum((h-Y)**2)
    return j

def calc_h(W, X):
    m = X.shape[0]
    h = np.dot(W[0,1:],X.T).reshape((m, 1))+W[0,0]
    return h

def gradient_desc(W,X,Y,m,n,learning_rate):
        h = calc_h(W,X)
        # print(h.shape)
        # print(h)
        j = cost(h, Y)
        print("cost: ",j)

        grads = {}
        grads["dw0"] = (1/m)*np.sum((h-Y))
        for i in range(1,n+1):
            grads["dw"+str(i)] = (1/m)*np.sum((h-Y)*X[:,i-1])
        # print(grads)
        for i in range(0,n+1):
            W[0,i] = W[0,i] - learning_rate*grads["dw"+str(i)]
        # print(W)
        return W,j

def plotRegression(W,X,Y,h,iterations,costs, X_original):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_original[:,0], X_original[:,1], Y[:]) 
    ax.scatter(X_original[:,0], X_original[:,1], h) 

    i = 2
    y = 1.109 - 0.050
    x = -0.596 - 0.030
    z = 0.11  - 0.01
    predX1 = []
    predX2 = []
    predH = []
    while y > 0:
        y = 1.109 - i*0.050
        x = -0.596 - i*0.030
        x2 = x**2
        x3 = x**3
        y2 = y **2
        y3 = y **3
        z = 0.269  - i*0.02
        h = calc_h(W,np.asarray([[x, y, x2,y2,x3,y3]]))
        predX1.append(x)
        predX2.append(y)

        predH.append(h)
        i+=1

    ax.scatter(predX1[:], predX2[:], predH[:]) # plot the point (2,3,4) on the figure
    plt.show()

    plt.plot(range(0,iterations),costs[:])
    plt.show()


regressaoPolinomial(kick2[:,:2], kick2[:,2])
# regressaoLinear(kick1[:,[1,2]], kick1[:,0])
# regressaoLinear(kick1[:,[0,2]], kick1[:,1])