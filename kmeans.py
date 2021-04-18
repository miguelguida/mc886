import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from random import *
import pandas as pd
import sys
from scipy.spatial import distance

from mpl_toolkits.mplot3d import Axes3D

def import_dataset():
    with open('cluster.dat') as binary_file:
        x = []
        y = []
        for d in binary_file:
            string = d.split()
            x.append(float(string[0]))
            y.append(float(string[1]))
            
    array = np.ndarray(shape=(len(x), 2), dtype=float)

    for i in range(len(x)):
        array[i][0] = x[i]
        array[i][1] = y[i]

    return array

# array = import_dataset()
def import_wine_dataset():
    wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', \
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',\
                  'Proline']
    wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names) 
    wine_with_class = pd.DataFrame(wine_data)
    wine_with_class.Class = wine_with_class.Class - 1 # formata a coluna "Class"

    # wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= 'Class', figsize=(12,8), colormap='jet')
    # show()
    # print(wine_df.to_numpy())
    wine_df = wine_with_class[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', \
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',\
                  'Proline']]
    return wine_df.to_numpy()

# array = import_dataset()
array = import_wine_dataset()

scaler = StandardScaler()
scaler.fit(array)
scaler.mean_
scaled_data = scaler.transform(array)

pca = PCA(n_components=9)
pca_data = pca.fit_transform(scaled_data)
print(pca.explained_variance_ratio_)

def euclideanDistance(p, q):
    return distance.euclidean(p[:9], q[:9], pca.explained_variance_ratio_)

# calculate euclidean distances
def getClusterMap(k, centroids, array):
    #initialize cluster_map
    cluster_map = {}
    for i in range(k):
        cluster_map[i] = []
    
    for p in array:
        min_dist = -1
        cluster = -1
        for c in range(len(centroids)):
#             index = centroids.index(c)
            c_xy = centroids[c]
            distance = euclideanDistance(c_xy, p)
            if (min_dist == -1) or (distance < min_dist):
                min_dist = distance
                cluster = c
        cluster_map[cluster].append(p)
    
    return cluster_map

def getNewCentroids(clusterMap):
    new_centroids = []
    for c in clusterMap:
        points = clusterMap[c]
        
        min_sum_dist = -1
#         new_centroid = -1
        for i in range(len(points)):
            sum_dist = 0
            for j in range(len(points)):
                sum_dist += euclideanDistance(points[i], points[j])
                
            if (min_sum_dist == -1) or (sum_dist < min_sum_dist):
                new_centroid = points[i]
                min_sum_dist = sum_dist
        new_centroids.append(new_centroid)
    
    return new_centroids
        
def plotCluster(clusterMap):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c in clusterMap:
        points = clusterMap[c]
        array = np.ndarray(shape=(len(points), 3), dtype=float)
        for i in range(len(points)):
            array[i][0] = points[i][0]
            array[i][1] = points[i][1]
            array[i][2] = points[i][2]

        ax.scatter(array[:,0], array[:,1], array[:,2], marker='o')
    
        #plt.scatter(array[:,0], array[:,1])
    plt.show()
    
def printCentroids(k, centroids):
    for i in range(k):
        centroid = centroids[i]
        print(centroid)
        
def ForgyInitialization(k, array):
    centroids = []
    for i in range(k):
        centroid_index = randrange(len(array))
        centroid = array[centroid_index]
        centroids.append(centroid)
    return centroids

def D(x, centers):
    dists = []

    for c in centers:
        dists.append(euclideanDistance(x, c))

    return np.amin(dists)

def plusPlusInitialization(k, points):
    # D(x) menor distancia ao centro mais prox
    # prob = D(x) / sum (D(x) ** 2)
    centroids = []
    # 1 - escolher 1 centro aleatoriamente
    centroids.append(points[randrange(len(points))])

    # 2 - pra cada nao escolhido calcular D(x) e a prob
    for n in range(k - 1):
        min_dist_acc = 0
        probabilities = []
        Ds = []
        for p in points:
            d = D(p, centroids)
            min_dist_acc += d ** 2
            Ds.append(d)
        for d in Ds:    
            probabilities.append(d / min_dist_acc)
        # 3 - Escolher outro ponto usando a função de prob proporcional a D(x)² - distante dos outros centros mais prox
        centroids.append(
            points[
                np.random.choice(len(points), size = 1, p = probabilities/np.sum(probabilities))[0]
            ]
        )
    
    # 5 retornar centroides
    return centroids
    
def kmeans(k, array, n_iter=5):
# start     initialize centroids --- TODO Escada
    centroids = plusPlusInitialization(k, array)
    
    # centroids = plusplusInitialization(k, array)
    #  RandomPartitionInitialization ?
    #  kmeans ++   !!!
    
# end     initialize centroids
        
    # Conseguimos uma lista com 'k' pontos para servirem de centroides iniciais
    for n in range(n_iter):
        # calculamos os clusters 
        # printCentroids(k, centroids)
        clusterMap = getClusterMap(k, centroids, array)

        # atualizaremos nossos centroids a partir da media das dist euclid
        centroids = getNewCentroids(clusterMap)
        plotCluster(clusterMap)
    clusterMap = getClusterMap(k, centroids, array)
    
    plotCluster(clusterMap)
                                 

kmeans(4, scaled_data, 10)
    
    