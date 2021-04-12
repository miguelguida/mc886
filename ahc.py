import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from random import *
import pandas as pd
import sys

def import_dataset():
    with open('cluster.dat') as binary_file:
        data = []
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

array = import_dataset()

scaler = StandardScaler()
scaler.fit(array)
scaler.mean_
scaled_data = scaler.transform(array)


'''
    Solucao para Agglomerative Hierarchical Clustering

    1- Inicializa as distancias e o clusterMap
        -> clusterMap vai ser um hashMap indicando a que cluster cada ponto pertence
    2- Faz n - k iteracoes para achar os melhores agrupamentos




'''


def euclideanDistance(p, q):
    soma = 0
    for i in range(len(p)):
        soma += (p[i] - q[i])**2
    
    return np.sqrt(soma)


def initializeClusterMap(points):
    clusterMap = {}
    for i in range(len(points)):
        clusterMap[i] = i

    return clusterMap

def initializeDistances(points):
    distances = np.empty((len(points), len(points)))
    for i in range(len(points)):
        for j in range(len(points)):
            distances[i][j] = euclideanDistance(points[i], points[j])

    return distances
    
# pi e pj sao indices dos pontos no array
def mergeCluster(clusterMap, pi, pj):
    ci = clusterMap[pi]
    cj = clusterMap[pj]
    for c in clusterMap:
        if clusterMap[c] == cj:
            clusterMap[c] = ci
            
    return clusterMap

def isOnSameCluster(pi, pj, clusterMap):
    if clusterMap[pi] == clusterMap[pj]:
        return True
    return False

def agglomerativeHierarchicalCluster(points, k):
    clusterMap = initializeClusterMap(points)
    distances = initializeDistances(points)
    n = len(points) # numero de clusteres
    
    while n > k:
        print("n= ", n)
        min_distance = sys.maxsize
        b = 1
        for i in range(len(points)):
            for j in range(b, len(points)):
                if (distances[i][j] < min_distance) and (not isOnSameCluster(i, j, clusterMap)):
                    
                    min_distance = distances[i][j]
                    min_distance_i = i
                    min_distance_j = j
            b += 1
        
        clusterMap = mergeCluster(clusterMap, min_distance_i, min_distance_j)
        print(min_distance, "   I: ", min_distance_i, "   J: ", min_distance_j, " IS: ", isOnSameCluster(min_distance_i, min_distance_j, clusterMap))
   
        n -= 1
        

       

    return clusterMap

print(agglomerativeHierarchicalCluster(scaled_data, 3))