import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from random import *
import pandas as pd
import sys

from mpl_toolkits.mplot3d import Axes3D

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
# print(array.shape)
# print(array)

pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)
print(pca.explained_variance_ratio_)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['Alcohol', 'Malic acid', 'Ash', 'Alc. of ash', 'Magnesium', 'Total phenols', \
              'Flavanoids', 'Nonfl. phen.', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',\
              'Proline']
'''
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of ExplainedVariance')
plt.xlabel('Principal Component')
plt.show()
'''

'''
plt.bar(x=range(1, 14), height=per_var[:13], tick_label=labels[:13])
plt.show()
'''
pca_df = pd.DataFrame(pca_data.T, index=labels[:3])
print(pca_df)

loading_scores = pd.Series(pca.components_[0])
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_scores = sorted_loading_scores[:5].index.values
print(loading_scores[top_scores])


plt.scatter(pca_df[0], pca_df[1])
plt.xlabel('Pc1({1}) - {0}%'.format(per_var[0], labels[0]))
plt.ylabel('Pc2({1}) - {0}%'.format(per_var[1], labels[1]))

for idx in pca_df.index:
    plt.annotate(idx, (pca_df[0].loc[idx], pca_df[1].loc[idx]))

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df[0], pca_df[1], pca_df[2], c='r', marker='o')

ax.set_xlabel('Pc1({1}) - {0}%'.format(per_var[0], labels[0]))
ax.set_ylabel('Pc2({1}) - {0}%'.format(per_var[1], labels[1]))
ax.set_zlabel('Pc3({1}) - {0}%'.format(per_var[2], labels[2]))

plt.show()


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
'''
print(agglomerativeHierarchicalCluster(scaled_data, 3))
print(agglomerativeHierarchicalCluster(pca_data, 3))
'''