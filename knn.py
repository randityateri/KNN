import pandas as pd
import numpy as np
from collections import Counter

#OOP
#euclidean_distance
def euclidean_dist(p1, p2):
    distance = np.sqrt(np.sum((p1-p2)**2))
    return distance

class knn():
    
    #inisialisasi
    def __init__(self,k=3):
        self.k=k
    
    #fit x train, y train
    def fit(self,X,Y):
        self.X_train=X
        self.y_train=Y
    
    #prediksi
    def predict(self,X):
        prediction=[self.predict2(x) for x in X] #have to create a helper function
        return prediction
    
    def predict2(self,x):
        
        #distance2
        distance2=[euclidean_dist(x,x_train) for x_train in self.X_train]
        
        #k terdekat
        k_indi=np.argsort(distance2)[:self.k]
        k_nearest=[self.y_train[i] for i in k_indi]
        
        #terbanyak
        common=Counter(k_nearest).most_common()
        return common[0][0]
    