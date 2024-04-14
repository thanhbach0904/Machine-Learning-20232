import numpy as np
class k_NN:
    def __init__(self,neighbors : int,coefficent :str):
        self.k = neighbors
        self.coef = coefficent
        self.Neighbors = []
    def Similarity(self,u1,u2):
        return ((u1.T).dot(u2))/((np.linalg.norm(u1))*np.linalg.norm(u2))
    def update(self,tup:tuple) -> None:
        self.Neighbors.append(tup)
    def RMSE(self,v1 : list,v2:list) -> float:
        l1 = np.array(v1)
        l2 = np.array(v2)
        return np.mean((l1-l2)**2)
    def precision_Rate(self,v1,v2) -> float:
        n = len(v1)
        temp = 0
        for i in range(n):
            if v1[i]*v2[i] > 0:
                temp += 1
        return round((temp/n)*100,2)




