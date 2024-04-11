import numpy as np
class k_NN:
    def __init__(self,neighbors : int,coefficent :str):
        self.k = neighbors
        self.coef = coefficent
        self.Neighbors = []
    def Similarity(self,u1,u2) -> float:
        return ((u1.T).dot(u2))/((np.linalg.norm(u1))*np.linalg.norm(u2))
    def update(self,tup:tuple) -> None:
        self.Neighbors.append(tup)
        if len(self.Neighbors) > self.k:
            self.Neighbors.pop(0)