import numpy as np
class SGD:
    def __init__(self,learning_rate,epochs,batch_size,tol):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.weight = None
        self.bias = None
    def predict(self,X):
        return np.dot(X,self.weight) + self.bias
    def mean_squarred_error(self,y_true,y_predicted):
        return np.mean((y_true-y_predicted)**2 )
    def gradient(self,X_batch,y_batch):
        y_pred = self.predict(X_batch)
        error = y_pred-y_batch
        gradiend_weight = np.dot(X_batch.T,error) / (X_batch.shape[0])
        gradiend_bias = np.mean(error)
        return gradiend_weight,gradiend_bias
    def fit(self,X,y):
        n_sample,n_feature = X.shape
        self.weight = np.random.randn(n_feature)
        self.bias = np.random.randn()

        for iteration in range(self.epochs):
            indices = np.random.permutation(n_sample)
            X_shuffle = X[indices]
            y_shuffle = y[indices]

            for i in range(0,n_sample,self.batch_size):
                X_batch = X_shuffle[i:i+self.batch_size]
                y_batch = y_shuffle[i:i+self.batch_size]
                gradient_weight,gradient_bias = self.gradient(X_batch,y_batch)
                self.weight -= gradient_weight*self.learning_rate
                self.bias -= gradient_bias*self.learning_rate
                
            if iteration % 100 == 0:
                y_pred = self.predict(X)
                loss = self.mean_squared_error(y, y_pred)
                print(f"Epoch {iteration}: Loss {loss}") 
            if np.linalg.norm(gradient_weight) < self.tolerance:
                print("Converged at ite = " ,iteration)
                break
        return gradient_weight,gradient_bias