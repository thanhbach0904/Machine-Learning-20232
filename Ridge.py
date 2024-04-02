import numpy as np
class RidgeRegression() : 
      
    def __init__( self, learning_rate, iterations, l2_penality ) : 
          
        self.learning_rate = learning_rate         
        self.iterations = iterations         
        self.l2_penality = l2_penality
                      
    def fit( self, X, Y ) : 
          
                 
        self.m, self.n = X.shape 
          
                 
        self.W = np.zeros( self.n ) 
          
        self.b = 0        
        self.X = X         
        self.Y = Y 
          
         
                  
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self.W,self.b
      
     
      
    def update_weights( self ) :            
        Y_pred = self.predict( self.X ) 
          
             
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +  ( 2 * self.l2_penality * self.W ) ) / self.m      
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
            
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db         
        return self
      
  
    def predict( self, X ) :     
        return X.dot( self.W ) + self.b