import numpy as np
from numpy.linalg import inv
   

class Linear_Regression:
    def __init__(self):
        pass    
    
    def linear_fit(self, x_train, y_train):
        x = self.add_padding(x_train)
        coeff = self.coefficient(x, y_train)
        self.coeff = coeff
        self.mse = self.mean_squared(y_train, self.predict(x_train))
        
    def predict (self, x):
        x_test = self.add_padding(x)
        y = np.matmul(self.coeff, np.transpose(x_test))
        return y
    
    def get_weights(self):
        return self.coeff[1:]
    
    def get_bias (self):
        return self.coeff[0]
    
    def get_mse(self):
        return self.mse
    
    def add_padding (self, X):
        n,m = X.shape # for generality
        Xnew = np.hstack((np.ones((n,1)),X))
        return Xnew

    def coefficient (self, x,y):
        xtx = (np.matmul(np.transpose(x),x))
        xtx_inverse = inv(xtx)
        coeff = np.matmul(np.matmul(xtx_inverse, np.transpose(x)), y)
        return coeff

    def mean_squared(self, x, y):
        if len(x) != len (y):
            print("Error: Dimensions of inputs not same")        
        else:
            sum = 0
            for i in range(len(x)):
                sum += (x[i]-y[i])**2
            return sum/len(x)    

if __name__ == '__main__':
    print(numpy.__version__)
