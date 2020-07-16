import numpy as np 

"""
Creating Class of Logistic Regression which contains fit function (Computation of gradient descent, Cost function )
in other terms fit function aims to Compute the Forward Propagation and the Backward Propagation 

Finally we have Predict function, aims to predict the target 

"""
class LogisticRegression: 
    def __init__(self,lr=0.001,nb_iters=1000): 
        self.lr=lr
        self.nb_iters=nb_iters 
        self.weights=None
        self.bias=None 

    def fit(self,X,y): 
        #Initialisation of Parameters 

        nb_samples, nb_features = X.shape  # (nx,mx) 
        self.weights= np.zeros(nb_features) 
        self.bias= 0 

        #Gradient Descent 
        for _ in range(self.nb_iters): 
            #Forward Propagation 
            linear_model = np.dot(X,self.weights) + self.bias 
            Y_predicted = self._sigmoid(linear_model) 


            #Backward Propagation 

            dw=(1/nb_samples) * np.dot(X.T,(Y_predicted-y)) 
            db=(1/nb_samples) * np.sum(Y_predicted-y)


            #update parameteres
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db







    def predict(self,X): 
        #we use Sigmoid function to predict 

        linear_model = np.dot(X,self.weights) + self.bias 
        Y_predicted = self._sigmoid(linear_model)  
        Y_predicted_cls = [1 if i > 0.5 else 0 for i in Y_predicted]
        return Y_predicted_cls


    def _sigmoid(self,x): 
        return 1/ (1 + np.exp(-x))
