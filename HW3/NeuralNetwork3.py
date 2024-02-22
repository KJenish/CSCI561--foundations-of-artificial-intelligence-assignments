import numpy as np

import sys
import time

class NN:
    def __init__(self,X,Y,X_test,h1=8,h2=4,learning_rate=0.01,epochs=100): #Defining the parameters of the neural network

     
      self.X=X
      self.Y=Y[:,None]
      self.X_test = X_test
      self.epochs = 25000

      np.random.seed(2)
      self.input_nodes = X.shape[1]   # number of features in the training data

      self.h1 = h1
      self.h2 = h2
      self.output_nodes = self.Y.shape[1]
      self.learning_rate = learning_rate


#----------------Initialising the weights and bias at random for the Neural network----------------#
     
      self.w1 = 2 * np.random.random((self.input_nodes,self.h1))-1
      self.b1 = 2 * np.random.random([1,self.h1]) - 1
      self.w2 = 2 * np.random.random((self.h1,self.h2))-1
      self.b2 = 2 * np.random.random([1,self.h2]) - 1
      self.w3 = 2 * np.random.random((self.h2,self.output_nodes))-1
      self.b3 = 2 * np.random.random([1,self.output_nodes]) - 1

      self.fit(self.X,self.Y)
      prediction = self.predict(self.X_test)
      prediction = np.around(prediction)
      #df = pd.DataFrame(prediction)
      #df.to_csv("test_predictions.csv", index = False, header = False)  
      np.savetxt("test_predictions.csv", prediction, delimiter=',', fmt='%d')

#----------------Activation function definitions(sigmoid,relu and their derivatives) and function to caluclate M.S.E---------------#
    def sigmoid(self,Z): 
      return 1.0/(1.0+np.exp(-Z))

    def sigmoid_prime(self,Z):
      return Z * (1-Z)
    
    def relu(self,x):
      return np.maximum(0,x)

    def MSE(self,x,y):
      return np.average((x-y)**2)

#--------------Training function which perfroms forward and back propagation, updating the weights and bias, thus training out ANN---------#
    def fit(self,X,Y):
      loss = 0

      #Forward Propagation
      for i in range(self.epochs):
        c1 = self.sigmoid((np.dot(X,self.w1) + self.b1)) 
        c1 = self.relu(c1)
        c2 = self.sigmoid(((np.dot(c1,self.w2) + self.b2)))
        c2 = self.relu(c2)
        c3 = self.sigmoid((np.dot(c2,self.w3)+self.b3))

        error = self.Y - c3
      
      #Back Propagation
        c3_d = error * self.sigmoid_prime(c3)
        c2_d = c3_d.dot(self.w3.T) * self.sigmoid_prime(c2)
        c1_d = c2_d.dot(self.w2.T) * self.sigmoid_prime(c1)

        self.w3 = np.add(self.w3,c2.T.dot(c3_d) * self.learning_rate)
        self.b3 = np.add(self.b3,np.sum(c3_d,axis=0) * self.learning_rate)
        self.w2 = np.add(self.w2, c1.T.dot(c2_d) * self.learning_rate)
        self.b2 += np.sum(c2_d,axis=0) * self.learning_rate      
        self.w1 = np.add(self.w1, X.T.dot(c1_d) * self.learning_rate)
        self.b1 += np.sum(c1_d,axis=0) * self.learning_rate     

        loss = self.MSE(Y,c3)
        
#-----------Function to predict classes of input data--------------------#
    def predict(self,X):
    
      
      c1 = self.sigmoid((np.dot(X,self.w1) + self.b1)) 
      c1 = self.relu(c1)
      c2 = self.sigmoid(((np.dot(c1,self.w2) + self.b2)))
      c2 = self.relu(c2)
      c3 = self.sigmoid((np.dot(c2,self.w3)+self.b3))

      return c3



if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]
    path3 = sys.argv[3]
    #path4 = sys.argv[4]

    file = open(path,'rb')
    train_data = np.loadtxt(file,delimiter=",")
    file = open(path2,'rb')
    train_label = np.loadtxt(file,delimiter=",")
    file = open(path3,'rb')
    test_data = np.loadtxt(file,delimiter=",")
   
   
    nn = NN(train_data,train_label,test_data) #Creating Class object and training our Neural network with our data
    