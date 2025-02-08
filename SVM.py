import numpy as np

class SVM():
  def __init__(self,learning_rate,no_of_iterations,Lamda_parameter):
    self.learning_rate=learning_rate
    self.no_of_iterations=no_of_iterations
    self.Lamda_parameter=Lamda_parameter

  def fit(self,X,Y):
    self.m,self.n=X.shape
    self.w=np.zeros(self.n)
    self.b=0
    self.X=X
    self.Y=Y

    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
    Y_label=np.where(self.Y<=0,-1,1)
    for index,x_i in enumerate(self.X):
      condition=Y_label[index]*(np.dot(x_i,self.w)-self.b)>=1
      if(condition==True):
        dw=2*self.Lamda_parameter*self.w
        db=0
      else:
        dw=2*self.Lamda_parameter*self.w-np.dot(x_i,Y_label[index])
        db=Y_label[index]

      self.w=self.w-self.learning_rate*dw
      self.b=self.b-self.learning_rate*db

  def predict(self,X):
    predicted_labels=np.sign(np.dot(X,self.w)-self.b)
    Y_hat=np.where(predicted_labels<=-1,0,1)
    return Y_hat
