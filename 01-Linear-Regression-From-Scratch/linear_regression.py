import numpy as np
import matplotlib.pyplot as plt
#===========================================
class linear_regression_model():
    def __init__(self,eta,epochs):
        self.eta=eta
        self.epochs=epochs
        self.cost_value=np.zeros(epochs)
        self.w=None
        self.b=None
    # traning the model
    def fit(self,x,y):
        np.random.seed(10)
        self.w=np.random.rand(1)[0]
        self.b=np.random.rand(1)[0]
        for i in range(self.epochs):
            yhat=(self.w*x)+self.b
            self.w= self.w -self.eta*(yhat-y)@x/len(x)
            self.b= self.b - self.eta*(np.mean(yhat - y))
            self.cost_value[i]=0.5*(np.mean((yhat-y)**2)) 
            if self.cost_value[i]<1e-5:
                self.last_epoch = i
                break
        else:
             self.last_epoch = self.epochs - 1
    
    #prediction of the model
    def predict(self, x):
         prediction=self.w * x + self.b
         print("Predictions:")
         for i, p in zip(x, prediction):
           print(f"x = {i}  -->  y = {p:.2f}")
    
    #drawing the best line
    def plot(self, x, y):

      plt.style.use("dark_background")
      plt.figure(figsize=(14,5))

      # الرسم الأول
      plt.subplot(1,2,1)

      plt.scatter(x, y, color="yellow", label="Data Points")

      x_value = np.arange(-10,20)
      y_value = self.w*x_value + self.b

      plt.plot(x_value, y_value, color="red", label="Fit Line")

      plt.title("Linear Regression")
      plt.xlabel("X")
      plt.ylabel("Y")
      plt.legend()
      plt.grid()

      # الرسم الثاني
      plt.subplot(1,2,2)

      plt.plot(
        range(1, self.last_epoch+2),
        self.cost_value[:self.last_epoch+1],
        color="cyan")

      plt.title("Cost Function")
      plt.xlabel("Epoch")
      plt.ylabel("Cost")
      plt.grid()

      plt.tight_layout()
      plt.show()

    #run of model
    def run_model(self,x,y,new_x):
        self.fit(x,y)
        self.plot(x,y)
        print("="*50)
        print("the prediction of model is ")
        self.predict(new_x)
        print("="*50)
        print("final value")
        print(f'Final w: {self.w: 0.4f}')
        print(f'Final b: {self.b: 0.4f}')
        print(f'Final cost: {self.cost_value[self.last_epoch]: 0.4f}')
        print(f'Training stopped at epoch: {self.last_epoch+1}')




model=linear_regression_model(0.01,1000)  
x=np.arange(1,10)
y=np.arange(1,10)
new_x=np.arange(1,15,2)
model.run_model(x,y,new_x)
