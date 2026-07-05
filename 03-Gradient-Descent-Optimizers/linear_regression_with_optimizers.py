import numpy as np
import matplotlib.pyplot as plt
#===========================================
class linear_regression_model():
    def __init__(self,eta,epochs,beta1,beta2,epsilon):
        self.eta=eta
        self.epochs=epochs
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.cost_value=np.zeros(epochs)
        self.w=None
        self.b=None
    # traning the model with monmentum optimizer
    def fit_momentum(self,x,y,num):
        np.random.seed(10)
        self.w=np.random.rand(1)[0]
        self.b=np.random.rand(1)[0]
        v_dw=0 # velocity for weight
        v_db=0 # velocity for bias
        #training loop using minibatch gradient descent with momentum
        for i in range(self.epochs):
            for j in range(0, len(x), num):
                # Calculate the predictions
                x_batch = x[j:j+num]  # Select a mini-batch of data
                y_batch = y[j:j+num]  # Select the corresponding mini-batch of labels
                yhat=(self.w*x_batch)+self.b
                # Calculate the gradients
                dw=(yhat-y_batch)@x_batch/len(x_batch)
                db=np.mean(yhat - y_batch)
                # Update the velocities
                v_dw=self.beta1*v_dw+(1-self.beta1)*dw
                v_db=self.beta1*v_db+(1-self.beta1)*db
                # Update the parameters
                self.w=self.w-self.eta*v_dw
                self.b=self.b-self.eta*v_db
            # Calculate the cost for the current epoch
            self.cost_value[i]=0.5*(np.mean((yhat-y_batch)**2))
            if self.cost_value[i]<1e-5:
                   self.last_epoch = i
                   break
            else:
              self.last_epoch = self.epochs - 1

    # traning the model with RMS prop optimizer
    def fit_rmsprop(self,x,y,num):
        np.random.seed(10)
        self.w=np.random.rand(1)[0]
        self.b=np.random.rand(1)[0]
        v_dw=0 # velocity for weight
        v_db=0 # velocity for bias
        #training loop using minibatch gradient descent with momentum
        for i in range(self.epochs):
            for j in range(0, len(x), num):
                # Calculate the predictions
                x_batch = x[j:j+num]  # Select a mini-batch of data
                y_batch = y[j:j+num]  # Select the corresponding mini-batch of labels
                yhat=(self.w*x_batch)+self.b
                # Calculate the gradients
                dw=(yhat-y_batch)@x_batch/len(x_batch)
                db=np.mean(yhat - y_batch)
                # Update the velocities
                v_dw=self.beta2*v_dw+(1-self.beta2)*dw**2
                v_db=self.beta2*v_db+(1-self.beta2)*db**2
                # Update the parameters
                self.w=self.w-self.eta*(dw/(np.sqrt(v_dw)+self.epsilon))
                self.b=self.b-self.eta*(db/(np.sqrt(v_db)+self.epsilon))
            # Calculate the cost for the current epoch
            self.cost_value[i]=0.5*(np.mean((yhat-y_batch)**2))
            if self.cost_value[i]<1e-5:
                  self.last_epoch = i
                  break
            else:
              self.last_epoch = self.epochs - 1

    # traning the model with ADAM optimizer
    def fit_adam(self,x,y,num):
        np.random.seed(10)
        self.w=np.random.rand(1)[0]
        self.b=np.random.rand(1)[0]
        v_dw1=0 # velocity for weight
        v_dw2=0 # velocity for weight
        v_db1=0 # velocity for bias
        v_db2=0 # velocity for bias
        #training loop using minibatch gradient descent with momentum
        for i in range(self.epochs):
            for j in range(0, len(x), num):
                # Calculate the predictions
                x_batch = x[j:j+num]  # Select a mini-batch of data
                y_batch = y[j:j+num]  # Select the corresponding mini-batch of labels
                yhat=(self.w*x_batch)+self.b
                # Calculate the gradients
                dw=(yhat-y_batch)@x_batch/len(x_batch)
                db=np.mean(yhat - y_batch)
                # Update the velocities
                v_dw1=self.beta1*v_dw1+(1-self.beta1)*dw
                v_dw2=self.beta2*v_dw2+(1-self.beta2)*dw**2
                v_db1=self.beta1*v_db1+(1-self.beta1)*db
                v_db2=self.beta2*v_db2+(1-self.beta2)*db**2
                # Update the parameters
                self.w=self.w-self.eta*(v_dw1/(np.sqrt(v_dw2)+self.epsilon))
                self.b=self.b-self.eta*(v_db1/(np.sqrt(v_db2)+self.epsilon))
            # Calculate the cost for the current epoch
            self.cost_value[i]=0.5*(np.mean((yhat-y_batch)**2))
            if self.cost_value[i]<1e-5:
                  self.last_epoch = i
                  break
            else:
              self.last_epoch = self.epochs - 1

    #prediction of the model
    def predict(self, x):
         print(f"equation of line is y = {self.w:.4f} * x + {self.b:.4f}")
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
      plt.savefig("output.png", dpi=300, bbox_inches="tight")
      plt.show()

    #run of model
    def run_model(self,x,y,new_x, name_optimizer,num):
        if name_optimizer == "momentum":
            self.fit_momentum(x,y,num)
        elif name_optimizer == "rmsprop":
            self.fit_rmsprop(x,y,num)
        elif name_optimizer == "adam":
            self.fit_adam(x,y,num)
        else:
            print("Invalid optimizer name. Please choose 'momentum', 'rmsprop', or 'adam'.")
            return
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




model=linear_regression_model(0.01,1000,0.9,0.999,1e-8)  
x=np.array(list(map(int, input("Enter the number of data points from n to t spraded by comma" ": ").split(","))))
y=np.array(list(map(int, input("Enter the corresponding y values: ").split(","))))
new_x=np.array(list(map(int, input("Enter the new x values for prediction: ").split(","))))
num=int(input("Enter the number of data points for each batch: "))
name_optimizer=input("Enter the name of optimizer (momentum, rmsprop, adam): ")
model.run_model(x,y,new_x, name_optimizer,num)
