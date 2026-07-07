import numpy as np
import matplotlib.pyplot as plt
#===========================================
class perceptron_model():
    def __init__(self,eta,epochs):
        self.eta=eta
        self.epochs=epochs
        self.w=None
        self.last_epoch=0

    #prepare the data
    def prepare_data(self,number_of_samples):
        np.random.seed(10)
        #prepare the weights and bias
        self.w=np.random.rand(3)
        #prepare the bais coulmn
        bais=np.ones(number_of_samples)
        #cats data
        cat_x1=np.random.uniform(1,5,number_of_samples)
        cat_x2=np.random.uniform(1,5,number_of_samples)
        cats=np.column_stack((cat_x1,cat_x2,bais))
        #cats labels
        cat_y=np.zeros(number_of_samples)
        #dogs data
        dog_x1=np.random.uniform(4,8,number_of_samples)
        dog_x2=np.random.uniform(4,8,number_of_samples)
        dogs=np.column_stack((dog_x1,dog_x2,bais))
        self.X=np.vstack((cats,dogs))
        # dogs labels
        dog_y=np.ones(number_of_samples)
        # combine the data and labels
        self.y=np.hstack((cat_y,dog_y))
        # filling the error array with zeros
    
    #training the model with perceptron learing algorithm
    def fit(self):
        self.last_epoch=self.epochs-1
        self.error=np.zeros(self.epochs)
        for i in range (self.epochs):
            self.error_counts=0
            # shuffle the data
            indices = np.random.permutation(len(self.X))
            for j in indices:
                z=np.dot(self.X[j],self.w) #dot product of input and weights
                yhat=1 if z>=0 else 0  #step function
                error=self.y[j]-yhat
                #update the weights
                if error!=0:
                    self.w+=self.eta*error*self.X[j]
                    self.error_counts+=1
            self.error[i]=self.error_counts
            if self.error_counts==0:
                self.last_epoch=i
                break

    
    # predict the output for new data
    def predict(self,x1,x2):
        bais=1
        x=np.array([x1,x2,bais])
        z=np.dot(x,self.w)
        yhat=1 if z>=0 else 0
        if yhat==0:
            print("The input data belongs to class: Cats")
        else:
            print("The input data belongs to class: Dogs")

    #accuracy of the model
    def accuracy(self):
        z = np.dot(self.X, self.w)
        yhat = (z >= 0).astype(int)
        correct = np.sum(yhat == self.y)
        total = len(self.y)
        accuracy = correct / total * 100
        print(f"Correct Predictions : {correct}")
        print(f"Wrong Predictions   : {total - correct}")
        print(f"Accuracy            : {accuracy:.2f}%")
    # drow the data on the graph
    def drow_data(self):
        plt.style.use('dark_background')
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.scatter(self.X[self.y==0][:,0],self.X[self.y==0][:,1],color='blue',label='cats')
        plt.scatter(self.X[self.y==1][:,0],self.X[self.y==1][:,1],color='red',label='dogs')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Cats and Dogs Data')

        # draw the decision boundary
        x1=np.linspace(0,10,100)
        x2=-(self.w[0]*x1+self.w[2])/self.w[1]
        plt.plot(x1,x2,color='green',label='Decision Boundary')
        plt.legend()
        plt.grid()

        #draw the error graph
        plt.subplot(1,2,2)
        plt.plot(range(1,self.last_epoch+2),self.error[:self.last_epoch+1],color='yellow')
        plt.title('Error vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.show()
    #run the model
    def run_model(self,number_of_samples,new_x1,new_x2):
        self.prepare_data(number_of_samples)
        self.fit()
        self.accuracy()
        self.drow_data()
        self.predict(new_x1,new_x2)

model=perceptron_model(0.01,1000)
number_of_samples=int(input("Enter the number of samples for each class: "))
new_x1=float(input("Enter the value of x1 for prediction: "))   
new_x2=float(input("Enter the value of x2 for prediction: "))
model.run_model(number_of_samples,new_x1,new_x2)
