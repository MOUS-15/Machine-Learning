import numpy as np
import matplotlib.pyplot as plt

#===========================================
class logistic_regression_model():
    def __init__(self, eta, epochs, epsilon=1e-5):
        self.eta = eta
        self.epochs = epochs
        self.epsilon = epsilon
        self.w = None
        self.last_epoch = 0

    # prepare the data
    def prepare_data(self, number_of_samples):
        np.random.seed(10)
        # prepare the weights and bias
        self.w = np.random.randn(3) * 0.01
        # prepare the bias column
        bias = np.ones(number_of_samples)
        
        # cats data
        cat_x1 = np.random.uniform(1, 9, number_of_samples)
        cat_x2 = np.random.uniform(1, 9, number_of_samples)
        cats = np.column_stack((cat_x1, cat_x2, bias))
        # Logistic Regression treats target as [0, 1]
        cat_y = np.zeros(number_of_samples)
        
        # dogs data
        dog_x1 = np.random.uniform(7, 16, number_of_samples)
        dog_x2 = np.random.uniform(7, 16, number_of_samples)
        dogs = np.column_stack((dog_x1, dog_x2, bias))
        dog_y = np.ones(number_of_samples)
        
        # combine the data and labels
        self.X = np.vstack((cats, dogs))
        self.y = np.hstack((cat_y, dog_y))
        self.m = len(self.X) # Store memory sample count

    # training the model with logistic regression algorithm
    def fit(self):
        self.last_epoch = self.epochs - 1
        self.cost_function = np.zeros(self.epochs)
        
        for i in range(self.epochs):
            z = np.dot(self.X, self.w)
            # apply the sigmoid function
            yhat = 1 / (1 + np.exp(-z))
            error = self.y - yhat
            
            # Weight update using Gradient Descent
            self.w = self.w + self.eta * np.dot(self.X.T, error) / self.m
            
            # الاحتواء الرياضي السليم لمنع الانفجار الرقمي دون تشويه الحساب
            yhat_clipped = np.clip(yhat, 1e-15, 1 - 1e-15)
            self.cost_function[i] = -np.mean(self.y * np.log(yhat_clipped) + (1 - self.y) * np.log(1 - yhat_clipped))
            
            # Early stopping check
            if i > 0 and abs(self.cost_function[i] - self.cost_function[i-1]) < self.epsilon:
                self.last_epoch = i
                break

    # predict the output for new data
    def predict(self, x1, x2):
        bias = 1
        x = np.array([x1, x2, bias])
        z = np.dot(x, self.w)
        self.probability = 1 / (1 + np.exp(-z))
        self.yhat = 1 if self.probability >= 0.5 else 0

    # accuracy of the model
    def accuracy(self):
        z = np.dot(self.X, self.w)
        yhat = 1 / (1 + np.exp(-z))
        preds = (yhat >= 0.5).astype(int)
        self.correct = np.sum(preds == self.y)
        self.total = len(self.y)
        self.accuracy = self.correct / self.total * 100
        
    # draw the data on the graph
    def draw_data(self):
        x1_grid = np.linspace(0, 18, 200)
        x2_grid = np.linspace(0, 18, 200)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
        x_grid = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel(), np.ones(x1_mesh.ravel().shape[0])))

        z_grid = np.dot(x_grid, self.w)    
        h_grid = 1 / (1 + np.exp(-z_grid))
        
        plt.style.use('dark_background')
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.scatter(self.X[self.y==0][:,0], self.X[self.y==0][:,1], color='blue', label='cats')
        plt.scatter(self.X[self.y==1][:,0], self.X[self.y==1][:,1], color='red', label='dogs')
        plt.contourf(x1_mesh, x2_mesh, h_grid.reshape(x1_mesh.shape), levels=20, cmap='RdYlBu_r', alpha=0.3)
        plt.xlim(0, 18)
        plt.ylim(0, 18)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Cats and Dogs Decision Boundary')

        # draw the decision boundary line
        x1 = np.linspace(0, 20, 100)
        x2 = -(self.w[0]*x1 + self.w[2]) / self.w[1]
        plt.plot(x1, x2, color='green', label='Decision Boundary Line')
        plt.legend()
        plt.grid()

        # draw the error graph
        plt.subplot(1,2,2)
        plt.plot(range(1, self.last_epoch+2), self.cost_function[:self.last_epoch+1], color='yellow')
        plt.title('Cost vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross-Entropy Cost')
        plt.grid()
        plt.show()

    # function to print the information of the model
    def print_info(self):
        print("="*50)
        print("Learning Rate (eta) : ", self.eta)
        print("Number of Epochs    : ", self.epochs)
        print("Final Epoch         : ", self.last_epoch+1)
        print("Final Cost          : ", self.cost_function[self.last_epoch])
        print(f"training completed in {self.last_epoch+1} epochs.")
        print("="*50)
        print("Final Weights and Bias:")
        print(f"W1   = {self.w[0]:.4f}")
        print(f"W2   = {self.w[1]:.4f}")
        print(f"Bias = {self.w[2]:.4f}")
        print("="*50)
        print(f"Accuracy of the model:")
        print(f"Total Samples: {self.total}")
        print(f"Correct Predictions: {self.correct}")
        print(f"Wrong Predictions: {self.total - self.correct}")
        print(f"Accuracy: {self.accuracy:.2f}%")
        print("="*50)
        print(f"Prediction for new data point ({new_x1}, {new_x2}): {'Dog' if self.yhat == 1 else 'Cat'}")
        print(f"Probability of being Dog: {self.probability*100:.2f}%")
        print(f"Probability of being Cat: {(1-self.probability)*100:.2f}%")
        print("="*50)
        
    # run the model
    def run_model(self, number_of_samples, new_x1, new_x2):
        self.prepare_data(number_of_samples)
        self.fit()
        self.accuracy()
        self.predict(new_x1, new_x2)  # 1. الحسابات تكتمل أولاً
        self.print_info()             # 2. طباعة البيانات في الـ Terminal فوراً
        self.draw_data()              # 3. فتح الرسوم البيانية كخطوة نهائية

number_of_samples = int(input("Enter the number of samples for each class: "))
new_x1 = float(input("Enter the value of x1 for prediction: "))   
new_x2 = float(input("Enter the value of x2 for prediction: "))

model = logistic_regression_model(0.05, 50000, 1e-6) # زيادة قيمة الـ eta لأن الـ Sigmoid تحتاج خطوات أثبت
model.run_model(number_of_samples, new_x1, new_x2)