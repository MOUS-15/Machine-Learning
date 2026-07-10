import numpy as np
import matplotlib.pyplot as plt

#===========================================
class multi_classifcation_logistic_regression_model():
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
        self.w = np.random.randn(3,3) * 0.01
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
        # bird data
        bird_x1=np.random.uniform(15,23,number_of_samples)
        bird_x2=np.random.uniform(15,23,number_of_samples)
        birds=np.column_stack((bird_x1,bird_x2,bias))
        bird_y=np.full(number_of_samples,2)
        # combine the data and labels
        self.X = np.vstack((cats, dogs,birds))
        # Convert labels to One-Hot Encoding
        labels=np.hstack((cat_y, dog_y,bird_y)).astype(int)
        self.y =np.eye(3)[labels]   # 012 =(100)(010)(001)
        self.m = len(self.X) # Store memory sample count
        
    # training the model with logistic regression algorithm
    def fit(self):
        self.last_epoch = self.epochs - 1
        self.cost_function = np.zeros(self.epochs)
        
        for i in range(self.epochs):
            z = np.dot(self.X, self.w)
            #apply the expontional function
            exp_z=np.exp(z-np.max(z,axis=1,keepdims=True))
            # apply the softmax function
            yhat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            error = self.y - yhat
            
            # Weight update using Gradient Descent
            self.w = self.w + self.eta * np.dot(self.X.T, error) / self.m
            
            # الاحتواء الرياضي السليم لمنع الانفجار الرقمي دون تشويه الحساب
            yhat_clipped = np.clip(yhat, 1e-15, 1 - 1e-15)
            self.cost_function[i] = -np.mean(np.sum(self.y * np.log(yhat_clipped), axis=1)
)
            
            # Early stopping check
            if i > 0 and abs(self.cost_function[i] - self.cost_function[i-1]) < self.epsilon:
                self.last_epoch = i
                break

    # predict the output for new data
    def predict(self, x1, x2):
        bias = 1
        x = np.array([x1, x2, bias])
        z = np.dot(x, self.w)
        exp_z=np.exp(z-np.max(z))
        self.probability = exp_z /np.sum(exp_z)
        self.yhat=np.argmax(self.probability)
        

    # accuracy of the model
    def accuracy(self):
        z = np.dot(self.X, self.w)
        exp_z = np.exp(z-np.max(z,axis=1,keepdims=True))
        yhat=exp_z/(np.sum(exp_z,axis=1,keepdims=True))
        preds = np.argmax(yhat, axis=1)
        true_labels = np.argmax(self.y, axis=1)
        self.correct = np.sum(preds == true_labels)
        self.total = len(true_labels)
        self.accuracy = self.correct / self.total * 100
        
    # draw the data on the graph
    def draw_data(self):
        x1_grid = np.linspace(0, 25, 200)
        x2_grid = np.linspace(0, 25, 200)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)#x1_mesh تكرار راسي x2_mesh تكرار افقي
        x_grid = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel(), np.ones(x1_mesh.ravel().shape[0])))

        z_grid = np.dot(x_grid, self.w)
        exp_grid = np.exp(z_grid - np.max(z_grid, axis=1, keepdims=True))
        h_grid = exp_grid / np.sum(exp_grid, axis=1, keepdims=True)   
        decision = np.argmax(h_grid, axis=1)
        decision = decision.reshape(200,200)
        plt.style.use('dark_background')
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.contourf( x1_mesh,x2_mesh,decision,alpha=0.3,levels=3,cmap="RdYlBu")
        # for loop for drow any number of class
        classes = ["Cat", "Dog", "Bird"]
        colors = ["blue", "red", "green"]
        labels = np.argmax(self.y, axis=1)
        for i in range(len(classes)):
            plt.scatter(self.X[labels==i][:,0],
                        self.X[labels==i][:,1],
                        color=colors[i],label=classes[i])
        plt.xlim(0, 25)
        plt.ylim(0, 25)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Cats and Dogs Decision Boundary')

        plt.legend()
        plt.grid()

        # draw the error graph
        plt.subplot(1,2,2)
        plt.plot(range(1, self.last_epoch+2), self.cost_function[:self.last_epoch+1], color='yellow')
        plt.title('Cost vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Categorical Cross-Entropy Cost')
        plt.grid()
        plt.show()

    # function to print the information of the model
    def print_info(self):
      classes = ["Cat", "Dog", "Bird"]

      print("=" * 50)
      print("Learning Rate (eta) : ", self.eta)
      print("Number of Epochs    : ", self.epochs)
      print("Final Epoch         : ", self.last_epoch + 1)
      print("Final Cost          : ", self.cost_function[self.last_epoch])
      print(f"Training completed in {self.last_epoch + 1} epochs.")
      print("=" * 50)

      print("Final Weights:")
      print(self.w)
      print("=" * 50)

      print("Accuracy of the model:")
      print(f"Total Samples      : {self.total}")
      print(f"Correct Predictions: {self.correct}")
      print(f"Wrong Predictions  : {self.total - self.correct}")
      print(f"Accuracy           : {self.accuracy:.2f}%")
      print("=" * 50)

      print(f"Prediction for new data point ({new_x1}, {new_x2}): {classes[self.yhat]}")
      print(f"Cat Probability : {self.probability[0] * 100:.2f}%")
      print(f"Dog Probability : {self.probability[1] * 100:.2f}%")
      print(f"Bird Probability: {self.probability[2] * 100:.2f}%")
      print("=" * 50)
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

model = multi_classifcation_logistic_regression_model(0.05, 50000, 1e-6) # زيادة قيمة الـ eta لأن الـ Sigmoid تحتاج خطوات أثبت
model.run_model(number_of_samples, new_x1, new_x2)