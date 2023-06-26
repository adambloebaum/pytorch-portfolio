# Import neccessary packages
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import Image # For displaying images in colab jupyter cell
from sklearn.datasets import load_iris

# iris dataset is available from scikit-learn package
iris = load_iris()

# Load the X (features) and y (targets) for training
X_train = iris['data']
y_train = iris['target']

# Load the name labels for features and targets
feature_names = iris['feature_names']
names = iris['target_names']

# Print the first 10 training samples for both features and targets
print(X_train[:10, :], y_train[:10]) 

# Print the dimensions of features and targets
print(X_train.shape, y_train.shape)

# Visualize the dataset before training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for target, target_name in enumerate(names):
    
    # Subset the rows of X_train that fall into each flower category using boolean mapping
    X_plot = X_train[y_train == target]
    
    # Plot the sepal length versus sepal width for the flower category
    ax1.plot(X_plot[:, 0], X_plot[:, 1], linestyle='none', marker='o', label=target_name)

# Label the plot
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.axis('equal')
ax1.legend()

# Repeat the above process but with petal length versus petal width
for target, target_name in enumerate(names):
    
    X_plot = X_train[y_train == target]
    
    ax2.plot(X_plot[:, 2], X_plot[:, 3], linestyle='none', marker='o', label=target_name)
    
ax2.set_xlabel(feature_names[2])
ax2.set_ylabel(feature_names[3])
ax2.axis('equal')
ax2.legend()

# Define model
class irisClassification(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(irisClassification, self).__init__()
        
        # Layer creation
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        
        # Forward propagation across both layers
        out = self.layer1(x)
        out = self.layer2(out)
        
        return out

# Max accuracy found through hidden_dem = 7 and a learning rate of 0.009
model = irisClassification(4, 7, 1)
learning_rate = 0.009
epochs  = 50

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Follow models performance over each epoch
train_loss_list = []

inputs = torch.from_numpy(X_train).float()
targets = torch.from_numpy(y_train).float()

# Reshape targets to match outputs
targets = targets.view(150, 1)

for epoch in range(epochs):
    
    optimizer.zero_grad()
    
    # Compute output values from model
    outputs = model(inputs)
    
    loss = loss_func(outputs, targets)
    
    # Add loss value to the training loss list
    train_loss_list.append(loss.item())
    
    # Backwards propagation across layers
    loss.backward()
    
    optimizer.step()
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Plot training loss throughout the training
plt.figure(figsize=(12, 6))
plt.plot(train_loss_list)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('MSE Loss per Epoch')

# Confirm that model's training accuracy is >90%
with torch.no_grad():
    
    # Compare model predictions with targets (y_train) to compute the training accuracy         
    predicted = model(torch.from_numpy(X_train).float()).numpy()
    predicted = np.round(predicted)
            
# Initializing number of correct predictions and samples
correct = 0
samples = 0

# Looping over predicted
for i in range(len(predicted)):
    
    # Checking if predicted value and actual value are equal
    if predicted[i] == y_train[i]:
        # Correct
        correct += 1
        samples += 1
    else:
        # Incorrect
        samples += 1

# Calculating training accuracy
training_accuracy = correct / samples

print(training_accuracy)




