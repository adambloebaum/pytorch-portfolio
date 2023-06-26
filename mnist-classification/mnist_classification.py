# Import necessary packages
%matplotlib inline
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load MNIST Dataset in Numpy

# 1000 training samples where each sample feature is a greyscale image with shape (28, 28)
# 1000 training targets where each target is an integer indicating the true digit
mnist_train_features = np.load('mnist_train_features.npy') 
mnist_train_targets = np.load('mnist_train_targets.npy')

# 100 testing samples + targets
mnist_test_features = np.load('mnist_test_features.npy')
mnist_test_targets = np.load('mnist_test_targets.npy')

# Print the dimensions of training sample features/targets
print(mnist_train_features.shape, mnist_train_targets.shape)
# Print the dimensions of testing sample features/targets
print(mnist_test_features.shape, mnist_test_targets.shape)

# Visualize some training samples

plt.figure(figsize = (10, 10))

plt.subplot(1,3,1)
plt.imshow(mnist_train_features[0], cmap = 'Greys')

plt.subplot(1,3,2)
plt.imshow(mnist_train_features[1], cmap = 'Greys')

plt.subplot(1,3,3)
plt.imshow(mnist_train_features[2], cmap = 'Greys')

# Reshape features via flattening the images
mnist_train_features = np.reshape(mnist_train_features, (1000, 784))
mnist_test_features = np.reshape(mnist_test_features, (100, 784))

# Scale the dataset according to standard scaling
scaler = StandardScaler()

mnist_train_features = scaler.fit_transform(mnist_train_features)
mnist_test_features = scaler.fit_transform(mnist_test_features)

# Split training dataset into Train (90%), Validation (10%)
mnist_train_features, mnist_validation_features, mnist_train_targets, mnist_validation_targets = train_test_split(mnist_train_features, mnist_train_targets, test_size=0.1, random_state=2)

# Define model
class mnistClassification(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim):
        
        super(mnistClassification, self).__init__()
        
        # Layer creation
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim2)
        self.layer3 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.layer4 = torch.nn.Linear(hidden_dim3, output_dim)
        
    def forward(self, x):
        
        # Forward propagation
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        out = self.layer4(x)
        
        return out

# Initialize neural network model with input and output dimensions
model = mnistClassification(784, 1000, 400, 100, 10)

# Define the learning rate and epoch 
learning_rate = 0.0008
epochs = 100

# Define loss function and optimizer
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model

# Identify tracked values
train_loss_list = np.zeros((epochs,))
validation_accuracy_list = np.zeros((epochs,))

import tqdm

# Convert the training, validation, testing dataset (NumPy arrays) into torch tensors
mnist_train_features = torch.from_numpy(mnist_train_features).float()
mnist_train_targets = torch.from_numpy(mnist_train_targets).long()
mnist_validation_features = torch.from_numpy(mnist_validation_features).float()
mnist_validation_targets = torch.from_numpy(mnist_validation_targets).long()

# Training Loop

for epoch in tqdm.trange(epochs):
    
    # Initialize gradients
    optimizer.zero_grad()
    
    y_pred = model(mnist_train_features)
    loss = loss_func(y_pred, mnist_train_targets)
    train_loss_list[epoch] = loss.item()
    
    # Zero gradients
    loss.backward()
    optimizer.step()
    
    # Compute Validation Accuracy
    with torch.no_grad():
        y_pred = model(mnist_validation_features)
        correct = (torch.argmax(y_pred, dim=1) == mnist_validation_targets).type(torch.FloatTensor)
        validation_accuracy_list[epoch] = correct.mean()

import seaborn as sns

# Visualize training loss

plt.figure(figsize = (12, 6))

# Visualize training loss with respect to iterations (1 iteration -> single batch)
plt.subplot(2, 1, 1)
plt.plot(train_loss_list, linewidth = 3)
plt.ylabel("training loss")
plt.xlabel("epochs")
sns.despine()

# Visualize validation accuracy with respect to epochs
plt.subplot(2, 1, 2)
plt.plot(validation_accuracy_list, linewidth = 3, color = 'gold')
plt.ylabel("validation accuracy")
sns.despine()

# Compute the testing accuracy 
with torch.no_grad():
    y_pred_test = model(mnist_validation_features)
    predicted_labels = torch.argmax(y_pred_test, dim=1)
    correct = (torch.argmax(y_pred_test, dim=1) == mnist_validation_targets).type(torch.FloatTensor)
    print("Testing Accuracy: " + str(correct.mean().numpy() * 100) + "%")
