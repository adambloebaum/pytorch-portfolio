%matplotlib inline
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Fashion-MNIST Dataset in Numpy

# 10000 training features/targets where each feature is a greyscale image with shape (28, 28)
train_features = np.load('fashion_mnist_train_features.npy') 
train_targets = np.load('fashion_mnist_train_targets.npy')

# 1000 testing features/targets 
test_features = np.load('fashion_mnist_test_features.npy')
test_targets = np.load('fashion_mnist_test_targets.npy')

# See the shapes of training/testing datasets
print("Training Features Shape: ", train_features.shape)
print("Training Targets Shape: ", train_targets.shape)
print("Testing Features Shape: ", test_features.shape)
print("Testing Targets Shape: ", test_targets.shape)

# Visualizing the first three training features (samples)
plt.figure(figsize = (10, 10))

plt.subplot(1,3,1)
plt.imshow(train_features[0], cmap = 'Greys')

plt.subplot(1,3,2)
plt.imshow(train_features[1], cmap = 'Greys')

plt.subplot(1,3,3)
plt.imshow(train_features[2], cmap = 'Greys')

# Reshape features via flattening the images
train_features = np.reshape(train_features, (10000, 784))
test_features = np.reshape(test_features, (1000, 784))

# Define scaling function
scaler = StandardScaler()

# Scale the dataset according to standard scaling
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)

# Take the first 1000 (or randomly select 1000) training features and targets as validation set 
validation_features = train_features[:1000]
validation_targets = train_targets[:1000]

# Take the remaining 9000 training features and targets as training set 
train_features = train_features[1000:]
train_targets = train_targets[1000:]

# Reshape train/validation/test sets to conform to PyTorch's (N, Channels, Height, Width) standard for CNNs
train_features = train_features.reshape(9000, 1, 28, 28)
validation_features = validation_features.reshape(1000, 1, 28, 28)
test_features = test_features.reshape(1000, 1, 28, 28)

# Define CNN architecture
class CNNModel(torch.nn.Module):
    
    def __init__(self):
        
        super(CNNModel, self).__init__()
        
        # Convolutional layer 1
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2)
        self.relu1 = torch.nn.ReLU()
        
        # Max pooling layer 1
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Convolutional layer 2
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.relu2 = torch.nn.ReLU()
        
        # Max pooling layer 2
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fcl1 = torch.nn.Linear(2048, 100)
        self.fcl2 = torch.nn.Linear(100, 10)
    
    
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fcl1(out)
        out = self.fcl2(out)
        
        return out

# Fix the random seed so that model performance is reproducible
torch.manual_seed(55)

# Initialize your CNN model
model = CNNModel()

# Define learning rate, epoch and batchsize for mini-batch gradient
learning_rate = 0.0001
epochs = 100
batchsize = 160
num_batches = len(train_features) // batchsize

# Define loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model

# Placeholders for training loss and validation accuracy during training
train_loss_list = np.zeros((epochs,))
validation_accuracy_list = np.zeros((epochs,))

import tqdm

# Convert the training, validation, testing dataset (NumPy arrays) into torch tensors
train_features = torch.from_numpy(np.array(train_features)).float()
train_targets = torch.from_numpy(np.array(train_targets)).long()

validation_features = torch.from_numpy(np.array(validation_features)).float()
validation_targets = torch.from_numpy(np.array(validation_targets)).long()

test_features = torch.from_numpy(np.array(test_features)).float()
test_targets = torch.from_numpy(np.array(test_targets)).long()

# Define early stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training Loop
for epoch in tqdm.trange(epochs):
    
    for batch_idx in range(num_batches):
    
        batch_features = train_features[batch_idx * batchsize : (batch_idx + 1) * batchsize]
        batch_targets = train_targets[batch_idx * batchsize : (batch_idx + 1) * batchsize]
    
        # Forward pass
        optimizer.zero_grad()

        batch_pred = model(batch_features)
        loss = loss_func(batch_pred, batch_targets)
        train_loss_list[epoch] = loss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

    # Compute Validation Accuracy    
    with torch.no_grad():
        val_pred = model(validation_features)
        correct = (torch.argmax(val_pred, dim=1) == validation_targets).type(torch.FloatTensor)
        validation_accuracy_list[epoch] = correct.mean()
        
    # Check for early stopping
    if loss < best_val_loss:
        best_val_loss = loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1)

# Visualize training loss
plt.figure(figsize = (15, 9))

plt.subplot(2, 1, 1)
plt.plot(train_loss_list, linewidth = 3)
plt.ylabel("training loss")
plt.xlabel("iterations")
sns.despine()

plt.subplot(2, 1, 2)
plt.plot(validation_accuracy_list, linewidth = 3, color = 'gold')
plt.ylabel("validation accuracy")
sns.despine()

# Compute testing accuracy 
with torch.no_grad():
        test_pred = model(test_features)
        correct = (torch.argmax(test_pred, dim=1) == test_targets).type(torch.FloatTensor)        
        print("Testing Accuracy: " + str(correct.mean().numpy() * 100) + "%")
