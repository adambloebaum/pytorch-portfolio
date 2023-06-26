%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

data_size_to_train = 10000

# Load the Sherlock Holmes data up to data_size_to_train
data = open('sherlock.txt', 'r').read()[4000:data_size_to_train + 4000]

# Find the set of unique characters within the training data
characters = sorted(list(set(data)))

# total number of characters in the training data and number of unique characters
data_size, vocab_size = len(data), len(characters)

print("Data has {} characters, {} unique".format(data_size, vocab_size))

# Use Python Dictionary to map the characters to numbers and vice versa
character_to_num = {ch:i for i,ch in enumerate(characters)}
num_to_character = {i:ch for i,ch in enumerate(characters)}

# Use the character_to_num dictionary to map each character in the training dataset to a number
data = list(data)
for i,ch in enumerate(data):
    data[i] = character_to_num[ch]

# Define model
class CharRNN(torch.nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size, num_layers, output_size):
        
        super(CharRNN, self).__init__()
        
        # Embedded layer
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        
        # RNN layer
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='relu')
        
        # Decoder layer
        self.decoder = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        
        # Forward pass
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        
        return output, hidden_state.detach()

# Fix random seed
torch.manual_seed(25)

# Define RNN network
rnn = CharRNN(num_embeddings=vocab_size, embedding_dim=100, input_size=100, hidden_size=512, num_layers=3, output_size=vocab_size)

# Define learning rate and epochs
learning_rate = 0.001
epochs = 50

# Size of the input sequence to be used during training and validation
training_sequence_len = 50
validation_sequence_len = 200 

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

rnn

# Tracking training loss per each input/target sequence fwd/bwd pass
train_loss_list = []

# Convert training data into torch tensor and make it into vertical orientation (N, 1)
data = torch.unsqueeze(torch.tensor(data), dim=1)

# Training Loop
for epoch in range(epochs):
    
    # Initializing values
    character_loc = np.random.randint(100)
    iteration = 0
    hidden_state = None
    
    while character_loc + training_sequence_len + 1 < data_size:
        
        # Inidice selection
        input_seq = data[character_loc : character_loc + training_sequence_len]
        target_seq = data[character_loc + 1 : character_loc + training_sequence_len + 1]
        
        output, hidden_state = rnn(input_seq, hidden_state)
        
        # Training loss
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        train_loss_list.append(loss.item())
        
        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step for loop
        character_loc += training_sequence_len
        iteration += 1
    
    print("Averaged Training Loss for Epoch ", epoch, ": ", np.mean(train_loss_list[-iteration:]))
    
    # Sample and generate a text sequence after every epoch

    # Initializing values
    character_loc = 0
    hidden_state = None
    
    # Indice selection
    rand_index = np.random.randint(data_size-1)
    input_seq = data[rand_index : rand_index+1]
    
    print("----------------------------------------")
    
    with torch.no_grad():
        
        while character_loc < validation_sequence_len:
            
            output, hidden_state = rnn(input_seq, hidden_state)
            
            output = torch.nn.functional.softmax(torch.squeeze(output), dim=0)
            character_distribution = torch.distributions.Categorical(output)
            character_num = character_distribution.sample()
            
            print(num_to_character[character_num.item()], end='')
            
            input_seq[0][0] = character_num.item()
            
            # Step for loop
            character_loc += 1

    print("\n----------------------------------------")

import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2.5)

# Plot the training loss and rolling mean training loss with respect to iterations
plt.figure(figsize = (15, 9))

plt.plot(train_loss_list, linewidth = 3, label = 'Training Loss')
plt.plot(np.convolve(train_loss_list, np.ones(100), 'valid') / 100, 
         linewidth = 3, label = 'Rolling Averaged Training Loss')
plt.ylabel("training loss")
plt.xlabel("Iterations")
plt.legend()
sns.despine()
