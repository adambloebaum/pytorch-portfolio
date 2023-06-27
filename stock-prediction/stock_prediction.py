%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns

# Seaborn plot styling
sns.set(style = 'white', font_scale = 2)

# Load stock dataset
tesla = pd.read_csv('TSLA.csv') 
tesla_np = tesla.to_numpy()

# Take a look at the format of the data
pd.DataFrame.head(tesla)

# Get the column of index 4 for closing data
#Format the data to be used for training and plot it.
data = tesla_np[:,4]
print(data.shape)
plt.plot(data)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Normalize data before passing it through the encoder-decoder.
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1,1))

# Create method to split the data into training and testing datasets in addition to breaking it up into iteratively stepped subsequences for the encoder.
def generate_input_output_seqs(y, encoder_inputseq_len, decoder_outputseq_len, stride = 1, num_features = 1):
  
    L = y.shape[0] # Length of y
    
    # Calculate how many input/target sequences there will be based on the parameters and stride
    num_samples = (L - encoder_inputseq_len - decoder_outputseq_len) // stride + 1
    
    # Numpy zeros arrray to contain the input/target sequences
    train_input_seqs = np.zeros([num_samples, encoder_inputseq_len, num_features])
    train_output_seqs = np.zeros([num_samples, decoder_outputseq_len, num_features])    
    
    # Iteratively fill in train_input_seqs and train_output_seqs
    for ff in np.arange(num_features):        
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + encoder_inputseq_len
            train_input_seqs[ii, :, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + encoder_inputseq_len
            end_y = start_y + decoder_outputseq_len 
            train_output_seqs[ii, :, ff] = y[start_y:end_y, ff]

    return train_input_seqs, train_output_seqs

# Define encoder input sequence length, decoder output sequence length and testing sequence length
encoder_inputseq_len = 15
decoder_outputseq_len = 6
testing_sequence_len = 50

# Use all but the last testing_sequence_len elements of our data for training.
y_train = data[:-testing_sequence_len]

# Plot transformed data
plt.plot(y_train, linewidth = 3, color = 'grey')
sns.despine()

# Generate the input and output sequences to train the encoder-decoder.
train_input_seqs, train_output_seqs = generate_input_output_seqs(y_train, encoder_inputseq_len, decoder_outputseq_len)

# Make sure train_input_seqs and train_output_seqs have correct dimensions as expected
print("Encoder Training Inputs Shape: ", train_input_seqs.shape)
print("Decoder Training Outputs Shape: ", train_output_seqs.shape)

# Define the encoder-decoder in two separate classes, then combine the two in a single superclass implementation.
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        
        super(Encoder, self).__init__()

        # Use LSTM for Encoder with batch_first = True
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                                  num_layers = num_layers, 
                                  batch_first = True)
        
    def forward(self, input_seq, hidden_state):
        
        out, hidden = self.lstm(input_seq, hidden_state)
        
        return out, hidden     

class Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        
        super(Decoder, self).__init__()
      
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                                  num_layers = num_layers,
                                  batch_first = True)
      
        self.fc_decoder = nn.Linear(hidden_size, output_size)  

    def forward(self, input_seq, encoder_hidden_states):
        
        out, hidden = self.lstm(input_seq, encoder_hidden_states)
        output = self.fc_decoder(out)     
        
        return output, hidden

class Encoder_Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, decoder_output_size, num_layers):

        super(Encoder_Decoder, self).__init__()

        self.Encoder = Encoder(input_size = input_size, hidden_size = hidden_size, 
                               num_layers = num_layers)
        
        self.Decoder = Decoder(input_size = input_size, hidden_size = hidden_size, 
                               output_size = decoder_output_size, num_layers = num_layers)

# Set a seed to get the model started
torch.manual_seed(2)

# Define the model
Encoder_Decoder_RNN = Encoder_Decoder(input_size = 1, hidden_size = 30, 
                                      decoder_output_size = 1, num_layers = 1)

# Set the learning rate and other hyperparameters
learning_rate = 0.00001   
epochs = 20

batchsize = 5
num_features = train_output_seqs.shape[2]

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(Encoder_Decoder_RNN.parameters(), lr=learning_rate)

Encoder_Decoder_RNN

# Identify tracked values
train_loss_list = []

# Convert training dataset into torch tensors
train_input_seqs_final = torch.from_numpy(train_input_seqs).float()
train_output_seqs_final = torch.from_numpy(train_output_seqs).float()

# Split the training dataset to mini-batches
train_batches_features = torch.split(train_input_seqs_final, batchsize)[:-1]
train_batches_targets = torch.split(train_output_seqs_final, batchsize)[:-1]

# Total number of mini-batches in the training set
batch_split_num = len(train_batches_features)

for epoch in range(epochs): # For each epoch
    
    for k in range(batch_split_num): # For each mini_batch
        
        # Initialize hidden states to Encoder
        hidden_state = None
        
        # Initialize empty torch tensor array to store decoder output sequence
        decoder_output_seq = torch.zeros(batchsize, decoder_outputseq_len, num_features)
        
        # Empty gradient buffer
        optimizer.zero_grad()
        
        # Feed k-th mini-batch for encoder input sequences to encoder with hidden state
        encoder_output, encoder_hidden = Encoder_Decoder_RNN.Encoder(train_batches_features[k], hidden_state)
        # Re-define the resulting encoder hidden states as input hidden states to decoder
        decoder_hidden = encoder_hidden
        
        # Initial input to decoder is the last timestep feature from the encoder input sequence
        decoder_input = train_batches_features[k][:, -1, :]
        decoder_input = torch.unsqueeze(decoder_input, 2)
        
        # Populating the decoder output sequence
        for t in range(decoder_outputseq_len): # for each timestep in output sequence
            
            decoder_output, decoder_hidden = Encoder_Decoder_RNN.Decoder(decoder_input, decoder_hidden)
            decoder_output_seq[:, t, :] = torch.squeeze(decoder_output, 2)
            decoder_input = train_batches_features[k][:, t, :]            
            decoder_input = torch.unsqueeze(decoder_input, 2)
        
        loss = loss_func(torch.squeeze(decoder_output_seq), torch.squeeze(train_batches_targets[k]))
        
        train_loss_list.append(loss.item())
        
        # Backwards pass
        loss.backward()
        optimizer.step()
    
    print("Averaged Training Loss for Epoch ", epoch,": ", np.mean(train_loss_list[-batch_split_num:]))

# Visualize and evaluate the model
plt.figure(figsize = (12, 7))

plt.plot(np.convolve(train_loss_list, np.ones(100), 'valid') / 100, 
         linewidth = 3, label = 'Rolling Averaged Training Loss')
plt.ylabel("training loss")
plt.xlabel("Iterations")
plt.legend()
sns.despine()

# Define the testing sequence, which comprises the last 100 data points in this case.
test_input_seq = data[-testing_sequence_len:]

# Visualize the testing sequence
plt.figure(figsize = (10, 5))
plt.plot(test_input_seq, linewidth = 3)
plt.title('Test Sequence')
sns.despine()

# Generate signal predictions for testing sequence with trained Encoder-Decoder

# Convert test sequence to tensor
test_input_seq = torch.from_numpy(test_input_seq).float()

# Initialize empty torch tensor array to store decoder output sequence
decoder_output_seq = torch.zeros(testing_sequence_len, num_features)

# First n-datapoints in decoder output sequence = First n-datapoints in ground truth test sequence
# n = encoder_input_seq_len
decoder_output_seq[:encoder_inputseq_len] = test_input_seq[:encoder_inputseq_len]

# Initialize index for prediction
pred_start_ind = 0

# Activate no_grad() since we aren't performing backprop
with torch.no_grad():
    
    # Loop continues until the RNN prediction reaches the end of the testing sequence length
    while pred_start_ind + encoder_inputseq_len + decoder_outputseq_len < testing_sequence_len:
        
        # Initialize hidden state for encoder
        hidden_state = None
        
        # Define the input to encoder
        input_test_seq = decoder_output_seq[pred_start_ind:pred_start_ind + encoder_inputseq_len]
        # Add dimension to first dimension to keep the input (sample_size, seq_len, # of features/timestep)
        input_test_seq = torch.unsqueeze(input_test_seq, 0)
        
        # Feed the input to encoder and set resulting hidden states as input hidden states to decoder
        encoder_output, encoder_hidden = Encoder_Decoder_RNN.Encoder(input_test_seq, hidden_state)
        decoder_hidden = encoder_hidden
        
        # Initial input to decoder is last timestep feature from the encoder input sequence 
        decoder_input = input_test_seq[:, -1, :]
        # Add dimension to keep the input (sample_size, seq_len, # of features/timestep)
        decoder_input = torch.unsqueeze(decoder_input, 2)
        
        # Populate decoder output sequence
        for t in range(decoder_outputseq_len):
            
            # Generate new output for timestep t
            decoder_output, decoder_hidden = Encoder_Decoder_RNN.Decoder(decoder_input, decoder_hidden)
            # Populate the corresponding timestep in decoder output sequence
            decoder_output_seq[pred_start_ind + encoder_inputseq_len + t] = torch.squeeze(decoder_output)
            # Use the output of the decoder as new input for the next timestep using the teacher forcing method
            decoder_input = test_input_seq[pred_start_ind + encoder_inputseq_len + t]
            decoder_input = torch.reshape(decoder_input, (1, 1, 1))
        
        # Update pred_start_ind
        pred_start_ind += decoder_outputseq_len

# Visualize predicted stock sequence vs the ground truth
plt.figure(figsize = (10, 5))
plt.plot(test_input_seq, linewidth = 3, label = 'GroundTruth')
plt.plot(decoder_output_seq, linewidth = 3, label = 'RNN Predicted')
plt.title('RNN Predicted vs GroundTruth')
plt.legend()
sns.despine()

# Compute the MSE error between test_input_seq and decoder_output_seq and print the value as Test MSE Error
MSE_loss =  loss_func(test_input_seq, decoder_output_seq)
print("Test mean standard error: ", MSE_loss.item())
