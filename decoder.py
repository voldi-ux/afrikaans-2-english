import torch
import torch.nn as nn
import torch.nn.functional as f
import utils
from Lang import EOS_token as eos_token, SOS_token as sos_token
from langs import MAX_LENGTH


class DecorderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecorderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size) # output size is the size of the output dictionary
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, encoder_hidden, tartget_tensor = None):
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.empty(batch_size,1, dtype=torch.long, device= utils.getDevice()).fill_(sos_token)  # shape : batch_size * 1
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        
        #we next feed one word at at time in the decoder
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output) # add predicted word to the outputs
            if tartget_tensor is not None:
                # use the next word from input tensor
                decoder_input = tartget_tensor[:, i].unsqueeze(1)  # shape : batch_size * 1
            else:
                # use its predicted word as input
                _, topi = decoder_output.topk(1)
                #topi shape: batch_size * 1 * 1
                decoder_input = topi.squeeze(-1).detach()
        #at this point the decoder_outputs has 10 predicted words
        decoder_outputs = torch.cat(decoder_outputs, dim=1) #shape: batch_size * 10 * output_size
        decoder_outputs = f.log_softmax(decoder_outputs, dim= -1) # apply 
        

        return decoder_outputs, decoder_hidden, None
    def forward_step(self, input, hidden):
        #input shape: bach_size * 1
        # hidden_shape : batch_size * hidden_size
        output = self.embedding(input) # shape: batch_size * 1 * hidden_size (embedding size)
        output = f.relu(output)
        output, hidden = self.gru(output, hidden) #hidden shape : batch_size * hidden_size
        output = self.out(output)  # shape: batch_size * 1 * output_size (target dic size)
        return output, hidden 