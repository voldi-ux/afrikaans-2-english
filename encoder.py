import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate = 0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size) # the embedding layer will map each word in a language to a continuous vector representation. we will represent each word by its index in word2idx. the input_size is essentially the dictionary size of the input lang and hidden_size is the dimension of the word in a vector form. 1 by hidden_size. The iput to the embedding will be an index of a word and it will output its continous vector representation
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input): # input shape : [batch_size, sentence length] i.e [N, L]
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    