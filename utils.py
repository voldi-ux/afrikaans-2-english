import torch
import torch.nn as nn
import torch.optim as optim
import random
from Lang import EOS_token
import langs

def getDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"



def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
       input_tensor, target_tensor = data
       encoder_optimizer.zero_grad()
       decoder_optimizer.zero_grad()
       
       encoder_outputs, encoder_hidden = encoder(input_tensor)
       decoder_outputs, _,_ = decoder(encoder_hidden, target_tensor)

       loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
       loss.backward()
       
       encoder_optimizer.step()
       decoder_optimizer.step()

       total_loss += loss.item()
    
    return total_loss / len(dataloader)




def train(data_loader, encoder, decoder, n_epochs = 5, lr = 0.002, print_every = 100, plot_every = 100):
    plot_losses = []
    decoder.train()
    encoder.train()
    # print_loss_total = 0  # Reset every print_every
    # plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(data_loader, encoder, decoder, encoder_optimizer,decoder_optimizer,criterion)
        print(f"epch: {epoch} and loss: {loss}")
        plot_losses.append(loss)
    return plot_losses




def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    decoder.eval()
    encoder.eval()
    with torch.no_grad():
        input_tensor = langs.tensorFromSentence(input_lang, sentence)

        _, encoder_hidden = encoder(input_tensor)
        decoder_outputs,_ , _= decoder(encoder_hidden)

        # docoder_outputs shape: 1 * 10 * out_lang size
        _, topi = decoder_outputs.topk(1)
        decoded_indices = topi.squeeze()

        words_decoded = []

        for idx in decoded_indices:
            if idx.item() == EOS_token:
                break
            else:
                words_decoded.append(output_lang.index2word[idx.item()])
        return words_decoded



def randomEval(encoder, decoder, input_lang, output_lang, pairs, size = 5):
    for i in range(size):
      pair = random.choice(pairs)
      decoded_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
      translated_sentence = " ".join(decoded_words)
      print(f"Input : {pair[0]}")
      print(f"True output : {pair[1]}")
      print(f"predicted : {translated_sentence}")


