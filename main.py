from decoder import DecorderRNN
from encoder import EncoderRNN
from langs import get_dataloader
import utils 
import matplotlib.pyplot as plt

hidden_size = 100
batch_size = 16

input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.lang_size, hidden_size).to(utils.getDevice())
decoder = DecorderRNN(hidden_size, output_lang.lang_size).to(utils.getDevice())

losses = utils.train(train_dataloader, encoder, decoder, 120, print_every=5, plot_every=5)

plt.plot(losses)
plt.show()


utils.randomEval(encoder, decoder, input_lang, output_lang, pairs)
