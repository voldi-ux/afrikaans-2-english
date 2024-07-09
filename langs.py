
import re
import unicodedata
from Lang import Lang
from Lang import EOS_token, SOS_token
import torch
import utils 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)  # add white space before each punctuation mark where \1 is the first capturing group and is either . ! or ? e.g voldi! becomes voldi !
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()



def readLangs(lang1, lang2, reverse = False):
    print("reading lines ...")
    #read the file and split into  lines
    lines = open(f"data/{lang1}-{lang2}.txt", encoding="utf-8").read().strip().split("\n") # red the file and split the file content into lines

    #next we split every line into pairs
    pairs = [ [normalizeString(line.split("\t")[0]), normalizeString(line.split("\t")[1])] for line in lines]

    if reverse:
        pairs = [pair.reverse() or pair for pair in pairs] # reverse the pair setences lang1 -> lang2 becomes lang2 -> lang1 # revese peforms an inplace reverse operation and returns None so None or pair will always resolve to pair
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

# l1, l2, pairs  = readLangs("eng", "afr", True)


MAX_LENGTH = 15; # the maximum sentence length allowed to be trained for both languages


# English will be our target language in this case
# we don't want the sentences to equal max_length because we will append an end-of-sentence to each sentence
def filterPair(pair):
    return len(pair[0].split(" ")) <  MAX_LENGTH and len(pair[1].split(" ")) <  MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



def prepareData(lang1, lang2, reverse = False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print(f"number of sentence pairs is :  {len(pairs)}")
    pairs = filterPairs(pairs)

    print(f"Number of sentence pairs after filtering is : {len(pairs)}")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print(f'{input_lang.name} has {input_lang.lang_size - 2} words ')
    print(f'{output_lang.name} has {output_lang.lang_size - 2 } words')

    return input_lang, output_lang, pairs




def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]

def tensorFromSentence(lang, sentence):
    indices = indexesFromSentence(lang, sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device= utils.getDevice()).view(1,-1) # returns a 1 by sentence length tensor



def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData("eng", "afr", True)
    n = len(pairs)
    print(n)
    input_words_indices = torch.zeros((n, MAX_LENGTH), dtype=torch.long)
    output_words_indices = torch.zeros((n, MAX_LENGTH), dtype=torch.long) # each row represents a sentence and entry in the row is a word in that language
    
    for idx , (inp_word, out_word) in enumerate(pairs):
        inp_word_indices = indexesFromSentence(input_lang, inp_word)
        out_word_indices = indexesFromSentence(output_lang, out_word)
        # we append an end of sentence token/ indicator
        inp_word_indices.append(EOS_token)
        out_word_indices.append(EOS_token)
        
        # update : insert/ inject sentence in each row. idx = row number
        input_words_indices[idx, : len(inp_word_indices)]  = torch.tensor(inp_word_indices )
        output_words_indices[idx, : len(out_word_indices)]  = torch.tensor(out_word_indices) 

    train_data = TensorDataset(input_words_indices, output_words_indices)
    sampler = RandomSampler(train_data) # will allow us to randomly select sentence pairs
    train_loader = DataLoader(train_data, sampler=None, batch_size=batch_size)
    return input_lang, output_lang, train_loader, pairs


