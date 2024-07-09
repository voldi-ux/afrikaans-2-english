
SOS_token = 0 # end of line/word token
EOS_token = 1 # start of line/word token


# this class will represent a single language together with all of its words and dictionary size
class Lang:
    def __init__(self, name):
        self.name  = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.lang_size = 2 # the number of words in a lnaguage, the 2 are for the start and end token
    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
          self.word2index[word] = self.lang_size # we need to increment this to reflect the edition of a new word
          self.word2count[word] = 1
          self.index2word[self.lang_size] = word
          self.lang_size += 1 # increase the size by one for each new word added
        else :
         self.word2count[word] += 1

