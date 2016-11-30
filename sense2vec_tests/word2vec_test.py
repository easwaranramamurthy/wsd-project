import os
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Corpus(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename, encoding = "utf8"):
            if not line.startswith("<doc id="):
                yield line.split()
 
wikiSentences = Corpus('C://Users/Easwaran/Desktop/wikipedia_sentences_tokenised.txt')

model = gensim.models.Word2Vec(wikiSentences, size = 500, window = 10, negative = 10, hs = 0, sample = 1e-5, iter = 3, min_count = 10)