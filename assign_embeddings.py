import gensim
from gensim.models import KeyedVectors

class WordEmbedding:
    def __init__(self, file_path, limit=1000000):
        self.file_path = file_path
        self.limit = limit
        self.model = self.load_word_embedding()

    def load_word_embedding(self):
        return KeyedVectors.load_word2vec_format(self.file_path, binary=True, limit=self.limit)
