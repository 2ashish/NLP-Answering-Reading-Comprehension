from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format("/home/pratik/glove/glove.6B.50d.txt.word2vec", binary=False)
def get_char_embedding(model):
    words = list(model.vocab.keys())
    cembedding = {}
    vectors = {}
    for w in words:
        for char in w:
            if ord(char) < 128:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + model.wv[w],
                                     vectors[char][1] + 1)
                else:
                    vectors[char] = (model.wv[w], 1)
    for c in vectors:
        cembedding[c] = np.round((vectors[c][0] / vectors[c][1]), 6).tolist()
    return cembedding

embe = get_char_embedding(model)
import pickle
with open("char_embeddings.pickle", "wb") as output_file:
    pickle.dump(embe, output_file)
# np.save('char_embeddings',embe)