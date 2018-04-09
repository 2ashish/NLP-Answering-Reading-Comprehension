import sklearn
import numpy as np
import re
import nltk
import json
import os
import sys
import gensim
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity 
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import sent_tokenize
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
# sbstemmer = SnowballStemmer('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# glove_input_file = './glove/glove.6B.50d.txt'
# word2vec_output_file = './glove/glove.6B.50d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)
# word2vec_output_file = './glove/glove.6B.50d.txt.word2vec'
# glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


word2vec_output_file = './glove/glove.6B.50d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = 50

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def our_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    stop_words = set(stopwords.words('english'))
    tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens if not w in stop_words]
    return tokens

context_file = open(os.path.join('./data/', 'train_context'), 'r')
c = context_file.read()
context = re.split('\n' + '-' + '\n', c)
del c

question_file = open(os.path.join('./data/', 'train_question'), 'r')
c = question_file.read()
questions = re.split('\n' + '-' + '\n', c)
del c

Average_vector = MeanEmbeddingVectorizer(glove_model)

c = context[1]
q = questions[1]

all_sent = sent_tokenize(c)
all_ques = re.split('\n', q)

sent_tokens = []
for sent in all_sent:
    tokens = our_tokenizer(sent)
    sent_tokens.append(tokens)

sent_vec = Average_vector.transform(sent_tokens)


que_tokens = []
for que in all_ques:
    tokens = our_tokenizer(que)
    que_tokens.append(tokens)

ques_vec = Average_vector.transform(que_tokens)

x = cosine_similarity(ques_vec,sent_vec)
print(all_sent)
print(all_ques)
for i in range(len(x)):
    index = np.argmax(x[i])
    print(x[i],index)
# def average_rep(x_train, q_data, model):
#     x_avg = []
#     q_avg = []
#     avg = []
#     for x in x_train:
#         x_tokens = nltk.word_tokenize(x)
#         x_vec = []
#         for x_token in x_tokens:
#             if x_token in model.wv.vocab:
#                 x_vec.append(model.wv[x_token])
#         x_avg.append([a/len(x_vec) for a in sum(x_vec)])
#     for ques in q_data:
#         q_vec = []
#         for q in ques:
#             q_tokens = nltk.word_tokenize(q)
#             for q_token in q_tokens:
#                 if q_token in model.wv.vocab:
#                     q_vec.append(model.wv[q_token])
#         q_avg.append([a/len(q_vec) for a in sum(q_vec)])
#     for i in range(len(x_avg)):
#         avg.append([(a+b)/2 for (a,b) in zip(x_avg[i], q_avg[i])])
#     return avg


# # In[14]:


# get_data(squad_dataset)


# # In[170]:


# def remove_newlines(text):
#     if text[0] == '\n':
#         text = text[-len(text)+1:]
#     elif text[-1] == '\n':
#         text = text[:len(text)-1]
#     return text

# def get_cdata(context_file):
#     c = context_file.read()
#     lines = re.split('\n' + '-'*30 + '\n', c)[:-1]
#     return lines

# def get_qdata(xfile):
#     final = []
#     line = []
#     for l in xfile:
#         l = remove_newlines(l)
#         if l != '-'*30:
#             line.append(l)
#         else:
#             final.append(line)
#             line = []
#     return final


# # In[171]:


# context_file = open('./data/train_context', 'r')
# questions_file = open('./data/train_question', 'r')
# context_data = get_cdata(context_file)
# questions_data = get_qdata(questions_file)


# # In[172]:


# glove_rep = average_rep(context_data[:2500], question_data[:2500], glove_model)


# # In[105]:


# # Check lengths
# tot = 0
# for data_id in range(len(squad_dataset['data'])):
#     data = squad_dataset['data'][data_id]
#     for para_id in range(len(data['paragraphs'])):
#         para = data['paragraphs'][para_id]
#         tot += 1

# print(tot)
# print(c_tot)
# print(len(context_data))
# print(len(questions_data))

