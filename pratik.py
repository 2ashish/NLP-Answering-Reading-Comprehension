
# coding: utf-8

# In[16]:


from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import math
import os
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity 
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import sent_tokenize
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.text import Tokenizer
tokenizer = RegexpTokenizer(r'[^\s]+')

# In[41]:


context_file = open(os.path.join('./data/', 'train_context'), 'r')
c = context_file.read()
context = re.split('\n' + '-' + '\n', c)
del c

question_file = open(os.path.join('./data/', 'train_question'), 'r')
c = question_file.read()
questions = re.split('\n' + '-' + '\n', c)
del c

answer_file = open(os.path.join('./data/', 'train_answer'), 'r')
c = answer_file.read()
answers = re.split('\n' + '-' + '\n', c)
del c

span_file = open(os.path.join('./data/', 'train_span'), 'r')
c = span_file.read()
spa = re.split('\n' + '-' + '\n', c)
del c


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
MAX_SEQUENCE_LENGTH = 200

MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 50
MAX_QUE_LENGTH = EMBEDDING_DIM
VALIDATION_SPLIT = 0.8
NUMCONTEXT = 100



# In[18]:


print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


def vectorize_stories(inp,que,ans):
    inputs, queries, answers = [], [], []
    for i in range(0,len(inp)):
        inputs.append([word_idx[w] for w in inp[i]])
        queries.append([word_idx[w] for w in que[i]])
        # answers.append(ans)
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(ans))

def para_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    spans = tokenizer.span_tokenize(data)
    sp = [span for span in spans]
    return x_tokens,sp
    #return tokens

def que_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    return x_tokens
# In[34]:


context = context[0:NUMCONTEXT]

inp = []
que = []
ans = []
i =0
for c in context:    
    q=questions[i]
    a=answers[i]
    all_ques = re.split('\n', q)
    all_ans = re.split('\n', a)
    all_s = re.split('\n', spa[i])
    for j in range (0,len(all_ques)):
        inp.append(c)
        que.append(all_ques[j])
        ans.append(all_ans[j])
    i+=1

print(len(inp))


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(inp)
tokenizer.fit_on_texts(que)
sequences = tokenizer.texts_to_sequences(inp)
sequences1 = tokenizer.texts_to_sequences(que)
answer1 = tokenizer.texts_to_sequences(ans)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

train_con = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
train_que = pad_sequences(sequences1, maxlen=MAX_QUE_LENGTH)
print(train_con[0])
print(train_que[0])
# print(answer1)

split = int(NUMCONTEXT*VALIDATION_SPLIT)
train_context = train_con[0:split]
val_context = train_con[split+1:NUMCONTEXT]
train_question = train_que[0:split]
val_question = train_que[split+1:NUMCONTEXT]
train_answer = answer1[0:split]
val_answer = answer1[split+1:NUMCONTEXT]


# In[38]:


num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
# print(embedding_matrix.shape)


# In[42]:


input_sequence = Input((MAX_SEQUENCE_LENGTH,))
question = Input((MAX_QUE_LENGTH,))

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
# input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=100))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(num_words,
                            MAX_QUE_LENGTH,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_QUE_LENGTH,
                            trainable=False))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer = LSTM(32)(answer)  # (samples, 32)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)
# one regularization layer -- more would probably be needed.
# answer1 = Dropout(0.3)(answer)
# answer1 = Dense(vocab_size)(answer1)  # (samples, vocab_size)
# # we output a probability distribution over the vocabulary
# answer1 = Activation('softmax')(answer1)
# # build the final model

# answer2 = Dropout(0.3)(answer)
# answer2 = Dense(vocab_size)(answer2)  # (samples, vocab_size)
# # we output a probability distribution over the vocabulary
# answer2 = Activation('softmax')(answer2)

# model = Model(inputs=[input_sequence, question], outputs=[answer1])
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# print(answers_train.shape)


# In[8]:


model.fit([inputs_train, queries_train], [anstrain1],
          batch_size=32,
          epochs=10,
          validation_data=([inputs_test, queries_test], [anstest1]))

