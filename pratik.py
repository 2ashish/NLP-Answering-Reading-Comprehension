
# coding: utf-8

# In[1]:


from __future__ import print_function

import numpy as np
from keras import backend as K
import keras
from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, Masking, Dropout, Flatten, Activation, Reshape, Lambda, Permute, merge, multiply, concatenate
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers.pooling import GlobalMaxPooling1D
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
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
from keras.preprocessing.text import text_to_word_sequence


# In[19]:


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
MAX_SEQUENCE_LENGTH = 500

MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 50
MAX_QUE_LENGTH = EMBEDDING_DIM
VALIDATION_SPLIT = 0.8
NUMCONTEXT = 1000


# In[3]:


print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

import pickle

with open("char_embeddings.pickle","rb") as fd:
    char_embeddings = pickle.load(fd)
    
def get_char_embedding(word):
    x = np.zeros(EMBEDDING_DIM)
    count = 0
    for i in range(len(word)):
        try:
            count = count +1
            temp = np.asarray(char_embeddings[word[i]])
        except:
            temp = np.zeros(EMBEDDING_DIM)
        x = x+temp
    return x/count


# In[46]:


# print(embeddings_index['bhangale'])
# print(type(get_char_embedding('bhangale')))


# In[20]:


tokenizer = RegexpTokenizer(r'[^\s]+')

def vectorize_stories(inp,que,ans):
    inputs, queries, answers = [], [], []
    for i in range(0,len(inp)):
        inputs.append([word_index[w] for w in inp[i]])
        queries.append([word_index[w] for w in que[i]])
        # answers.append(ans)
    return (pad_sequences(inputs, maxlen=MAX_SEQUENCE_LENGTH,padding='post'),
            pad_sequences(queries, maxlen=MAX_QUE_LENGTH,padding='post'),
            np.array(ans))

def para_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    spans = tokenizer.span_tokenize(data)
    sp = [span for span in spans]
    return x_tokens,sp

def que_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    return x_tokens


context = context[0:NUMCONTEXT]

inp = []
que = []
ans = []
i =0
for c in context:
    tokens,sp = para_tokenizer(c)
    
    q=questions[i]
    a=answers[i]
    all_ques = re.split('\n', q)
    all_ans = re.split('\n', a)
    all_s = re.split('\n', spa[i])
    for j in range (0,len(all_ques)):
        inp.append(tokens)
        x = re.split(',',all_s[j])
        x = list(map(int, x))
        k = 0
        for span in sp:
            if span[0] <= x[0] <= span[1]:
                st = k
            if span[0] <= x[1] <= span[1]:
                en = k
            k+=1
        que.append(que_tokenizer(all_ques[j]))
        ans.append([st,en])
        #ans.append(st)
    i+=1

print(len(inp))
# print(inp[0])
# print(que[0])
# print(ans[1])


vocab = set()
for i in range(0,len(inp)):
    vocab |= set(inp[i] + que[i])
vocab = sorted(vocab)
print(len(vocab))

vocab_size = len(vocab) + 1
# story_maxlen = max(map(len, (x for x in inp)))
# query_maxlen = max(map(len, (x for x in que)))
# print(story_maxlen,query_maxlen)

word_index = dict((c, i + 1) for i, c in enumerate(vocab))
index_word = dict((i+1, c) for i, c in enumerate(vocab))
train_con, train_que, answers = vectorize_stories(inp,que,ans)
train_ans_start = to_categorical(answers[:,0],MAX_SEQUENCE_LENGTH)
train_ans_end = to_categorical(answers[:,1],MAX_SEQUENCE_LENGTH)

split = int(NUMCONTEXT*VALIDATION_SPLIT)
train_context = train_con[0:split]
val_context = train_con[split+1:NUMCONTEXT]
train_question = train_que[0:split]
val_question = train_que[split+1:NUMCONTEXT]
train_answer_start = train_ans_start[0:split]
val_answer_start = train_ans_start[split+1:NUMCONTEXT]
train_answer_end = train_ans_end[0:split]
val_answer_end = train_ans_end[split+1:NUMCONTEXT]


# In[57]:


# with open('context_char.pickle','wb') as fd:
#     pickle.dump(chr_embedded_context,fd)


# In[21]:



num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
#     print(word,i)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = get_char_embedding(word)
print(embedding_matrix.shape)


# In[22]:


W = EMBEDDING_DIM
N = MAX_SEQUENCE_LENGTH
M = MAX_QUE_LENGTH
dropout_rate = 0
input_sequence = Input((MAX_SEQUENCE_LENGTH,))
question = Input((MAX_QUE_LENGTH,))
context_encoder = Sequential()
context_encoder.add(Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))

question_encoder = Sequential()
question_encoder.add(Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_QUE_LENGTH,
                            trainable=False))


context_encoded = context_encoder(input_sequence)
question_encoded = question_encoder(question)
encoder = Bidirectional(LSTM(units=W,return_sequences=True))

passage_encoding = context_encoded
passage_encoding = encoder(passage_encoding)
passage_encoding = Dense(W,use_bias=False,trainable=True)(passage_encoding)

question_encoding = question_encoded
question_encoding = encoder(question_encoding)
question_encoding = Dense(W,use_bias=False,trainable=True)(question_encoding)

question_attention_vector = Dense(1)(question_encoding)
# question_attention_vector = Activation('softmax')(question_attention_vector)
question_attention_vector = Lambda(lambda q: keras.activations.softmax(q, axis=1))(question_attention_vector)
print(question_attention_vector)

question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
question_attention_vector = RepeatVector(N)(question_attention_vector)

ans_st = multiply([passage_encoding, question_attention_vector])
answer_start = concatenate([passage_encoding,question_attention_vector, ans_st])

answer_start = Dense(W, activation='relu')(answer_start)
answer_start = Dense(1)(answer_start)
answer_start = Flatten()(answer_start)
answer_start = Activation('softmax')(answer_start)
def s_answer_feature(x):
    maxind = K.argmax(
        x,
        axis=1,
    )
    return maxind

x = Lambda(lambda x: K.tf.cast(s_answer_feature(x), dtype=K.tf.int32))(answer_start)
start_feature = Lambda(lambda arg: K.tf.gather_nd(arg[0], K.tf.stack(
    [K.tf.range(K.tf.shape(arg[1])[0]), K.tf.cast(arg[1], K.tf.int32)], axis=1)))([passage_encoding, x])
start_feature = RepeatVector(N)(start_feature)


ans_1 = multiply([passage_encoding, question_attention_vector])
ans_2 = multiply([passage_encoding, start_feature])
answer_end = concatenate([passage_encoding,question_attention_vector,start_feature, ans_1,ans_2])

answer_end = Dense(W, activation='relu')(answer_end)
answer_end = Dense(1)(answer_end)
answer_end = Flatten()(answer_end)
answer_end = Activation('softmax')(answer_end)

inputs = [input_sequence, question]
outputs = [answer_start, answer_end]
model = Model(inputs,outputs)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[23]:


print(train_context.shape,train_question.shape,train_answer_start.shape,train_answer_end.shape)
model.fit([train_context, train_question], [train_answer_start,train_answer_end],
          batch_size=30,
          epochs=20,
          validation_data=([val_context, val_question], [val_answer_start,val_answer_end]))

