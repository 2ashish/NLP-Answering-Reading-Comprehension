from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM, Bidirectional
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


NUMCONTEXT = 100

def loadGloveModel(gloveFile):
    #print "Loading Glove Model..."
    f = open(gloveFile,'r')
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs 
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def vectorize_stories(inp,que,ansb,anse):
    inputs, queries, answers, ans_begin, ans_end= [], [], [], [], []
    for i in range(0,len(inp)):
        inputs.append([word_idx[w] for w in inp[i] if w in word_idx])
        queries.append([word_idx[w] for w in que[i] if w in word_idx])
        a_begin = np.zeros(len(inp[i]))
        a_begin[ansb[i]] = 1
        a_end = np.zeros(len(inp[i]))
        a_end[anse[i]] = 1
        ans_begin.append(a_begin)
        ans_end.append(a_end)

    return (pad_sequences(inputs, maxlen=story_maxlen, padding='post'),
            pad_sequences(queries, maxlen=query_maxlen, padding='post'),
            pad_sequences(ans_begin, maxlen=story_maxlen, padding='post'),
            pad_sequences(ans_end, maxlen=story_maxlen, padding='post')
            )


#train dataset
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
span = re.split('\n' + '-' + '\n', c)
del c

context = context[0:NUMCONTEXT]
#context = context[0:len(context)-1]

inp_train = []
que_train = []
ans_train = []
ansb_train = []
anse_train = []
i =0
for c in context:
    context_token = tokenize(c)
    
    all_ques = re.split('\n', questions[i])
    all_ans = re.split('\n', answers[i])
    all_s = re.split('\n', span[i])
    for j in range (0,len(all_ques)):
        x = re.split(',',all_s[j])
        question_token = tokenize(all_ques[j])
        answer_token = tokenize(all_ans[j])
        contextToAnswerFirstWord = c[:int(x[0]) + len(answer_token[0])]
        answerBeginIndex = len(tokenize(contextToAnswerFirstWord)) - 1
        answerEndIndex = answerBeginIndex + len(answer_token) - 1

        inp_train.append(context_token)
        que_train.append(question_token)
        ans_train.append(answer_token)
        ansb_train.append(answerBeginIndex)
        anse_train.append(answerEndIndex)
    i+=1

print(len(inp_train))
# print(inp_train[0])
# print(que[0])
# print(ans[0])
# print(ansb[0])
# print(anse[0])


# test datasets
context_file = open(os.path.join('./data/', 'test_context'), 'r')
c = context_file.read()
context = re.split('\n' + '-' + '\n', c)
del c

question_file = open(os.path.join('./data/', 'test_question'), 'r')
c = question_file.read()
questions = re.split('\n' + '-' + '\n', c)
del c

answer_file = open(os.path.join('./data/', 'test_answer'), 'r')
c = answer_file.read()
answers = re.split('\n' + '-' + '\n', c)
del c

span_file = open(os.path.join('./data/', 'test_span'), 'r')
c = span_file.read()
span = re.split('\n' + '-' + '\n', c)
del c

context = context[0:NUMCONTEXT]
#context = context[0:len(context)-1]

inp_test = []
que_test = []
ans_test = []
ansb_test = []
anse_test = []
i =0
for c in context:
    context_token = tokenize(c)
    
    all_ques = re.split('\n', questions[i])
    all_ans = re.split('\n', answers[i])
    all_s = re.split('\n', span[i])
    for j in range (0,len(all_ques)):
        x = re.split(',',all_s[j])
        question_token = tokenize(all_ques[j])
        answer_token = tokenize(all_ans[j])
        contextToAnswerFirstWord = c[:int(x[0]) + len(answer_token[0])]
        answerBeginIndex = len(tokenize(contextToAnswerFirstWord)) - 1
        answerEndIndex = answerBeginIndex + len(answer_token) - 1

        inp_test.append(context_token)
        que_test.append(question_token)
        ans_test.append(answer_token)
        ansb_test.append(answerBeginIndex)
        anse_test.append(answerEndIndex)
    i+=1

print(len(inp_test))
# print(inp_train[0])
# print(que[0])
# print(ans[0])
# print(ansb[0])
# print(anse[0])

GloveDimOption = '50' # this  could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
embeddings_index = loadGloveModel('glove.6B.' + GloveDimOption + 'd.txt') 

vocab = set()
for i in range(0,len(inp_train)):
    vocab |= set(inp_train[i] + que_train[i])
vocab = sorted(vocab)
print(len(vocab))

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x in inp_train)))
query_maxlen = max(map(len, (x for x in que_train)))
print(story_maxlen,query_maxlen)

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, ans_begin_train, ans_end_train = vectorize_stories(inp_train,que_train,ansb_train,anse_train)
inputs_test, queries_test, ans_begin_test, ans_end_test = vectorize_stories(inp_test,que_test,ansb_test,anse_test)
# print(inputs[0])
# print(queries[0])
# print(ans_begin[0])
# print(ans_end[0])
#divide in training and testing
# k= int(math.floor(0.8*len(inp)))
# # print(k)
# inputs_train = inputs[0:k]
# inputs_test = inputs[k:len(inp)]
# queries_train = queries[0:k]
# queries_test = queries[k:len(inp)]
# ansb_train = ans_begin[0:k]
# ansb_test = ans_begin[k:len(inp)]
# anse_train = ans_end[0:k]
# anse_test = ans_end[k:len(inp)]



# print('-')
# print('inputs: integer tensor of shape (samples, max_length)')
# print('inputs_train shape:', inputs_train.shape)
# print('inputs_test shape:', inputs_test.shape)
# print(inputs_train[0])
# print('-')
# print('queries: integer tensor of shape (samples, max_length)')
# print('queries_train shape:', queries_train.shape)
# print('queries_test shape:', queries_test.shape)
# print(queries_train[0])
# print('-')
# print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
# print('answers_train shape:', answers_train.shape)
# print('answers_test shape:', answers_test.shape)
# print(answers_train[1])
# print('-')
# print('Compiling...')

embedding_matrix = np.zeros((vocab_size, 50))
for word, i in word_idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# placeholders
input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                            output_dim=50,
                            weights=[embedding_matrix],
                            input_length=story_maxlen,
                            trainable=True))
input_encoder_m.add(Bidirectional(LSTM(50, return_sequences=True)))
input_encoder_m.add(Dropout(0.3))

# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen,
                              input_length=story_maxlen,
                              trainable=True))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=50,
                               weights=[embedding_matrix],
                               input_length=query_maxlen))
question_encoder.add(Bidirectional(LSTM(50, return_sequences=True)))
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
#answer = LSTM(32)(answer)  # (samples, 32)
answer = Bidirectional(LSTM(256, implementation=2), merge_mode='mul')(answer)
answer = Dropout(0.3)(answer)
answer = Dense(story_maxlen, activation='softmax')(answer)  # (samples, vocab_size)

# we output a probability distribution over the vocabulary
# answer = Activation('softmax')(answer)
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
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

# print(answers_train.shape)
# train
model.fit([inputs_train, queries_train], ans_begin_train,
          batch_size=128,
          epochs=100,
          validation_data=([inputs_test, queries_test], ans_begin_test))
