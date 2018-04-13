from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Merge, Permute, RepeatVector, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.layers import recurrent
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import math
import string
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
# sbstemmer = SnowballStemmer('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[^\s]+')


NUMCONTEXT = 1000

def unicodetoascii(text):

    TEXT = (text.
            replace('\\xe2\\x80\\x99', "'").
            replace('\\xc3\\xa9', 'e').
            replace('\\xe2\\x80\\x90', '-').
            replace('\\xe2\\x80\\x91', '-').
            replace('\\xe2\\x80\\x92', '-').
            replace('\\xe2\\x80\\x93', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x98', "'").
            replace('\\xe2\\x80\\x9b', "'").
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9d', '"').
            replace('\\xe2\\x80\\x9e', '"').
            replace('\\xe2\\x80\\x9f', '"').
            replace('\\xe2\\x80\\xa6', '...').
            replace('\\xe2\\x80\\xb2', "'").
            replace('\\xe2\\x80\\xb3', "'").
            replace('\\xe2\\x80\\xb4', "'").
            replace('\\xe2\\x80\\xb5', "'").
            replace('\\xe2\\x80\\xb6', "'").
            replace('\\xe2\\x80\\xb7', "'").
            replace('\\xe2\\x81\\xba', "+").
            replace('\\xe2\\x81\\xbb', "-").
            replace('\\xe2\\x81\\xbc', "=").
            replace('\\xe2\\x81\\xbd', "(").
            replace('\\xe2\\x81\\xbe', ")")

                 )
    return TEXT

def vectorize_stories(inp,que,ans):
    inputs, queries, answers ,t = [], [], [], []
    for i in range(0,len(inp)):
        # print(i)
        inputs.append([word_idx[w] for w in inp[i]])
        queries.append([word_idx[w] for w in que[i]])
        t = ([word_idx[w] for w in ans[i]])
        a = np.zeros(len(word_idx) + 1)
        for w in t:
            a[w] = 1
        answers.append(a)
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))

def para_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    #stop_words = set(stopwords.words('english'))
    #tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens if not w in stop_words]
    #tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens]
    spans = tokenizer.span_tokenize(data)
    sp = [span for span in spans]
    # stop_words = set(stopwords.words('english'))
    # tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens]
    return x_tokens,sp
    #return tokens

def que_tokenizer(data):
    x_tokens = tokenizer.tokenize(data)
    #stop_words = set(stopwords.words('english'))
    #tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens if not w in stop_words]
    #tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens]
    # stop_words = set(stopwords.words('english'))
    # tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens]
    return x_tokens
    #return tokens

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        #text = re.sub(re'[^a-zA-Z0-9 ,*\u2019-]', u'', text.decode('utf8'), 0, re.UNICODE).encode("utf8")
        p2 = re.compile('\[[nN] [0-9]+\]')
        text = p2.sub(' ', text)
        p2 = re.compile('[—–—$\-\+\[”]')
        text = p2.sub(' ', text)
        p2 = re.compile('((?<![0-9])\.)|(\.(?![0-9]))')
        text = p2.sub(' ', text)
        p2 = re.compile('\'s')
        text = p2.sub(' ', text)
        exclude = set(string.punctuation)-set('.')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))

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

answer_file = open(os.path.join('./data/', 'train_answer'), 'r')
c = answer_file.read()
answers = re.split('\n' + '-' + '\n', c)
del c
# print(len(context))
context = context[0:NUMCONTEXT]

inp = []
que = []
ans = []
i =0
for c in context:
    # print(c)
    c = normalize_answer(c)
    # print(c)
    tokens = our_tokenizer(c)
    
    q=questions[i]
    a=answers[i]
    all_ques = re.split('\n', q)
    all_ans = re.split('\n', a)
    # all_s = re.split('\n', spa[i])
    for j in range (0,len(all_ques)):
        inp.append(tokens)
        # x = re.split(',',all_s[j])
        # x = list(map(int, x))
        # k = 0
        # for span in sp:
        #     if span[0] <= x[0] <= span[1]:
        #         st = k
        #     if span[0] <= x[1] <= span[1]:
        #         en = k
        #     k+=1
        que.append(our_tokenizer(normalize_answer(all_ques[j])))
        ans.append(our_tokenizer(normalize_answer(all_ans[j])))
        # ans.append([st,en])
        #ans.append(st)
    i+=1

print(len(inp))
print(inp[0])
print(que[0])
print(ans[0])

vocab = set()
for i in range(0,len(inp)):
    vocab |= set(inp[i] + que[i] + ans[i])
vocab = sorted(vocab)
print(len(vocab))

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x in inp)))
query_maxlen = max(map(len, (x for x in que)))
print(story_maxlen,query_maxlen)

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs, queries, answers = vectorize_stories(inp,que,ans)
print(inputs[0])
print(queries[0])
print(answers[0])
#divide in training and testing
k= int(math.floor(0.8*len(inp)))
# print(k)
inputs_train = inputs[0:k]
inputs_test = inputs[k:len(inp)]
queries_train = queries[0:k]
queries_test = queries[k:len(inp)]
answers_train = answers[0:k]
answers_test = answers[k:len(inp)]
# anstrain1 = answers_train[:,0].flatten()
# anstrain2 = answers_train[:,1].flatten()
# anstest1 = answers_test[:,0].flatten()
# anstest2 = answers_test[:,1].flatten()
# print(anstrain1[0])


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

#placeholders
input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=100))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=100,
                               input_length=query_maxlen))
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

# # concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# # # the original paper uses a matrix multiplication for this reduction step.
# # # we choose to use a RNN instead.
answer = LSTM(32)(answer)  # (samples, 32)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size ,activation='softmax')(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
#answer = Activation('softmax')(answer)
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
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 100

# model = Sequential()
# model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))(answer)
# model.add(Dropout(0.3))
# model.add(Dense(vocab_size, activation='softmax'))


# sentrnn = Sequential()
# sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
#                       input_length=story_maxlen))
# sentrnn.add(Dropout(0.3))

# qrnn = Sequential()
# qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
#                    input_length=query_maxlen))
# qrnn.add(Dropout(0.3))
# qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
# qrnn.add(RepeatVector(story_maxlen))

# model = Sequential()
# model.add(Merge([sentrnn, qrnn], mode='sum'))
# model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
# model.add(Dropout(0.3))
# model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print(answers_train.shape)
# train
model.fit([inputs_train, queries_train], answers_train,
          batch_size=32,
          epochs=120,
          validation_data=([inputs_test, queries_test], answers_test))
