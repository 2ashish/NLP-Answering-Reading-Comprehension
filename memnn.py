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
# sbstemmer = SnowballStemmer('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[^\s]+')


NUMCONTEXT = 1000

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


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
story_maxlen = max(map(len, (x for x in inp)))
query_maxlen = max(map(len, (x for x in que)))
print(story_maxlen,query_maxlen)

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs, queries, answers = vectorize_stories(inp,que,ans)
print(answers.shape)
#divide in training and testing
k= int(math.floor(0.8*len(inp)))
# print(k)
inputs_train = inputs[0:k]
inputs_test = inputs[k:len(inp)]
queries_train = queries[0:k]
queries_test = queries[k:len(inp)]
answers_train = answers[0:k]
answers_test = answers[k:len(inp)]
anstrain1 = answers_train[:,0].flatten()
anstrain2 = answers_train[:,1].flatten()
anstest1 = answers_test[:,0].flatten()
anstest2 = answers_test[:,1].flatten()


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

# placeholders
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

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer = LSTM(32)(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer1 = Dropout(0.3)(answer)
answer1 = Dense(vocab_size)(answer1)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer1 = Activation('softmax')(answer1)
# build the final model

answer2 = Dropout(0.3)(answer)
answer2 = Dense(vocab_size)(answer2)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer2 = Activation('softmax')(answer2)

model = Model(inputs=[input_sequence, question], outputs=[answer1])
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# print(answers_train.shape)
# train
model.fit([inputs_train, queries_train], [anstrain1],
          batch_size=32,
          epochs=5,
          validation_data=([inputs_test, queries_test], [anstest1]))
