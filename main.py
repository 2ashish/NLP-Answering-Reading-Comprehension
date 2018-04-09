
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import re
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import json
import os
import sys


# In[2]:


squad_dataset = json.load(open('SQuAD/train.json'))


# In[3]:


squad_dataset['data'][0]['paragraphs'][0]['context']


# In[4]:


squad_dataset['data'][0]['paragraphs'][0]['qas']


# In[5]:


squad_dataset['data'][0]['paragraphs'][0]['context'][0]


# In[6]:


context_file = open(os.path.join('./', 'train.context'), 'w')


# In[7]:


contexts = []
for pid in range(len(squad_dataset['data'][0]['paragraphs'])):
    context = squad_dataset['data'][0]['paragraphs'][pid]['context']
    contexts.append(context)

for context in contexts:
    context_file.write(context + '\n')


# In[8]:


context = squad_dataset['data'][0]['paragraphs'][0]['context']


# In[9]:


def tokenize(sequence):
    # tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    tokens = nltk.word_tokenize(sequence)
    return tokens


def token_map(context, context_tokens):
    word = ''
    current_token_id = 0
    token_map = dict()

    for cid, c in enumerate(context):
        if c != ' ':
            word += c
            context_token = context_tokens[current_token_id]
            if word == context_token:
                start = cid - len(word) + 1
                token_map[start] = [word, current_token_id]
                word = ''
                current_token_id += 1
    return token_map


# In[10]:


def get_data(dataset):
    with open(os.path.join('./', 'train_context'), 'w') as context_file,     open(os.path.join('./', 'train_question'), 'w') as question_file,     open(os.path.join('./', 'train_answer'), 'w') as answer_file,     open(os.path.join('./', 'train_span'), 'w') as span_file:
        for data_id in range(len(dataset['data'])):
            data = dataset['data'][data_id]
            for para_id in range(len(data['paragraphs'])):
                para = data['paragraphs'][para_id]
                context = para['context']
                qas = para['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    answer_text = qas[qid]['answers'][0]['text']
                    answer_start = qas[qid]['answers'][0]['answer_start']
                    answer_end = answer_start + len(answer_text)
                    
                    question_file.write(question + '\n')
                    answer_file.write(answer_text + '\n')
                    span_file.write(str(answer_start) + ',' + str(answer_end) + '\n')
                
                context_file.write(context + '\n')


# In[11]:


ans_map = token_map(context, ctokens)


# In[12]:


get_data(squad_dataset)

