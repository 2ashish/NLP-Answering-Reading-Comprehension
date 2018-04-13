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
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

squad_dataset = json.load(open('SQuAD/train.json'))

def get_data(dataset):
    with open(os.path.join('./data/', 'train_context'), 'w') as context_file,     open(os.path.join('./data/', 'train_question'), 'w') as question_file,     open(os.path.join('./data/', 'train_answer'), 'w') as answer_file,     open(os.path.join('./data/', 'train_span'), 'w') as span_file:
        for data_id in range(len(dataset['data'])):
            data = dataset['data'][data_id]
            for para_id in range(len(data['paragraphs'])):
                para = data['paragraphs'][para_id]
                context = para['context']
                if context[0] == '\n':
                    context = context[-len(context)+1:]
                qas = para['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    answer_text = qas[qid]['answers'][0]['text']
                    answer_start = qas[qid]['answers'][0]['answer_start']
                    answer_end = answer_start + len(answer_text)
                    
                    question_file.write(question + '\n')
                    answer_file.write(answer_text + '\n')
                    span_file.write(str(answer_start) + ',' + str(answer_end) + '\n')
                
                context_file.write(context + '\n' + '-' + '\n')
                question_file.write('-' + '\n')
                answer_file.write('-' + '\n')
                span_file.write('-' + '\n')

get_data(squad_dataset)

squad_dataset = json.load(open('SQuAD/dev.json'))

def get_data_test(dataset):
    with open(os.path.join('./data/', 'test_context'), 'w') as context_file,     open(os.path.join('./data/', 'test_question'), 'w') as question_file,     open(os.path.join('./data/', 'test_answer'), 'w') as answer_file,     open(os.path.join('./data/', 'test_span'), 'w') as span_file:
        for data_id in range(len(dataset['data'])):
            data = dataset['data'][data_id]
            for para_id in range(len(data['paragraphs'])):
                para = data['paragraphs'][para_id]
                context = para['context']
                if context[0] == '\n':
                    context = context[-len(context)+1:]
                qas = para['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    answer_text = qas[qid]['answers'][0]['text']
                    answer_start = qas[qid]['answers'][0]['answer_start']
                    answer_end = answer_start + len(answer_text)
                    
                    question_file.write(question + '\n')
                    answer_file.write(answer_text + '\n')
                    span_file.write(str(answer_start) + ',' + str(answer_end) + '\n')
                
                context_file.write(context + '\n' + '-' + '\n')
                question_file.write('-' + '\n')
                answer_file.write('-' + '\n')
                span_file.write('-' + '\n')

get_data_test(squad_dataset)