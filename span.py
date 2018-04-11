import os,re
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
tokenizer = RegexpTokenizer(r'[^\s]+')

def our_tokenizer(data):
    x_tokens = tokenizer.tokenize(data.lower())
    spans = tokenizer.span_tokenize(data.lower())
    sp = [span for span in spans]
    # stop_words = set(stopwords.words('english'))
    tokens = [wordnet_lemmatizer.lemmatize(w) for w in x_tokens]
    return tokens,sp

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
spans = re.split('\n' + '-' + '\n', c)
del c


c = context[8]
q = questions[8]
a = answers[8]
s = spans[8]

all_q = re.split('\n', q)
all_a = re.split('\n', a)
all_s = re.split('\n', s)
x = re.split(',',all_s[1])
x = list(map(int, x))

t ,sp = our_tokenizer(c)

k = 0
for span in sp:
	if span[0] <= x[0] <= span[1]:
		st = k
	if span[0] <= x[1] <= span[1]:
		en = k
	k+=1

print(x[0],x[1])
print(all_a[1])

for i in range(st,en+1):
	print(t[i])
