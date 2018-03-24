import json
import pickle
data = json.load(open('SQuAD/train.json'))
data = data['data']
train_data = []
for d in data:
	paragraphs = d['paragraphs']
	for para in paragraphs:
		train_data.append(para)
with open('train_data.pkl', 'wb') as f:
	pickle.dump(train_data, f)

# with open('train_data.pkl', 'rb') as f:
# 	mynewlist = pickle.load(f)
# print(len(train_data))