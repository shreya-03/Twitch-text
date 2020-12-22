import sys, re
import ast
import math, random, requests, json
import numpy as np 
import pandas as pd 
from collections import OrderedDict, Counter
from os import listdir
from os.path import isfile, join, isdir
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import label_propagation
from preprocessing import *
from nltk.corpus import brown
from string import punctuation
from conversations import *
from Userclustering import *
from sklearn.metrics import accuracy_score

en_vocab = set(i.lower() for i in brown.words())
emoji_dict = dict()
slang_dict = dict()

def getChannelFollowers(path,channel_name):
	#print channel_name	
	follows = set()
	for file in listdir(path):
		#print file
		if file[13:-2] == channel_name:
			#print "entered"
			with open(join(path,file),'r') as f:
				lines = f.readlines()
				users = lines[1].split(' ')
				#print users
				for user in users:
					follows.add(user)
			f.close()
	return list(follows)

def textNormalization(message):
	spellcorrect = spellCorrection()
	tokens = message.split(' ')
	normalized_msg = ''
	for token in tokens:
		if token.lower() in en_vocab:
			normalized_msg += token.lower() + ' '
		else:
			token = token.strip(punctuation)
			if token.startswith('@'):
				normalized_msg += token
			else:
				if token in emoji_dict.keys():
					normalized_msg += emoji_dict[token]
				else:
					token = token.lower()
					matcher = re.compile(r'(.)\1*')
					uniq_tokens = [match.group() for match in matcher.finditer(token)]
					normalized_token = ''
					if len(uniq_tokens) > 3:
						try:
							if (uniq_tokens[0] == uniq_tokens[2] and uniq_tokens[1] == uniq_tokens[3]):
								normalized_token += uniq_tokens[:2]
							if (uniq_tokens[0] == uniq_tokens[3] and uniq_tokens[1] == uniq_tokens[4] and uniq_tokens[2] == uniq_tokens[5]):
								normalized_token += uniq_tokens[:3]
						except:
							normalized_token += token
					if len(normalized_token) == 0:
						for i in range(len(uniq_tokens)):
							if len(uniq_tokens[i]) <= 2:
								normalized_token += uniq_tokens[i]
							else:
								normalized_token += uniq_tokens[i][:2]
					if normalized_token in slang_dict.keys():
						normalized_msg += slang_dict[normalized_token]
					else:
						if normalized_token in en_vocab:
							normalized_msg += normalized_token + ' '
						else:
							normalized_msg += spellcorrect.correction(normalized_token)
	return normalized_msg
 
def extractTagUsers(chat_info,users):
	for i in range(len(chat_info)):
		user_id = users.index(chat_info[i]['user'])
		message = chat_info[i]['msgs']
		tokens = message.split(' ')
		for token in tokens:
			if token.startswith('@'):
				tag_user_id = users.index(token[1:])
				tag_user_fts[user_id][tag_user_id] += 1

def getTotalUsers(chat_info):
	users = []
	for i in range(len(chat_info)):
		if chat_info[i]['user'] not in users:
			users.append(chat_info[i]['user'])
	return users

def getRealUsers(filename):
	real_users = set()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			real_users.add(user)
	f.close()
	return list(real_users)
 
def extractUserMessages(filename):
	chat_info =[]
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user_dict = dict()
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			message = str(line.split(',"m":')[1].split(',"nm"')[0].replace('"',''))
			normalized_msg = textNormalization(message)
			user_dict['user'] = user
			user_dict['msgs'] = normalized_msg
			chat_info.append(user_dict)
	return chat_info

def data_labelprop(X,real_users_index,bot_users_index):
	label_X = []
	label_Y = []

	for index in range(len(X)):
		if index in real_users_index:
			label_X.append(X.iloc[index,:].values.tolist())
			label_Y.append(0)
		elif index in bot_users_index:
			label_X.append(X.iloc[index,:].values.tolist())
			label_Y.append(1)
		else:
			label_X.append(X.iloc[index,:].values.tolist())
			label_Y.append(-1)
	return label_X,label_Y


if __name__ == "__main__":
	   
	filename = sys.argv[1]
	real_filename = sys.argv[2]
	real_users = getRealUsers(real_filename)
	global tag_user_fts
	chat_info = extractUserMessages(filename)
	print("Extracted and normalized user messages")
	users = getTotalUsers(chat_info)
	print("Users list in main")
	print(users)
	labels = []
	for user in users:
		if user in real_users:
			labels.append(0)
		else:
			labels.append(1)
	tag_user_fts = [[0 for y in range(len(users))] for x in range(len(users))]
	emoji_dict = extractEmojiDict('./emoji_desc.txt')
	slang_dict = expandAcronyms('./slang.txt')
	extractTagUsers(chat_info,users)
	print("obtained tag user features")
	conversation_fts = pd.DataFrame(ConversationalFeatures(chat_info))
	print("obtained conversational features")
	tag_user_fts = pd.DataFrame(tag_user_fts)
	user_features = pd.concat([conversation_fts,tag_user_fts],axis=1)
	print("completed feature extraction")
	print(user_features.shape)
	
	real_user_index, bot_user_index = labeling_data(filename, users)
	channel_followers = get_channel_followers('../followers_cnt/',filename.split('#')[1].split('database')[0])
	for user in channel_followers:
		if user in users:
			real_users_index.append(users.index(user))
	
	label_X,label_Y = data_labelprop(user_features,real_user_index,bot_user_index)
	# Learn with LabelSpreading
	label_spread = label_propagation.LabelSpreading(kernel='rbf', alpha=0.6)
	label_spread.fit(label_X, label_Y)
	output_labels = label_spread.transduction_
	# print output_labels
	# print labels
	'''
	orig_tot,orig_cor = 0,0
	total,correct = 0,0
	for i in range(len(output_labels)):
		if label_Y[i] == -1:
			if labels[i] == output_labels[i]:
				correct += 1
			else:
				print label_X[i],i
			total += 1
		if orig_labelY[i] == -1:
			if labels[i] == orig_output_labels[i]:
				orig_cor += 1
			orig_tot += 1
	print correct,total
	print (float(correct)/total)*100
	print accuracy_score(np.array(labels),output_labels)*100
	return accuracy_score(np.array(labels),output_labels)*100
	print orig_cor,orig_tot
	print (float(orig_cor)/orig_tot)*100
	'''
	print accuracy_score(np.array(labels),output_labels)*100