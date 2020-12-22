import sys
import random
import threading
import time
import numpy as np 
from math import ceil
from UserBins import *
from ConditionalEntropy import *
from collections import OrderedDict, Counter
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd 
import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from statistics import mean

certified_new_reals = []
certified_new_reals = []

def user_features(user_imd, user_ml):
	imd_bins = map_user_timestamp_bins(user_imd)
	s = ''.join(str(b) for b in imd_bins)
	tree = Trie()
	MIN_CCE = sys.float_info.max
	for length in range(1, len(s)+1):
		for i in range(len(s)-length+1):
			tree.AddString(s[i:i+length])
		tree.LevelSum()
		tree.AssignNodeProbability()
		tree.ConditionalEntropy()
		imd_cce = tree.conditionalentropy[length-1] + \
				((float(tree.leveluniquepatterns[length-1])/tree.levelsum[length-1])*100)*tree.conditionalentropy[0]
		if MIN_CCE < imd_cce:
			imd_cce = MIN_CCE
			break

	ml_bins = map_user_msg_bins(user_ml)
	s = ''.join(str(b) for b in ml_bins)
	tree = Trie()
	MIN_CCE = sys.float_info.max
	for length in range(1, len(s)+1):
		for i in range(len(s)-length+1):
			tree.AddString(s[i:i+length])
		tree.LevelSum()
		tree.AssignNodeProbability()
		tree.ConditionalEntropy()
		msg_len_cce = tree.conditionalentropy[length-1] + \
					((float(tree.leveluniquepatterns[length-1])/tree.levelsum[length-1])*100)*tree.conditionalentropy[0]
		if MIN_CCE < msg_len_cce:
			msg_len_cce = MIN_CCE
			break
	return imd_cce, msg_len_cce

def extract_stream_features(filename):
	print(filename)
	users = cluster_user_timestamps_msgs(filename)
	user_info = user_imd_ml(users)
	imds_cce = []
	ml_cce = []
	for user in user_info.keys():
		user_imd = user_info[user]['t']
		user_ml = user_info[user]['m']
		imd_cce, msg_len_cce = user_features(user_imd, user_ml)
		imds_cce.append(imd_cce)
		ml_cce.append(msg_len_cce)
	'''
	try:
		st_features = []
		imds_cce = np.array(imds_cce)
		ml_cce = np.array(ml_cce)
		st_features.extend([np.quantile(imds_cce,0.25), np.quantile(imds_cce,0.5), np.quantile(imds_cce, 0.75)])
		st_features.extend([np.quantile(ml_cce, 0.25), np.quantile(ml_cce,0.5), np.quantile(ml_cce, 0.75)])
		if 'dip_7777' in filename:
			st_features.append(1)
		else:
			if ('data' in filename or 'Viewers' in filename) and (filename.split('#')[1].split('database')[0] in certified_reals or \
			filename.split('#')[1].split('database')[0] in certified_new_reals):
				st_features.append(0)
			else:
				st_features.append(1)
		return st_features
	except:
		return None
	'''
	return imds_cce, ml_cce

def getRealUsers(real_filename):
	real_users = set()
	with open(real_filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			real_users.add(user)
	f.close()
	return list(real_users)

def getAllFilesRecursively(path):
	dirs = [d for d in listdir(path) if isdir(join(path,d)) and 'Merged' in d]
	for d in dirs:
		print(d)
		for files_in_d in listdir(join(path,d)):
			tokens = files_in_d.split('#')
			real_filename = './Real Data/' + tokens[0] + '/#' + tokens[1] + '.txt'
			real_users = getRealUsers(real_filename)
			filename = join(join(path,d),files_in_d)	
			real_x = []
			real_y = []
			bot_x = []
			bot_y = []
			imd_x = []
			ml_y = []
			print(files_in_d)
			users = cluster_user_timestamps_msgs(filename)
			user_info = user_imd_ml(users)
			for user in user_info.keys():
				user_imd = user_info[user]['t']
				user_ml = user_info[user]['m']
				imd_cce, msg_len_cce = user_features(user_imd, user_ml)
				if user in real_users:
					real_x.append(imd_cce)
					real_y.append(msg_len_cce)
				else:
					bot_x.append(imd_cce)
					bot_y.append(msg_len_cce)
				imd_x.append(imd_cce)
				ml_y.append(msg_len_cce)
			try:
				imd_avg_x = mean(imd_x)
				ml_avg_y = mean(ml_y)
				fourquad_x = []
				fourquad_y = []
				onequad_x = []
				onequad_y = []
				for i in range(len(imd_x)):
					if imd_x[i] < imd_avg_x:
						onequad_x.append(imd_x[i])
					if imd_x[i] > imd_avg_x:
						fourquad_x.append(imd_x[i])
					if ml_y[i] < ml_avg_y:
						onequad_y.append(ml_y[i])
					if ml_y[i] > ml_avg_y:
						fourquad_y.append(ml_y[i])
				onequad_avg_x = mean(onequad_x)
				onequad_avg_y = mean(onequad_y)
				fourquad_avg_x = mean(fourquad_x)
				fourquad_avg_y = mean(fourquad_y)
				plt.scatter(real_x,real_y,c='blue',label='real')
				plt.scatter(bot_x,bot_y, c='red',label='bot')
				plt.axvline(x=imd_avg_x,c='yellow')
				plt.axhline(y=ml_avg_y,c='green')
				plt.axvline(x=onequad_avg_x,c='yellow',linestyle='--')
				plt.axhline(y=onequad_avg_y,c='green',linestyle='--')
				plt.axvline(x=fourquad_avg_x,c='yellow',linestyle='--')
				plt.axhline(y=fourquad_avg_y,c='green',linestyle='--')
				plt.title('IMD CCE vs ML CCE')
				plt.xlabel('User IMD CCE')
				plt.ylabel('User ML CCE')
				plt.show()
				#plt.savefig('./plots/'+files_in_d.split('.')[0] + '.png')
			except:
				pass

if __name__ == "__main__":

	getAllFilesRecursively('./Real Data/')
