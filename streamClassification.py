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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import multiprocessing as mp 
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

DATA_PATH = './Real Data/'
certified_new_reals = []
certified_new_reals = []
real_files_lt = []
cc_files_lt = []
r1_files_lt = []
r2_files_lt = []
og_files_lt = []
train_files = []
train_fts = []
test_fts = []

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

def getAllFilesRecursivelyinList(path):
	dirs = [d for d in listdir(path) if isdir(join(path,d)) and ('data' in d or 'Viewers' in d)]
	for d in dirs:
		for files_in_d in listdir(join(path,d)):
			filename = join(join(path,d), files_in_d)
			if 'dip_7777' in filename:
				if 'chatterscontrolled' in filename:
					cc_files_lt.append(filename)
				elif 'random1' in filename:
					r1_files_lt.append(filename)
				elif 'random2' in filename:
					r2_files_lt.append(filename)
				else:
					og_files_lt.append(filename)
			else:
				if 'database' in filename:
					real_files_lt.append(filename)
	print("# cc files:", len(cc_files_lt))
	print("# r1 files:", len(r1_files_lt))
	print("# r2 files:", len(r2_files_lt))
	print("# og files:", len(og_files_lt))
'''
def getAllFilesRecursively(path):
	#features = []
	#abels = []
	files = []
	dirs = [d for d in listdir(path) if isdir(join(path,d))]
	for d in dirs:
		if 'Viewers' in str(d):
			for files_in_d in listdir(join(path,d)):
				filename = join(join(path,d),files_in_d)
				if isfile(filename) and 'database' in str(filename):
					print(filename)
					files.append(filename)
					
					#if extract_stream_features(filename) is not None:
					#	features.append(extract_stream_features(filename))
					#	labels.append(0)
					
		elif 'data' in str(d):
			for files_in_d in listdir(join(path,d)):
				filename = join(join(path,d),files_in_d)
				if isfile(filename) and 'database' in str(filename):
					print(filename)
					files.append(filename)
					
					#if extract_stream_features(filename) is not None:
					#	features.append(extract_stream_features(filename))
					#	labels.append(0)
					
		else:
			for files_in_d in listdir(join(path,d)):
				filename = join(join(path,d),files_in_d)
				if isfile(filename) and 'database' in str(filename):
					print(filename)
					files.append(filename)
					
					#if extract_stream_features(filename) is not None:
					#	features.append(extract_stream_features(filename))
					#	labels.append(1)
					
	return files
'''

def run_model(model, alg_name, X, y):
	#model.fit(X_train,y_train)
	#y_pred = model.predict(X_test)
	#accuracy = accuracy_score(y_test, y_pred) * 100
	accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=5)
	precision = cross_val_score(model, X, y, scoring='precision', cv=5)
	recall = cross_val_score(model, X, y, scoring='recall', cv=5)
	#y_pred = cross_val_predict(model, X, y_pred, cv=5)
	#cm = confusion_matrix(Y, y_pred)
	print("Classifier:" + alg_name + ' ' + "Accuracy:" + str(accuracy.mean()*100))
	print("Classifier:" + alg_name + ' ' + "Precision:" + str(precision.mean()*100))
	print("Classifier:" + alg_name + ' ' + "Recall:" + str(recall.mean()*100))
	#print("Confusion matrix")
	#print(cm)

if __name__ == "__main__":
	
	with open(join(DATA_PATH,"certified real.txt")) as f:
		certified_reals = f.readlines()
	certified_reals = [real.strip('\n') for real in certified_reals]

	with open(join(DATA_PATH, "certified real for new data.txt")) as f:
		certified_new_reals = f.readlines()
	certified_new_reals = [real.strip('\n') for real in certified_new_reals]

	#files = getAllFilesRecursively(DATA_PATH)
	getAllFilesRecursivelyinList(DATA_PATH)
	'''
	ratio = 0.6
	train_files = []
	train_files.extend(real_files_lt[:ceil((ratio+0.1)*len(real_files_lt))])
	train_files.extend(cc_files_lt[:ceil(ratio*len(cc_files_lt))])
	train_files.extend(r1_files_lt[:ceil(ratio*len(r1_files_lt))])
	train_files.extend(r2_files_lt[:ceil(ratio*len(r2_files_lt))])
	train_files.extend(og_files_lt[:ceil(ratio*len(og_files_lt))])
	'''
	#print('Storing real world stream features')
	print("Real vs Real")
	#df_real = pd.read_csv('s1_real_features.csv',index_col=0)
	#X_train = df_real.iloc[:,0:6]
	#y_train = df_real.iloc[:,6:7]
	pool = mp.Pool(processes=200)
	train_features = pool.map(extract_stream_features, real_files_lt)
	filtered_features = [ft for ft in train_features if ft]
	df = pd.DataFrame(filtered_features)
	df.columns = ['25%_imd_quan','50%_imd_quan','75%_imd_quan','25%_ml_quan','50%_ml_quan','75%_ml_quan','label']
	#df.dropna()
	print("Shape of real df:", df.shape)
	#print("Shape of real df:", df_real.shape)
	X = df.iloc[:,0:6]
	y = df.iloc[:,6:7]
	#X = X_train.append(X_test)
	#y = y_train.append(y_test)
	#X.to_csv('s1_real_r2_features.csv')

	#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=100)
	
	#-------Decision Tree-----------

	print("Real vs Real")

	model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
	#run_model(model,"Decision Tree",X_train,y_train,X_test,y_test)
	run_model(model, "Decision Tree", X, y)

	#-------Random Forest-----------

	model = RandomForestClassifier(n_estimators=10)
	#run_model(model,"Random Forest",X_train,y_train,X_test,y_test)
	run_model(model, "Random Forest", X, y)

	#-------xgboost-----------------

	model = XGBClassifier()
	#run_model(model,"XGBoost",X_train,y_train,X_test,y_test)
	run_model(model, "XGBoost", X, y)

	#-------SVM Classifier-----------

	model = SVC()
	#run_model(model,"SVM Classifier",X_train,y_train,X_test,y_test)
	run_model(model, "SVM Classifier", X, y)

	#-------Nearest Classifier-------

	model = neighbors.KNeighborsClassifier()
	#run_model(model,"Nearest Neighbors Classifier",X_train,y_train,X_test,y_test)
	run_model(model, "Nearest Neighbors Classifier", X, y)

	#-------SGD Classifier-----------

	model = OneVsRestClassifier(SGDClassifier())
	#run_model(model,"SGD Classifier",X_train,y_train,X_test,y_test)
	run_model(model, "SGD Classifier", X, y)

	#-------Gaussian NB--------------

	model = GaussianNB()
	#run_model(model,"Gaussian NB",X_train,y_train,X_test,y_test)
	run_model(model, "Gaussian NB", X, y)

	#-------NN-MLP-------------------

	model = MLPClassifier()
	#run_model(model,"NN-MLP",X_train,y_train,X_test,y_test) 
	run_model(model, "NN-MLP", X, y)
	