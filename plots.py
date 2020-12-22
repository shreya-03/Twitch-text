import sys,math
#import seaborn as sns
#from tfidf import *
#from ConditionalEntropy import *
from collections import Counter,OrderedDict
import operator
import matplotlib.pyplot as plt
#import plotly.plotly as py
#import plotly.graph_objs as go
import numpy as np
#import plotly
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join,isdir
#from fitter import Fitter
#from pylab import *
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
from math import floor,ceil

def plotUserLevelML(bot_names,syn_bot_names):
	noRealUsersvsMLdict = {}
	noBotsUsersVsMLdict = {}
	DATA_PATH = './Real Data/'
	
	count = 1
	dirs = [d for d in listdir(DATA_PATH) if 'Viewers' in str(d) or 'data' in str(d) and isdir(join(DATA_PATH,d))]
	for d in dirs:
		for files_in_d in listdir(join(DATA_PATH,d)):
			if 'database' in files_in_d:
				filename = join(join(DATA_PATH,d),files_in_d)
				print(count)
				count += 1

	#filename = './Real Data/Merged_Data/35-40 Viewers#zilabeardatabase_new#dip_7777database_chatterscontrolled.txt'
				with open(filename,'r') as f:
					lines = f.readlines()
					for line in lines:
						message = str(line.split(',"m":')[1].split(',"nm":')[0].replace('"',''))
						user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
						ml = len(message.split(' '))
						if user in bot_names or user in syn_bot_names:
							if ml in noBotsUsersVsMLdict.keys():
								noBotsUsersVsMLdict[ml].add(user)
							else:
								noBotsUsersVsMLdict[ml] = set()
								noBotsUsersVsMLdict[ml].add(user)
						else:
							if ml in noRealUsersvsMLdict.keys():
								noRealUsersvsMLdict[ml].add(user)
							else:
								noRealUsersvsMLdict[ml] = set()
								noRealUsersvsMLdict[ml].add(user)
	#print(noRealUsersvsMLdict)
	for ml in noRealUsersvsMLdict.keys():
		noRealUsersvsMLdict[ml] = len(list(noRealUsersvsMLdict[ml]))/10
	for ml in noBotsUsersVsMLdict.keys():
		noBotsUsersVsMLdict[ml] = len(list(noBotsUsersVsMLdict[ml]))/10
	plt.rcParams['figure.figsize'] = (6,6)
	plt.rcParams['axes.facecolor'] = 'white'
	plt.rcParams['axes.edgecolor'] = '#777777'
	plt.rcParams['axes.labelweight'] = 'bold'
	plt.rc('font', family='sans-serif',weight='bold')
	plt.rc('xtick', labelsize='large')
	plt.rc('ytick', labelsize='large')	
	plt.xlabel('Message Lengths')
	plt.ylabel('Number of users')
	bots_lists = sorted(noBotsUsersVsMLdict.items()) 
	x, y = zip(*bots_lists) 
	plt.plot(x, y, color='red', label='bot users')
	real_lists = sorted(noRealUsersvsMLdict.items())
	x, y = zip(*real_lists)
	plt.plot(x, y, color='blue',label='real users')
	plt.xticks([0,40,80,120])
	plt.yticks([0,500,1000,1500,2000])
	plt.legend(loc='upper right',prop={'size':20})
	plt.show()

def plotStreamLevelML(certified_reals, certified_new_reals):
	fracRealUsersVsML = {}
	fracBotUsersVsML = {}
	DATA_PATH = './Real Data/'
	real_count = 0
	bot_count = 0
	dirs = [d for d in listdir(DATA_PATH) if 'Viewers' in str(d) or 'data' in str(d) and isdir(join(DATA_PATH,d))]
	for d in dirs:
		for files_in_d in listdir(join(DATA_PATH,d)):
			if 'database' in files_in_d:
				filename = join(join(DATA_PATH,d),files_in_d)
				streamML_dict = {}
				users = set()
				with open(filename,'r') as f:
					lines = f.readlines()
					for line in lines:
						message = str(line.split(',"m":')[1].split(',"nm":')[0].replace('"',''))
						user = str(lines.split(',"u":')[1].split(',"e":')[0].replace('"',''))
						ml = len(message.split(' '))
						if ml in streamML_dict.keys():
							streamML_dict[ml] += 1
						else:
							streamML_dict[ml] = 1
						users.add(user)
				f.close()
				total_users = len(list(users))
				for ml in streamML_dict.keys():
					streamML_dict[ml] /= float(total_users)
				lists = sorted(streamML_dict.items())
				x, y = zip(*lists)
				if filename in certified_reals or filename in certified_new_reals:
					if real_count == 0:
						plt.plot(x, y, color='blue', label='real stream')
						real_count += 1
					else:
						plt.plot(x, y, color='blue')
				else:
					if bot_count == 0:
						plt.plot(x, y, color='red', label='botted stream')
						bot_count += 1
					else:
						plt.plot(x, y, color='red')
	plt.rc('font', family='serif')
	plt.rc('xtick', labelsize='small')
	plt.rc('ytick', labelsize='medium')
	plt.xlabel('Message Lengths')
	plt.ylabel('Fraction of users')
	plt.legend()
	plt.show()

bot_names = [line.strip() for line in open('./all_bots_names.txt','r')]
syn_bot_names = [line.strip() for line in open('./synbots.txt','r')]
with open(join('./Real Data/',"certified real.txt")) as f:
	certified_reals = f.readlines()
	certified_reals = [real.strip('\n') for real in certified_reals]

with open(join('./Real Data/', "certified real for new data.txt")) as f:
	certified_new_reals = f.readlines()
	certified_new_reals = [real.strip('\n') for real in certified_new_reals]

plotUserLevelML(bot_names,syn_bot_names)