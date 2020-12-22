import sys,math
from collections import namedtuple
from queue import Queue

def calculate_entropy(bins_list):
	counter = Counter(bins_list)
	sum = 0
	for value in counter.values():
		sum += value
	entropy = 0.0
	for value in counter.values():
		entropy += -1 * (float(value)/sum)* math.log((float(value)/sum))
	return entropy

class Node(object):
	def __init__(self,data=None,freq=0,probability=0.0,entropy=0.0):
		self.data = data
		self.freq = freq
		self.probability = probability
		self.entropy = entropy
		#self.conditionalentropy = conditionalentropy
		self.children = [None] * 6;

class Trie(object):
	def __init__(self):
		self.root = self.NewNode(0)
		self.levelsum = [] 
		self.conditionalentropy = []
		self.leveluniquepatterns = []

	def NewNode(self,data):
		Q = Node()
		Q.data = data
		Q.freq = 1
		Q.probability = 0.0
		Q.entropy = 0.0
		#Q.conditionalentropy = 0.0
		return Q

	def AddString(self,s):
		#cur = node()
		cur = self.root
		for i in range(len(s)):
			if cur.children[ord(s[i])-48] == None:
				cur.children[ord(s[i])-48] = self.NewNode(s[i])
			else:
				cur.children[ord(s[i])-48].freq += 1
			cur = cur.children[ord(s[i])-48]

	def LevelOrderTraversal(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		while q.empty() == False:
			n = q.qsize()
			while n > 0:
				p = q.get()
				print("value:" + str(p.data) + ' ' + "entropy:" + str(p.entropy))
				for i in range(1,6):
					if p.children[i]:
						q.put(p.children[i])
				n -= 1
			#print '\n'

	def LevelSum(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		depth = 0
		while q.empty() == False:
			n = q.qsize()
			Sum = 0
			uniquepatterns = 0
			while n > 0:
				p = q.get()
				Sum += p.freq
				uniquepatterns += 1
				for i in range(1,6):
					if p.children[i]:
						q.put(p.children[i])
				n -= 1
			
			if depth < len(self.levelsum):
				self.levelsum[depth] = Sum
				self.leveluniquepatterns[depth] = uniquepatterns
			else:
				self.levelsum.append(Sum)
				self.leveluniquepatterns.append(uniquepatterns)
			depth += 1

	def AssignNodeProbability(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		depth = 1
		while q.empty() == False:
			n = q.qsize()
			while n > 0:
				p = q.get()
				p.probability = float(p.freq)/self.levelsum[depth-1]
				p.entropy = -p.probability*math.log(p.probability)
				for i in range(1,6):
					if p.children[i]:
						q.put(p.children[i])
				n -= 1
			depth += 1

	def ConditionalEntropy(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		index = 0
		while q.empty() == False:
			n = q.qsize()
			entropy = 0.0
			while n > 0:
				p = q.get()
				for i in range(1,6):
					if p.children[i]:
						entropy += p.children[i].entropy
						q.put(p.children[i])
				n -= 1
			if index == 0:
				self.conditionalentropy.append(entropy)
			else:
				if q.empty() == False:
					self.conditionalentropy.append(entropy-self.conditionalentropy[index-1])			
			index += 1



