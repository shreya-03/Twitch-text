import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown
from collections import Counter
import ast

class spellCorrection(object):

	def __init__(self):
		self.WORDS = Counter(self.words(open('big.txt').read()))

	def words(self,text):
		return re.findall(r'\w+',text.lower())

	def probability(self,word):
		# Probability of word
		N = sum(self.WORDS.values())
		return self.WORDS[word]/float(N)

	def correction(self,word):
		# Most probable spelling correction for word
		return max(self.candidates(word), key=self.probability)

	def candidates(self,word):
		# Generate posible spelling corrections for word
		return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

	def known(self,words):
		# The subset of words that appear in the doctionary of WORDS
		return set(w for w in words if w in self.WORDS)

	def edits1(self,word):
		#All edits that are one edit away from word
		letters    = 'abcdefghijklmnopqrstuvwxyz'
		splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
		deletes    = [L + R[1:]               for L, R in splits if R]
		transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
		replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
		inserts    = [L + c + R               for L, R in splits for c in letters]
		return set(deletes + transposes + replaces + inserts)

	def edits2(self,word): 
		#All edits that are two edits away from `word`
		return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

def formatLine(line):
	line = re.sub("\'s","is",line)
	line = re.sub("\'ve","have",line)
	line = re.sub("n\'t","not",line)
	line = re.sub("\'ll","will",line)
	line = re.sub("\'m","am",line)
	line = re.sub("\'d","would",line)
	line = re.sub("\'re","are",line)
	line = re.sub("&gt",">",line)
	line = re.sub("&lt","<",line)
	return line

def removeNonAscii(words):
	"""Remove non-ASCII characters from list of tokenized words"""
	new_words = []
	for word in words:
		new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		new_words.append(new_word)
	return new_words

def toLowercase(words):
	"""Convert all characters to lowercase from list of tokenized words"""
	new_words = []
	for word in words:
		new_word = word.lower()
		new_words.append(new_word)
	return new_words

def removePunctuation(words):
	"""Remove punctuation from list of tokenized words"""
	new_words = []
	for word in words:
		new_word = re.sub(r'[^\w\s]', '', word)
		if new_word != '':
			new_words.append(new_word)
	return new_words

def replaceNumbers(words):
	"""Replace all interger occurrences in list of tokenized words with textual representation"""
	p = inflect.engine()
	new_words = []
	for word in words:
		if word.isdigit():
			new_word = p.number_to_words(word)
			new_words.append(new_word)
		else:
			new_words.append(word)
	return new_words

def removeStopwords(words):
	"""Remove stop words from list of tokenized words"""
	new_words = []
	for word in words:
		if word not in stopwords.words('english'):
			new_words.append(word)
	return new_words

def stemWords(words):
	"""Stem words in list of tokenized words"""
	stemmer = LancasterStemmer()
	stems = []
	for word in words:
		stem = stemmer.stem(word)
		stems.append(stem)
	return stems

def extractEmojiDict(filename):
	emoji_dict = dict()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			res = ast.literal_eval(line)
			emoji = res['emoji']
			desc = res['description']
			emoji_dict[emoji] = desc
	f.close()
	return emoji_dict

def expandAcronyms(filename):
	slang_dict = dict()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			#print(line)
			acronym = line.split('=')[0]
			expansion = line.split('=')[1]
			slang_dict[acronym] = expansion
	return slang_dict

def lemmatizeVerbs(words):
	"""Lemmatize verbs in list of tokenized words"""
	lemmatizer = WordNetLemmatizer()
	lemmas = []
	for word in words:
		lemma = lemmatizer.lemmatize(word, pos='v')
		lemmas.append(lemma)
	return lemmas

def normalize(words):
	words = remove_non_ascii(words)
	words = to_lowercase(words)
	words = remove_punctuation(words)
	words = replace_numbers(words)
	words = remove_stopwords(words)
	return words
