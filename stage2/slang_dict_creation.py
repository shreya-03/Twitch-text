from bs4 import BeautifulSoup
import urllib3
import json

http = urllib3.PoolManager()
Abbr_dict={}

def getAbbr(alpha):
	global Abbr_dict
	r=http.request('GET','https://www.noslang.com/dictionary/'+alpha)
	soup=BeautifulSoup(r.data,'html.parser')
	for i in soup.findAll('div',{'class':'dictionary-word'}): 
		abbr=i.find('abbr')['title']
		Abbr_dict[i.find('span').text[:-2]]=abbr

if __name__ == "__main__":
	
	linkDict=[]
	for one in range(97,123):
		linkDict.append(chr(one))
	for i in linkDict:
		getAbbr(i)
	#with open('ShortendText.json','w') as file:
	#	jsonDict=json.dump(Abbr_dict,file)