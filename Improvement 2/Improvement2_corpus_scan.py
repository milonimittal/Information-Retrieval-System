#!/usr/bin/env python
# coding: utf-8

# In[3]:


from gensim import corpora,models,similarities
from collections import defaultdict
from nltk.corpus import stopwords
import re
import numpy as np
from bs4 import BeautifulSoup
import nltk
import pickle


# In[4]:


#loading the file
f = open("wiki_01txt.txt", "r", encoding="utf8")


# In[5]:


#extracting the text
soup = BeautifulSoup(f, 'html.parser')
wiki_text = soup.get_text()


# In[6]:


corpus_text=[]
corpus_title=[]
for item in soup.find_all('doc'):
    item_text = item.get_text()
    item_title=item.get('title')
    item_text = item_text.lower()
    item_text = re.sub(r'[^\w\s]', '', item_text)
    corpus_text.append(item_text)
    corpus_title.append(item_title)   


# In[ ]:


#removing stop words
stop_words = ([set(stopwords.words('english'))])
texts = [[ word.lower() for word in document.split()
	   if word.lower() not in stop_words]
	   for document in corpus_text]

#building collective frequency table
freq = defaultdict(int)
for text in texts:
	for token in text: 
		freq[token] += 1

#removing words with frequency equal to 1
texts = [[token for token in text if freq[token] > 1] 
	for text in texts]

dictionary = corpora.Dictionary(texts)

#Document-Term-Matrix, doc2bow returns a sparse vector
corpus = [dictionary.doc2bow(text) for text in texts]  


tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=4000)  #from the gensim package
corpus_lsi = lsi_model[corpus_tfidf]



np.save('titles_improv2.npy',corpus_title)


# In[ ]:


import pickle
with open('corpus_lsi','wb') as corpus_lsi_file:
    pickle.dump(corpus_lsi, corpus_lsi_file)


# In[ ]:


with open('dictionary','wb') as dictionary_file:
    pickle.dump(dictionary, dictionary_file)


# In[ ]:


with open('lsi_model','wb') as lsi_model_file:
    pickle.dump(lsi_model, lsi_model_file)


# In[ ]:




