#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
from collections import Counter
from nltk.util import ngrams
import re 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import math


# In[4]:


#loading saved posting lists and normalization values list
posting_list = np.load('posting_list_part1.npy',allow_pickle='TRUE').item()
norm_part1 = np.load('norm_part1.npy')
# print(happy[0])
title=np.load('titles_part1.npy')
N=len(norm_part1)


# In[37]:


#####ENTER YOUR QUERY HERE#####
query=input("Enter your query:")
print(query)


# In[38]:


#testing query
query = query.lower()
query = re.sub(r'[^\w\s]', '', query)
query_uni = nltk.word_tokenize(query)

#obtaining word count of query 
query_tf_raw = Counter(query_uni)

#calculating query term frequency weight
dictkeys = list (posting_list.keys())
query_tf_wt = dict([(key, val) for key, val in 
           query_tf_raw.items() if key in dictkeys])
for i in query_tf_wt:
    query_tf_wt[i]=1+math.log10(query_tf_wt[i])
    
#calculating query document frequency weight
query_df_wt = query_tf_wt
query_df_wt= {x: math.log10(N/len(posting_list[x])) for x in query_df_wt}
query_wt = query_tf_wt
query_wt={x: (query_df_wt[x]*query_tf_wt[x])*(query_df_wt[x]*query_tf_wt[x]) for x in query_wt}


#finding final query weights using ltc scheme
dlist=list(query_wt.values())
norm=(math.fsum(dlist))**0.5
query_wt={x: (query_df_wt[x]*query_tf_wt[x])for x in query_wt}
query_wt.update((k,v/norm) for (k,v) in query_wt.items())



# In[39]:


#finding weights of query words in docs using lnc scheme
normal= {i: [] for i in query_wt} 
final_score=[0]*N
for i in query_wt:
    for x in range(len(posting_list[i])):
        normal[i].append((posting_list[i][x][0],((1+math.log(posting_list[i][x][1]))/math.sqrt(norm_part1[posting_list[i][x][0]])*query_wt[i])))
        final_score[normal[i][x][0]]=final_score[normal[i][x][0]]+normal[i][x][1]



# In[40]:


#sorting the scores array and printing titles of results
final=np.argsort(final_score)

final_final=final[-10:]
for i in range(len(final_final)):
    print(title[final_final[9-i]])
    print(final_score[final_final[9-i]])
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




