# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.cluster import KMeans

#Data processing(remove the other column)
df = pd.read_excel('C:\\dataset.xlsx')
df = pd.DataFrame(df)
df = df["postContent"]

#斷字結果置放陣列
total=[]

#fetch the data and transfer to DF、List
for i in df.index:
    tags = jieba.analyse.extract_tags(str(df.iloc[i]), topK=10)
    #tags = jieba.cut(str(df.iloc[i]), cut_all=False)
    content = " ".join(tags)
    total.append(content)
#print(total)
    #tags = jieba.cut(str(df.iloc[i]), cut_all=False)
    #print("Split "+ str(i) + " ".join(tags))

#TF IDF Setting
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
#the related label
X = vectorizer.fit_transform(total)
#caculate weight
tfidf=transformer.fit_transform(X)

word=vectorizer.get_feature_names()
#transfer to array
feature=X.toarray()
weight=tfidf.toarray()
#print(word)
#print(feature)

kmeans = KMeans(n_clusters=20, random_state=0).fit(feature)

register = kmeans.labels_
results = pd.DataFrame(register)

results_list = register.tolist()

cluster_num = range(20)
print(kmeans.labels_)
#print(kmeans.cluster_centers_)
#print(kmeans.inertia_)
total_cluster = []
for i in cluster_num: #1~20群
    sum1 = 0
    cluster = []
    for j in results_list: #
        sum1 = sum1 + 1
        if i == j:
            cluster.append(sum1)
    print("第"+str(i+1)+"群:"+"".join(str(cluster)))
    #total_cluster.append(cluster)
#print(total_cluster)


