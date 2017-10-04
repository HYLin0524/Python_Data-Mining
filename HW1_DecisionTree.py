# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:27:41 2016

@author: Acer7
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score , recall_score

#The Function for Encode
def coding(col,codeDict):
    colCoded = pd.Series(col , copy=True)
    for key , value in codeDict.items():
        colCoded.replace(key,value,inplace=True)
    return colCoded

#use Pandas Pakage to read data and transfer to DFrame
data = pd.read_csv('D:\\character-deaths.csv')
df = pd.DataFrame(data=data)#才可以使用concat方法

#data proceccing
df["Death Year"] = coding(df["Death Year"],{'297':1,'298':1,'299':1,'300':1}) #死亡年改1
df = df.fillna(0) #將NaN的值改成0s

#set the dummy var
alle = pd.get_dummies(data.Allegiances)
##刪除不必要的欄位
del df["Name"], df["Book of Death"], df["Death Chapter"], df["Allegiances"]

test = pd.concat([df,alle], axis=1) #test變數 -- 資料處理完成
#print(test)  
                 
#split the train and test set75 25 %
trainset = test[0:round(len(test)*0.75)] 
testset = test[round(len(test)*0.75):]
y=trainset["Death Year"]
x=trainset

#Train the Model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainset.values[:,1:],y)

#Predict the result
z = clf.predict(testset.values[:,1:])

#Caculate Accuracy
print("precision:%f" % precision_score(testset["Death Year"],z,average='macro'))
print("accuracy:%f" % accuracy_score(testset["Death Year"],z))
print("recall:%f" % recall_score(testset["Death Year"],z))

#graph decision tree
import pydotplus 
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data , max_depth=3 ,filled=True, feature_names = test.columns.values[1:])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("1007tree.pdf")

