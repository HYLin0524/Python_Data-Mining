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
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score , recall_score
from sklearn.datasets import dump_svmlight_file

#The Function for Encode
def coding(col,codeDict):
    colCoded = pd.Series(col , copy=True)
    for key , value in codeDict.items():
        colCoded.replace(key,value,inplace=True)
    return colCoded

#use Pandas Pakage to read data and transfer to DFrame
data = pd.read_csv('C:\\character-deaths.csv')
df = pd.DataFrame(data=data)#才可以使用concat方法

#data proceccing
df["Death Year"] = coding(df["Death Year"],{'297':1,'298':1,'299':1,'300':1}) #死亡年改1
df = df.fillna(0) #將NaN的值改成0s

#set the dummy var
alle = pd.get_dummies(data.Allegiances)
##刪除不必要的欄位
del df["Name"]
del df["Book of Death"]
del df["Death Chapter"]
del df["Allegiances"]

test = pd.concat([df,alle], axis=1) #test變數 -- 資料處理完成
#print(test)
                 
#split the train and test set75 25 %
trainset = test[0:round(len(test)*0.75)] 
testset = test[round(len(test)*0.75):]
y=test["Death Year"]
x=test[np.setdiff1d(test.columns,["Death Year"])]
print(x)
print(y)

dump_svmlight_file(x,y,'out1.dat',zero_based=True,multilabel=False)

"""
#graph decision tree
import pydotplus 
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data , max_depth=3 ,filled=True, feature_names = test.columns.values[1:])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("1007tree.pdf")
"""
