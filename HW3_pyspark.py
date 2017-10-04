# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:51:51 2016

@author: HONGYANG
"""

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

#load the data and set the spilt
data = MLUtils.loadLibSVMFile(sc, 'out1.dat')
(trainingData, testData) = data.randomSplit([0.75, 0.25])

#Train Model
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},impurity='entropy', maxDepth=5, maxBins=32)

#predict
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

#caculate the accuracy
Acc = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())

#def matrix
TP = labelsAndPredictions.filter(lambda (v, p): v==1 and p==1).count() / float(testData.count())
FP = labelsAndPredictions.filter(lambda (v, p): v==0 and p==1).count() / float(testData.count())
FN = labelsAndPredictions.filter(lambda (v, p): v==1 and p==0).count() / float(testData.count())
TN = labelsAndPredictions.filter(lambda (v, p): v==0 and p==0).count() / float(testData.count())

print('Accuracy = ' + str(Acc))
print('Precision = ' + str(TP/(TP+FP)))
print('Recall = ' + str(TP/(TP+FN)))

"""
print('Learned classification tree model:')
print(model.toDebugString())
"""

model.save(sc, "myDecisionTreeClassificationModel")
sameModel = DecisionTreeModel.load(sc,"myDecisionTreeClassificationModel")