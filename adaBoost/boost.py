#-------------------------------------------------------------------------
# AUTHOR: Anthony Spencer
# FILENAME: boost.py
# SPECIFICATION: description of the program
# FOR: CS 4210- project/adaboost
#-----------------------------------------------------------*/

from sklearn.ensemble import AdaBoostClassifier
import sklearn
import csv

X_test = []
Y_test = []
X_training = []
Y_training = []

h_acc=0
h_n=''
h_l=''

n_estimators = [5,10,50,250,500,1000]
l_rate=[0.01,0.1,.5,1]

with open('heartTraining.csv', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        if i > 0:
            X_training.append(row[:-1])
            Y_training.append(row[-1])

#reading the data in a csv file
with open('heartTesting.csv', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        if i > 0:
            X_test.append(row[:-1])
            Y_test.append(row[-1])

for n in n_estimators:
    for l in l_rate:

        
        ab = AdaBoostClassifier(n_estimators=n,  learning_rate=l)
        model = ab.fit(X_training, Y_training)

        y_pred = model.predict(X_test)
        tempacc=sklearn.metrics.accuracy_score(Y_test, y_pred)

        #print("for n = " + str(n) + " learning rate = "+ str(l))
        #print("Accuracy:",tempacc)

        if tempacc > h_acc:
            print('new highest acc =' + str(tempacc) + ' at n = ' + str(n) + " and learning rate = "+ str(l))
            h_acc=tempacc
            h_l=l
            h_n=n


print('final  highest acc =' + str(h_acc) + ' at n = ' + str(h_n) + " and L = "+ str(h_l))