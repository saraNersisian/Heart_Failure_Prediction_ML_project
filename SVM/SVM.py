#importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('SVM/input_data/heartTraining_V2.csv', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('SVM/input_data/heartTesting_V2.csv', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append(row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for i in c: #iterates over c
    for j in degree: #iterates over degree
        for k in kernel: #iterates kernel
           for l in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=i,degree=j,kernel=k,decision_function_shape=l)

                #Fit SVM to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                counter=0


                for m in range(len(dbTest)):
                    class_predicted = clf.predict([dbTest[m][0:11]])
                    if class_predicted==dbTest[m][11]:
                        counter = counter +1

                acc = counter/len(dbTest) #temporary value of accuracy

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here

                if counter==0:
                    highestAccuracy = acc
                    print("Highest SVM accuracy so far:" + str(highestAccuracy) + ", Parameters: c = " + str(i) + ", degree = " + str(j) + ", kernel = " + str(k) + ", decision_function_shape = " + str(l))
                else:
                    if acc >highestAccuracy: 
                        highestAccuracy = acc
                        print("Highest SVM accuracy so far:" + str(highestAccuracy) + ", Parameters: c = " + str(i) + ", degree = " + str(j) + ", kernel = " + str(k) + ", decision_function_shape = " + str(l))










