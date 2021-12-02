
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pre

from keras import backend as k
from keras.models import Sequential
from keras.layers import *


# Categorical data
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina" , "ST_Slope"]

# load data.
# training and testing sets are combined here.
db = pd.read_csv("./input_data/heart.csv")
print(db.info() , '\n')

# encoding the categorical data into numerics so we can use them to train the model.
for col in categorical:
    # label_encoder object knows how to understand word labels.
    label_encoder = pre.LabelEncoder()
    # fit and transform data from the original set and put it back
    db[col] = label_encoder.fit_transform(db[col])

# Display encoded data.
print(db)

# # Dropping the cols with categorical value initially to make the model
# db = db.drop(columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina" , "ST_Slope"])
# print(db.head())

# shape X and y
# db.loc[rows , cols by lables]
X , y = db.loc[:, 'Age':'Oldpeak'] , db.loc[ : , 'HeartDisease']
X=np.asarray(X).astype(np.float32)
y=np.asarray(y).astype(np.float32)

# making test and train samples
# X_train includes 642 rows
# X_test includes 276
X_train , X_test, Y_train, Y_test = train_test_split(X , y , test_size=0.3, random_state=10)

# making the model
model = Sequential()


# input layer
model.add(Dense(11 ,input_dim=10, activation='relu'))

# hidden layer
model.add(Dense(20 ,activation='relu'))
model.add(Dense(21 ,activation='relu'))
model.add(Dense(21 ,activation='relu'))
model.add(Dense(21 ,activation='relu'))

# output layer
model.add(Dense(1, activation='linear'))

# Compile model
# Using mean squared error
model.compile(loss='mse' , optimizer="adam")
# Set learning rate manually to 0.001
k.set_value(model.optimizer.learning_rate, 0.001)


for i in range(100):
    # training
    model.fit(
        # Training feautures
        X_train,
        Y_train,
        # how many training passes
        epochs= 1175,
        # networks typically perform best when shuffle is true
        shuffle=True,
        # Mode detail info
        verbose=2
    )

    # Error rate with test data
    error = model.evaluate(X_test, Y_test , verbose=1)
    # f1
    # recall

    # Calculate accuracy and round to two decimal places
    accuracy = round((1-error)*100 , 2)
    print(f"Accuracy is : {accuracy}%")

    # put the results in the txt file for avg later
    with open("./output_data/accuracy.txt" , "a+") as f:
        f.write(str(accuracy))
        f.write("\n")


# Save the nural network
# .h5 aka htf5 format is a binary file format for python array data
model.save("./output_data/NN_model.h5")
print("Model saved to output_data.")




