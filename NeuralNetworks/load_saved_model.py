import pandas as pd
from keras.models import load_model
import sklearn.preprocessing as pre

# Categorical data
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina" , "ST_Slope"]
label_encoder = pre.LabelEncoder()

# Load the trined model
model = load_model("output_data/NN_model.h5")

# load CSV data
data = pd.read_csv("./input_data/heart.csv")

# encoding the categorical data into numerical
for i in categorical:
    data[i] = label_encoder.fit_transform(data[i])

# Display encoded data
print(data)



X , Y = data.loc[:,'Age':'Oldpeak'] , data.loc[: , "HeartDisease"]

# Using nested [[]] because it puts it in a horizontal format
to_test = X.iloc[[914]]
print(to_test)


prediction = model.predict(to_test)[0][0]
print("Model prediction is : " + str(prediction) + " The patient is highly likely to have a heart disease!")




