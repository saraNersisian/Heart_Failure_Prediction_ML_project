import pandas as pd

from sklearn.preprocessing import LabelEncoder

# This code is used to encode the categorical values in our dataset

df = pd.read_csv('./heart.csv')
df.head()

# encoding the categorical values
encoder = LabelEncoder()

categorical_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

df.to_csv('newHeart.csv', index=False)

df.head()