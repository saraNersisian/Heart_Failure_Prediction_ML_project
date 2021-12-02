import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./heart.csv')

# drop 0 values in the dataframe, replace them with not a number
df.replace(0, np.nan, inplace=True)

plt.figure(figsize = (9,9))

plt.subplot(2,2,1)
sns.scatterplot(x='Age', y='MaxHR', data=df, palette='crest').set(title='Age vs Max Heart Rate')

plt.subplot(2,2,2)
sns.scatterplot(x='Age', y='Cholesterol', data=df, palette='crest').set(title='Age vs Cholesterol')

plt.subplot(2,2,3)
sns.scatterplot(x='Age', y='RestingBP', data=df, palette='crest').set(title='Age vs Resting Blood Pressure')

plt.subplot(2,2,4)
sns.scatterplot(x='MaxHR', y='Cholesterol', data=df, palette='crest').set(title='Max Heart Rate vs Cholesterol')

plt.show()