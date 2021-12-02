import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./heart.csv')

plt.figure(figsize = (9,9))

plt.subplot(2,2,1)
sns.countplot(x='HeartDisease', data=df, palette='crest')

# seaborn computes the average age
plt.subplot(2,2,2)
sns.barplot(x='Sex', y='Age', data=df, palette='crest')

plt.subplot(2,2,3)
sns.countplot(x='Sex', data=df, palette='crest')

plt.subplot(2,2,4)
sns.countplot(x='ChestPainType', data=df, palette='crest')

plt.show()