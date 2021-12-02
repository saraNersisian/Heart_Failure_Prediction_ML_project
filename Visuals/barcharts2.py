import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./heart.csv')

plt.figure(figsize = (9,9))

plt.subplot(2,2,1)
sns.countplot(x='FastingBS', data=df, palette='crest')

plt.subplot(2,2,2)
sns.countplot(x='RestingECG', data=df, palette='crest')

plt.subplot(2,2,3)
sns.countplot(x='ExerciseAngina', data=df, palette='crest')

plt.subplot(2,2,4)
sns.countplot(x='ST_Slope', data=df, palette='crest')

plt.show()