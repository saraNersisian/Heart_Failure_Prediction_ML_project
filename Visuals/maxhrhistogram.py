import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./heart.csv')

df_MaxHR = df['MaxHR']

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='MaxHR', kde=True, hue='HeartDisease')
sns.despine()
plt.show()

# The maximum heartrate of people with heart disease appears to be 140 or below

# People without heart disease seem to have a higher heartrate,
# probably because they are more active, so the odds of them
# getting heart disease is lower