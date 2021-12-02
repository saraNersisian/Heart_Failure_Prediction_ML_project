import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./heart.csv')

df_age = df['Age']

plt.figure(figsize=(10,5))
sns.histplot(x=df.Age, palette=sns.color_palette("pastel"), kde=True)
sns.despine()
plt.show()

