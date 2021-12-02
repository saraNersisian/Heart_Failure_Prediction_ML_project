import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./heart.csv')

plt.figure(figsize = (15,9))

sns.heatmap(df.corr(), annot=True, cmap='viridis')

plt.show()

