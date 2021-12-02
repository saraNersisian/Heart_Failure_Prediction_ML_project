import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./heart.csv')
df.head()

x = df.drop('HeartDisease', axis = 1)
y = df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)

model = LogisticRegression(random_state = 0, max_iter = 1000).fit(x_train, y_train)
y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))