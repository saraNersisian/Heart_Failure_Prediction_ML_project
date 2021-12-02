import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('./heartTraining.csv')
test_df = pd.read_csv('./heartTesting.csv')

x_train = train_df.drop('HeartDisease', axis = 1)
y_train = train_df['HeartDisease']

x_test = train_df.drop('HeartDisease', axis = 1)
y_test = train_df['HeartDisease']

model = LogisticRegression(random_state = 0, max_iter = 1000).fit(x_train, y_train)
y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))