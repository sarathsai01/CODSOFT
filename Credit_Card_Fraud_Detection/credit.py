import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

drop_cols = ["Unnamed: 0","trans_date_trans_time","cc_num","first","last","street","city","state","zip","dob","trans_num"]

train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

categorical = ["merchant","category","gender","job"]

for col in categorical:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

X_train = train.drop("is_fraud",axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud",axis=1)
y_test = test["is_fraud"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
