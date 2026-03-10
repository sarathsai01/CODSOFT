import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Churn_Modelling.csv")

df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)

le1 = LabelEncoder()
le2 = LabelEncoder()

df["Gender"] = le1.fit_transform(df["Gender"])
df["Geography"] = le2.fit_transform(df["Geography"])

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Accuracy:",accuracy)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
