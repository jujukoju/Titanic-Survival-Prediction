import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = ["Pclass", "Age", "Sex", "Fare", "Parch"]
train = train[features + ['Survived']].dropna()

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])

X = train[features]
Y = train['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

sample_input = pd.DataFrame({
    'Pclass': [1],
    'Age': [26],
    'Sex': [0],
    'Fare': [53.5],
    'Parch': [2]
})

prediction = clf.predict(sample_input)


def survival_count(prediction):
    if prediction == [1]:
        print("Yay, you survived!")
    else:
        print("Nope, no survival😔")


survival_count(prediction)

print("Model Accuracy:", clf.score(X_test, Y_test))
