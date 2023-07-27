from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
data = load_breast_cancer()
# print(data)
print("--------------------------------")
print("Features: ", data.feature_names)
print("--------------------------------")
print("Targets: ", data.target_names)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data),np.array(data.target), test_size = 0.2)

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(x_train, y_train)

print("Training Data Predictions:")
counter = 0
for i, (y, y_pred) in enumerate(zip(y_test, clf.predict(x_test)), 1):
    info = f"x{i}: y_true: {data.target_names[y]}, y_pred: {data.target_names[y_pred]}"
    if data.target_names[y] != data.target_names[y_pred] :
        counter+=1
        info += f" - wrong! {counter}/{i}"
    print(info)
score = clf.score(x_test, y_test)
n = x_test.shape[0]

correct = score * n
print("Correct: ", correct, "/",n)
wrong = n-correct
print("Wrong: ", wrong, "/",n)
print("Model accuracy: ", score, "%")
