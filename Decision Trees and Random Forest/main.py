from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = SVC(kernel='linear', C=3)
clf3 = DecisionTreeClassifier()
clf4 = RandomForestClassifier()

clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)
clf3.fit(x_train, y_train)
clf4.fit(x_train,y_train)

print("KNN: ", clf1.score(x_test, y_test))
print("SVC: ", clf2.score(x_test, y_test))
print("DecisionTreeClassifier: ", clf3.score(x_test, y_test))
print("RandomForestClassifier: ", clf4.score(x_test, y_test))