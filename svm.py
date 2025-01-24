import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import joblib

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel="linear")  
clf.fit(x_train, y_train)

joblib.dump(clf, 'model.pkl')

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)


def predict_new_data():
    input_features = [input_features]
    prediction = clf.predict(input_features)[0]
    return cancer.target_names[prediction]
predict_new_data()