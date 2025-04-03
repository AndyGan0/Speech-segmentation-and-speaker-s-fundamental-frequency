from _1_2_Import import importDataStandard, MedianFilter

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import joblib


#   Creating the model

#   model = SVC(kernel='linear')
#   model = SVC(kernel='poly', degree=3, gamma='auto')
#   model = SVC(kernel='sigmoid', gamma='scale')
model = SVC(kernel='rbf', gamma='scale')

def predict(x_test, y_test, title):
    #   predicting
    predictions = model.predict(x_test)

    predictions = MedianFilter(predictions, 5)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("-----------------------")
    print(title)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", conf_matrix)
    print("-----------------------")
    print()






#   Importing the data using the method in _1_2_Import.py
Data, Labels = importDataStandard()


#   Spliting into training set and test set
x_train = Data[0:50000]
y_train = Labels[0:50000]

x_test = Data[50000:]
y_test = Labels[50000:]


#   Training
model.fit(x_train, y_train)

predict(x_train, y_train, "Training Set")
predict(x_test, y_test, "Test Set")


#   saving the model
joblib.dump(model, 'svm_model.pkl')