from _1_2_Import import importDataStandard

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix



#   Creating the model
model = LinearRegression()

def predict(x_test, y_test, title):
    #   predicting
    predictions = model.predict(x_test)

    #   turning predictions from a probability to binary number (0 or 1)
    binary_predictions = np.where(predictions >= 0.5, 1, 0)

    accuracy = accuracy_score(y_test, binary_predictions)

    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    conf_matrix = confusion_matrix(y_test, binary_predictions)

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