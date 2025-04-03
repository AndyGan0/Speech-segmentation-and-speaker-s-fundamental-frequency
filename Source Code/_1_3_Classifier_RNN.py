from _1_2_Import import importDataForRNN, RNN_batch_size

import numpy as np
import matplotlib.pylab as plt

import keras
from keras import Sequential
from keras._tf_keras.keras.layers import SimpleRNN, Dense



#   Importing the data using the method in _1_2_Import.py
Data, Labels = importDataForRNN()

#   Spliting into training set and test set
train_test_split = int(Data.shape[0]*0.8)
x_train = Data[0: train_test_split]
y_train = Labels[0:train_test_split]

x_test = Data[train_test_split:]
y_test = Labels[train_test_split:]




#   Creating the model
model = Sequential()
model.add(SimpleRNN(200, input_shape=(RNN_batch_size, 80),  return_sequences=True))
model.add(Dense(1, activation='sigmoid') )    #   output layer

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), 
              loss='binary_crossentropy', 
              metrics=['accuracy',
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall')])


model.summary()


#   Training
history =model.fit(x_train, y_train, 
          batch_size=64, 
          epochs=100, 
          verbose=2, 
          shuffle=True,
          validation_data=(x_test, y_test)  )


plt.plot(history.history['accuracy'], color='C0' )
plt.plot(history.history['val_accuracy'], color='red' )
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend( ['Training Data', 'Test Data'], loc='lower right' )

plt.figure()
plt.plot(history.history['precision'], color='C0' )
plt.plot(history.history['val_precision'], color='red' )
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('Epochs')
plt.legend( ['Training Data', 'Test Data'], loc='lower right' )

plt.figure()
plt.plot(history.history['recall'], color='C0' )
plt.plot(history.history['val_recall'], color='red' )
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('Epochs')
plt.legend( ['Training Data', 'Test Data'], loc='lower right' )

plt.figure()
plt.plot(history.history['loss'], color='C0' )
plt.plot(history.history['val_loss'], color='red' )
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend( ['Training Data', 'Test Data'], loc='upper right' )

plt.show()





#predict(x_train, y_train, "Training Set")
#predict(x_test, y_test, "Test Set")