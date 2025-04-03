from _1_2_Import import importDataStandard

import numpy as np
import matplotlib.pylab as plt

import keras
from keras import Sequential



#   Importing the data using the method in _1_2_Import.py
Data, Labels = importDataStandard()

#   Spliting into training set and test set
x_train = Data[0:50000]
y_train = Labels[0:50000]

x_test = Data[50000:]
y_test = Labels[50000:]







#   Creating the model
model = Sequential()
model.add( keras.layers.Dense(80, input_shape=(80,), activation='relu') )
model.add( keras.layers.Dense(40, activation='relu') )
model.add( keras.layers.Dense(1, activation='sigmoid') )    #   output layer

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy',
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall')])


model.summary()


#   Training
history =model.fit(x_train, y_train, 
          batch_size=64, 
          epochs=30, 
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



