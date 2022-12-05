from keras.models import Sequential
from keras.layers import Dense, Input
import pandas as pd
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

data_set = pd.read_csv('diabetes.csv')

input = data_set.loc[:,data_set.columns!='Outcome']
output = data_set.loc[:,'Outcome']

#splitting the data set with 90% for training and 10% for validation
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.2, random_state = 1)

model = Sequential()

#creating the 3 layers, first is the input layer, the second is a hidden layer and the last one is the output layer
model.add(Input(shape=(8,)))
model.add(Dense(32,activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.L2(1e-4), activity_regularizer=regularizers.L2(1e-5)))
model.add(Dense(16,activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)))
model.add(Dense(1, activation='sigmoid'))

#defining the model with the learning rate and parameters we want to use
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#training the model
model.fit(x_train, y_train, epochs=200, batch_size=24)

#seeing the accuracy at validation
_, accuracy = model.evaluate(x_test, y_test)

print('Accuracy: %.2f' % (accuracy*100))
