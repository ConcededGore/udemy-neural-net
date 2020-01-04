# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Prepossessing ---------------------------------------------------------------------------------------------------

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_x_geo = LabelEncoder()
x[:, 1] = LabelEncoder_x_geo.fit_transform(x[:, 1])
LabelEncoder_x_gender = LabelEncoder()
x[:, 2] = LabelEncoder_x_gender.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[1])], remainder='passthrough')
x = ct.fit_transform(x)
x = x[:, 1:] # This is to prevent the dummy variable trap (highly collinear variables)

"""  ---------------------------------------------- This is the code that prints the matrix to the temp file for testing
np.set_printoptions(threshold=sys.maxsize)
temp = open('temp.csv', 'w')
print(str(x), file=temp)
temp.close()
"""

# Splitting the data into a training set and a test set
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0) # means 20% of the data will be test data

# Feature scaling (making all the variables be on the same scale)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Building the ANN -----------------------------------------------------------------------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer / softmax is sigmoid but for more than one output
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN (Applying stochastic gradient descent) / adam is a very efficient stochastic gradient descent algorithm,
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # would use catagorical_crossentropy if more than one output

# Fitting the nueral network to the training data
classifier.fit(x_train, y_train, batch_size=10,epochs=100)

# Making the predictions and evaluating the model ----------------------------------------------------------------------

# Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print("Accuracy: " + str((cm[0,0] + cm[1,1]) / 2000))
