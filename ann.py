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

# Making the predictions and evaluating the model ----------------------------------------------------------------------
