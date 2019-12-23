# Data Prepossessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print(str(dataset) + "\n")

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(str(x) + "\n")

# Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_x = LabelEncoder()
x[:, 0] = LabelEncoder_x.fit_transform(x[:, 0])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[0])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
x = ct.fit_transform(x)

print(str(x) + '\n')

LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

print(str(y) + '\n')

# Splitting the data into a training set and a test set
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0) # means 20% of the data will be test data

print(str(x_train) + '\n' + str(y_train) + '\n\n' + str(x_test) + '\n' + str(y_test) + '\n')

# Feature scaling (making all the variables be on the same scale)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print(str(x_train) + '\n\n' + str(x_test) + '\n')
