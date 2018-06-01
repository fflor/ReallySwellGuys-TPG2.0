# -*- coding: utf-8 -*-


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt



# Importing the dataset
dataset = pd.read_csv('C:\\Users\\florefe\\Documents\\TrashPandaGames\\Dataset\\Telem.csv')

#convert Time to time since the start
dataset["Time"] = pd.to_datetime(dataset["Time"])
dataset["Time"] = (dataset["Time"]- dataset["Time"].min()) / np.timedelta64(1,'D')

#set x to location and y to time
x = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 0].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)



