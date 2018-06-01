# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\florefe\\Documents\\TrashPandaGames\\Dataset\\Telem.csv')

#convert Time to time since the start
dataset["Time"] = pd.to_datetime(dataset["Time"])
dataset["Time"] = (dataset["Time"]- dataset["Time"].min()) / np.timedelta64(1,'D')

# Taking care of missing data
filter = (dataset["Latitude"].notna()) & (dataset["Longitude"].notna()) & (dataset["Time"].notna())
dataset = dataset[filter]

#set x to location and y to time
x = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

#TO-DO
""""
turn into function that takes as parameters:
    current lat/long
    customer addrss or lat/long
then outputs:
    time it will take to arrive

"""