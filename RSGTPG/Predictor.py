# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:22:31 2018

@author: fflor
"""
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import mapper as map

class Predictor():
    def __init__(self):
        self.regressors = []
    #creates linear regression model
    def train(self, filekey):

        # Importing the dataset
        mapr = map.Mapper()
        dataset = mapr.get_entry(filekey)
        #dataset = pd.read_csv('..\\data\\tele.csv')
        #dataset = pd.read_parquet('..\\data\\data.parquet.gz')

        #convert Time to time since the start
        dataset['Time'] = pd.to_datetime(dataset['Time'])
        dataset['Time'] = (dataset['Time']- dataset['Time'].min()) / np.timedelta64(1,'m')

        # Taking care of missing data
        filter = (dataset['Latitude'].notna()) & (dataset['Longitude'].notna()) & (dataset['Time'].notna())
        dataset = dataset[filter]

        #set x to location and y to time
        x = dataset.iloc[:, 1:3].values
        y = dataset.iloc[:, 0].values

        # Splitting the dataset into the Training set and Test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

        # Fitting Multiple Linear Regression to the Training set
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

# =============================================================================
         #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x[:,0], x[:,1],y)
        plt.show()
# =============================================================================
        return regressor


    # Predicting the Test set results
    def load_trainers(self, first_file, last_file):
        training_set = []

        for i in range(first_file,last_file+1):
            training_set.append('tele'+str(i))
        for file in training_set:
            self.regressors.append(self.train(file))

    def predict(self, lat, long, time_begin):
        predictions = []
        for regressor in self.regressors:
            #take the lat long and plug it into the model
            destination = [[lat,long]]
            #get the predited time from start
            predictions.append(regressor.predict(destination))

        #average the predictions
        print (predictions)
        prediction = np.mean(predictions)
        print (prediction)
        #add prediction to time_begin
        predicted_time = pd.to_datetime(time_begin) + np.timedelta64(int(prediction), 'm')
        #return time
        return predicted_time

