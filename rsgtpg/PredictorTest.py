# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 08:56:23 2018

@author: florefe
"""

import Predictor as pr
import pandas as pd
import numpy as np

num_files_training_with = 4
predictor = pr.Predictor()
predictor.load_trainers(1,num_files_training_with)
#test_data = [[29.99254335, -95.42296647, '12/8/2017 1:33']]

inputs = pd.read_csv('../data/notification_submission.csv')

#predict arrival at current spot and store as delta
test_data = []
for i in range(0,6):
    test_data.append([[inputs['lat'][i], inputs['lng'][i],
                       inputs['route_begin'][i], inputs['timestamp'][i],
                       inputs['dest_lat'][i], 
                       inputs['dest_lng'][i]]])
    
delta =[]

for test in test_data:  
    for t in test:
        delta.append(predictor.predict(t[0], t[1], t[2], t[3]))
print (delta) 
#get prediction for destination
predictions = []
for test in test_data:  
    for t in test:
        predictions.append(predictor.predict(t[4], t[5], t[2], t[3]))
#subtract delta
for i in range(0,6):
    predictions[i] = predictions[i]-delta[i]
print (predictions)
    