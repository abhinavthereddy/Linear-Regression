import numpy as np
import csv
from pandas import DataFrame, read_csv
import pandas as pd
from linear_regression import Linear_Regression

file = r'day (train).csv'
x_train = pd.read_csv(file)
y_train = np.array(x_train['cnt'])
x_train = x_train.drop(['cnt', 'dteday', 'registered', 'casual', 'instant'], axis = 1)
x_train = x_train.values

file2 = r'day (test).csv'
x_test = pd.read_csv(file2)
y_test = np.array(x_test['cnt'])
x_test = x_test.drop(['cnt', 'dteday', 'casual', 'registered', 'instant'], axis = 1)
x_test = x_test.values

model = Linear_Regression()
model.linear_fit(x_train, y_train)
print("Weights: ", model.get_weights())
print("")
print("Bias: ", model.get_bias())
print("")
y_test_predicted = model.predict(x_test)
print("Predicted Test Points: ", y_test_predicted)
print("")
print("Mean Squared Error: " , model.mean_squared(y_test, model.predict(x_test)))
