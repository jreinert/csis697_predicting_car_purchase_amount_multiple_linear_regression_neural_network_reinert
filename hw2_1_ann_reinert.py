# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:02:25 2020

@author: reine
"""
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

#Import Dataset
df = pd.read_csv('car_purchase_history.csv', encoding='iso-8859-1')

#Prepare For Data
X = df.iloc[:, 3:4]
y = df['Car Purchase Amount']
y = y.values.reshape(-1,1)

# Build the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#Train the Model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1)

#Predict the Purchase Amount
X_test_sample = np.array([20000])
y_predict_sample = model.predict(X_test_sample)
print(f"""\nPredicted purchase price of someone with an annual salary of $20,000: ${format(y_predict_sample[0][0], ',.2f')}\n""")

X_test_sample = np.array([40000])
y_predict_sample = model.predict(X_test_sample)
print(f"""Predicted purchase price of someone with an annual salary of $40,000: ${format(y_predict_sample[0][0], ',.2f')}\n""")

X_test_sample = np.array([60000])
y_predict_sample = model.predict(X_test_sample)
print(f"""Predicted purchase price of someone with an annual salary of $60,000: ${format(y_predict_sample[0][0], ',.2f')}\n""")


#Plot Loss Values
plt.plot(history.history['loss'])
plt.ylabel('Losses')
plt.xlabel('Epoch Number')

#Predict Purchase Amounts for Total ALL Test
y_predict = model.predict(X_test)
#print(y_predict_x_test)

"""CALCULATE R_SQUARE VALUE"""

sst = ((y_test - y_test.mean())**2).sum()
ssr = ((y_predict - y_test.mean())**2).sum()
sse = ((y_test - y_predict)**2).sum()

r2 = float(1 - (sse/sst))

print('r squared: %.2f' %r2)


