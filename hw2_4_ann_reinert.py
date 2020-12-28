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
from sklearn.metrics import r2_score
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense 

#Import Dataset
df = pd.read_csv('car_purchase_history.csv', encoding='iso-8859-1')

#Drop Gender
df = df.drop(['Gender'], axis=1) #Drop Gender


#Normalize the Data
print('Normalizing data using Z-Transformation Method...')
age_mean = df['Age'].mean()
age_std = df['Age'].std()
annsal_mean = df['Annual Salary'].mean()
annsal_std = df['Annual Salary'].std()
ccdebt_mean = df['Credit Card Debt'].mean()
ccdebt_std = df['Credit Card Debt'].std()
networth_mean = df['Net Worth'].mean()
networth_std = df['Net Worth'].std()

for indx, row in df.iterrows():
    df.at[indx, 'Age'] = (row['Age'] - age_mean) / age_std
    df.at[indx, 'Annual Salary'] = (row['Annual Salary'] - annsal_mean) / annsal_std
    df.at[indx, 'Credit Card Debt'] = (row['Credit Card Debt'] - ccdebt_mean) / ccdebt_std
    df.at[indx, 'Net Worth'] = (row['Net Worth'] - networth_mean) / networth_std   
print('Normalization process done...')

#Prep for data
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
#y = df['Car Purchase Amount']
y = y.values.reshape(-1,1)

# Build the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
model = Sequential()
model.add(Dense(25, input_dim=4, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#Train the Model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

#Predict the Purchase Amounts
y_predict = model.predict(X_test)
#X_test_sample = np.array([60000])
#y_predict_sample = model.predict(X_test_sample)

#Plot Loss Values
plt.plot(history.history['loss'])
plt.ylabel('Losses')
plt.xlabel('Epoch Number')

# CALCULATE R2 VALUE

print('R Square: %.2f' %r2_score(y_test, y_predict))

# HW 2-4
print('\n HW 2-4 \n')
predict_age = (40 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (10000 - ccdebt_mean) / ccdebt_std
predict_networth = (50000 - networth_mean) / networth_std

x_predict = np.array([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])

predict_amount = model.predict(x_predict)
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""Predicted purchase price of someone with an annual salary of $60,000: ${format(predict_amount[0][0], ',.2f')}\n""")

predict_age = (20 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (10000 - ccdebt_mean) / ccdebt_std
predict_networth = (50000 - networth_mean) / networth_std

x_predict = np.array([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])

predict_amount = model.predict(x_predict)
print('Age: 20 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""Predicted purchase price of someone with an annual salary of $60,000: ${format(predict_amount[0][0], ',.2f')}\n""")

predict_age = (40 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (50000 - ccdebt_mean) / ccdebt_std
predict_networth = (50000 - networth_mean) / networth_std

x_predict = np.array([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])

predict_amount = model.predict(x_predict)
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $50,000 | Net Worth: $50,000')
print(f"""Predicted purchase price of someone with an annual salary of $60,000: ${format(predict_amount[0][0], ',.2f')}\n""")

predict_age = (40 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (10000 - ccdebt_mean) / ccdebt_std
predict_networth = (80000 - networth_mean) / networth_std

x_predict = np.array([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])

predict_amount = model.predict(x_predict)
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $80,000')
print(f"""Predicted purchase price of someone with an annual salary of $60,000: ${format(predict_amount[0][0], ',.2f')}\n""")

predict_age = (60 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (5000 - ccdebt_mean) / ccdebt_std
predict_networth = (100000 - networth_mean) / networth_std

x_predict = np.array([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])

predict_amount = model.predict(x_predict)
print('Age: 60 | Annual Salary: $60,000 | Credit Card Debt: $5,000 | Net Worth: $100,000')
print(f"""Predicted purchase price of someone with an annual salary of $60,000: ${format(predict_amount[0][0], ',.2f')}\n""")
