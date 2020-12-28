# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:40:06 2020

@author: reine
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

# load dataset
df = pd.read_csv('car_purchase_history.csv', encoding='iso-8859-1')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# create training/testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# predict the results and show the regression line
y_predict = reg.predict(X_test)
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
r_squared = reg.score(X_test, y_test)

# print values
print('slope: ', reg.coef_)
print('intercept: ', reg.intercept_)
print('r squared: ', r_squared)
print('RMSE: ', mlr_rmse)

# predict unseen values
predict_amount = reg.predict([[1, 40, 60000, 10000, 50000]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""\nPredicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_amount = reg.predict([[0, 40, 60000, 10000, 50000]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_amount = reg.predict([[1, 20, 60000, 10000, 50000]])
print('Age: 20 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_amount = reg.predict([[1, 40, 60000, 50000, 50000]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $50,000 | Net Worth: $50,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_amount = reg.predict([[1, 40, 60000, 10000, 80000]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $80,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_amount = reg.predict([[1, 60, 60000, 5000, 100000]])
print('Age: 60 | Annual Salary: $60,000 | Credit Card Debt: $5,000 | Net Worth: $100,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

# HW 2-2 Average Amount Spent for Males/Females
print('\n HW 2-2: \n')
df2 = df[['Gender', 'Car Purchase Amount']]
is_male = df2['Gender'] == 0
df2_male = df2[is_male]

is_female = df2['Gender'] == 1
df2_female = df2[is_female]

male_mean = df2_male['Car Purchase Amount'].mean()
female_mean = df2_female['Car Purchase Amount'].mean()

print(f"""The mean purchase price by males (0) is: ${format(male_mean, ',.2f')}""")
print(f"""The mean purchase price by females (1) is: ${format(female_mean, ',.2f')}""")

# HW 2-3
# Drop Gender from df
print('\n HW 2-3: \n')
df = df.drop(['Gender'], axis=1)

# Normalize Data
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

print(df.head())

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
reg2 = LinearRegression()
reg2.fit(X_train, y_train)

# predict the results and show the regression line
y_predict = reg2.predict(X_test)
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
r_squared = reg2.score(X_test, y_test)

# print values
print('slope: ', reg2.coef_)
print('intercept: ', reg2.intercept_)
print('r squared: ', r_squared)
print('RMSE: ', mlr_rmse)

# HW 2-3
print('\n HW 2-3 \n')
predict_age = (40 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (10000 - ccdebt_mean) / ccdebt_std
predict_networth = (50000 - networth_mean) / networth_std

predict_amount = reg2.predict([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_age = (20 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (10000 - ccdebt_mean) / ccdebt_std
predict_networth = (50000 - networth_mean) / networth_std

predict_amount = reg2.predict([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])
print('Age: 20 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $50,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_age = (40 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (50000 - ccdebt_mean) / ccdebt_std
predict_networth = (50000 - networth_mean) / networth_std

predict_amount = reg2.predict([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $50,000 | Net Worth: $50,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_age = (40 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (10000 - ccdebt_mean) / ccdebt_std
predict_networth = (80000 - networth_mean) / networth_std

predict_amount = reg2.predict([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])
print('Age: 40 | Annual Salary: $60,000 | Credit Card Debt: $10,000 | Net Worth: $80,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

predict_age = (60 - age_mean) / age_std
predict_annualsal = (60000 - annsal_mean) / annsal_std
predict_ccdebt = (5000 - ccdebt_mean) / ccdebt_std
predict_networth = (100000 - networth_mean) / networth_std

predict_amount = reg2.predict([[predict_age, predict_annualsal, predict_ccdebt, predict_networth]])
print('Age: 60 | Annual Salary: $60,000 | Credit Card Debt: $5,000 | Net Worth: $100,000')
print(f"""Predicted purchase amount: ${format(predict_amount[0], ',.2f')}\n""")

