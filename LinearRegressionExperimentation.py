# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

datasetPD = pd.read_csv("BitcoinOnly.csv", 
                        parse_dates=[0],
                        names = ("Date", "Open", "High", "Low", "Close", "Volume", "Market", "Variance", "Volatility"),
                        usecols=[0,1,2,3,4,5,6,9,10], 
                        index_col=0,
                        skiprows=1
                        )

predictors  = pd.DataFrame(datasetPD, columns = ["High", "Low","Open", "Close", "Variance", "Volume", "Market", "Volatility"])
targetHigh = pd.DataFrame(datasetPD, columns=["High"])
targetLow= pd.DataFrame(datasetPD, columns=["Low"])

X1 = predictors[["Open", "Close", "Variance", "Volume", "Market"]]
y1 = targetHigh["High"]
y2 = targetLow["Low"]
X1 = sm.add_constant(X1)
 
modelHigh = sm.OLS(y1, X1).fit()
predictionsHigh = modelHigh.predict(X1)
 
modelLow= sm.OLS(y2, X1).fit()
predictionsLow = modelLow.predict(X1)
 
print(modelHigh.summary())
print(modelLow.summary())
 
lmHigh = linear_model.LinearRegression()
modelHigh = lmHigh.fit(X1, y1)
predHigh = lmHigh.predict(X1)
 
lmLow = linear_model.LinearRegression()
modelLow = lmLow.fit(X1, y2)
predLow = lmLow.predict(X1)
 
print(predHigh[0:5])
print(lmHigh.intercept_)
print(predLow[0:5])
print(lmLow.intercept_)

targetVol = pd.DataFrame(datasetPD, columns=["Volatility"])
X2 = predictors[["High", "Low", "Open", "Close", "Variance", "Volume", "Market"]]
y3 = targetVol["Volatility"]
X2 = sm.add_constant(X2)
 
modelVol = sm.OLS(y3, X2).fit()
predictionsVol = modelVol.predict(X2)
print(modelVol.summary()) 

targetMark = pd.DataFrame(datasetPD, columns=["Market"])
X3 = predictors[["High", "Low", "Open", "Close", "Variance", "Volume", "Volatility"]]
y4 = targetMark["Market"]
X3 = sm.add_constant(X3)
 
modelMark = sm.OLS(y4, X3).fit()
predictionsMark = modelMark.predict(X3)
print(modelMark.summary())

targetHigh = pd.DataFrame(datasetPD, columns=["High"])
targetClose= pd.DataFrame(datasetPD, columns=["Close"])

X4 = predictors[["Open", "Variance", "Volume", "Market", "Volatility"]]
y5 = targetHigh["High"]
y6 = targetClose["Close"]

X_train, X_test, y_train, y_test = train_test_split(X4, y5, test_size=0.2)

lm = linear_model.LinearRegression()
modelOpen = lm.fit(X_train, y_train)
predictionsOpen = lm.predict(X_test)

print("Score:", modelOpen.score(X_test, y_test))

plt.scatter(y_test, predictionsOpen)
plt.xlabel("True Values")
plt.ylabel("Predictions")

lmVol = linear_model.LinearRegression()
modelVol = lmVol.fit(X2, y3)
predVol = lmVol.predict(X2)
print("SKLearn predicted values", predVol[0:5])

