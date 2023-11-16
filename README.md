## Import Libraries
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
## Loading Dataset
data = pd.read_csv("Real Estate.csv")
data.info()
data.describe()
data.plot()
sn.heatmap(data.corr())
sn.pairplot(data)
## Model Debelopment
X = data['X4 number of convenience stores']
Y = data['Y house price of unit area']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train.shape
model = LinearRegression()
model.fit(X,Y)
