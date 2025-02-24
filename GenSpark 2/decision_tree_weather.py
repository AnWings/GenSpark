import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the data(Generate by ChatGPT)
data = pd.read_csv('C:/Python Codes/GenSpark3/weather_data.csv')
print(data.sample(10))

# Preprocessing
print(data.isnull().sum())
print(data.duplicated().sum())

#split the data into test and train
y = data['Temperature']
X = data.drop('Temperature', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

# Decision Tree
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)