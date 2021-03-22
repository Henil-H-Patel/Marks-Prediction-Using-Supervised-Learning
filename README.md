# Marks-Prediction-Using-Supervised-Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('hari.csv')
print(data)
""" first we show simple graph """
data.plot(x="Hours"  ,y="Scores" ,style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

"""After that, we need to extract the dependent and independent variables from the given dataset."""
x=data[['Hours']]
y=data[['Scores']]
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
"""We have split our data into training and testing sets, and now is finally the time to train our algorithm. Execute following command"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
""" y=mx+c in this equation we find out the c is intercept """
print(regressor.intercept_)
""" For retrieving the slope (m) """
print(regressor.coef_)
print("\n")
"""Now that we have trained our algorithm, it's time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make pre-dictions on the test data, execute the following script:"""
y_pred = regressor.predict(x_test)
"""To compare the actual output values for X_test with the predicted values, execute the following script:"""
print(y_pred)
print("\n")
print(y_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
x_new=[[9.25]]
print(regressor.predict(x_new))
