# predicting-the-salary-using-ml
Predicting the salary of an employee based on his previous position in the previous company by using current company database.

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:, 2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=y.reshape(-1, 1)
y=sc_y.fit_transform(y)

#Fitting the svr model to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#predicting the salary using input as no of years
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([6]).reshape(1,-1))))

#Visualization of SVR using graphs
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('truth or bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
