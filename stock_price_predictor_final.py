# Data Preprocessing Template   

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Importing the dataset
dataset = pd.read_csv('data2.csv')
dataset=dataset.dropna(how='any')
dataset.shape
a=dataset.iloc[:,0].values
X = dataset.iloc[:, -2:].values
y = dataset.iloc[:, 4].values

#filtering dates
X=np.column_stack((X,np.array(list(map(int,("".join(filter(lambda c:c.isdigit(),s))for s in a))))))
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

su=0
z=np.column_stack((y_pred,y_test))
for row in z:
    temp=row[0]-row[1]
    print(temp)
    su+=abs(temp)
print("the average error is\tRs:- %f" %(su/len(z)))

"""#backward elemination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((6898, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [ 1, 2]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
print(sum(y)/len(y))
plt.scatter(X_train[:,2], y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('stock price prediction')
plt.xlabel('date')
plt.ylabel('price')
plt.show()
"""
"""
plt.scatter(X_test[300:,2], y_test[300:], color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('stock price prediction')
plt.xlabel('date')
plt.ylabel('price')
plt.show()"""