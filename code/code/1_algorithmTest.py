import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression    ##for linear_regression
from sklearn.neighbors import KNeighborsRegressor       ##for knn regression
from sklearn.tree import DecisionTreeRegressor          ##for DecisionTreeRegression
from sklearn.ensemble import RandomForestRegressor       ##RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

# import dataset
data = pd.read_csv('finaldataset/updatedataset.csv')
print("\n\n dispalay data : \n " , data.head())

###data columns:
print("\n\n data columns : \n " ,data.columns)

###separate row and columns:
x = data.loc[:,['Year', 'Kilometers_Driven', 'Owner_Type', 'Seats',
                 'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)', 'Diesel',
                 'LPG', 'Petrol','Manual']]

y = data.loc[: ,['Price']]

print("\n\n x shape : \n" , x.shape)



###cinvert into array format:
x = np.array(x)
y = np.array(y)

# check x, y
print(x)
print(y)

# split test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# check train and test data length
print("x_test :",  len(x_test))
print("y_test :",  len(y_test))
print("x_trian :", len(x_train))
print("y_trian :", len(y_train))

##apply algorithms:
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
acc =  linear_regression.score(x_test, y_test)
print("Linar Regression : ", acc)


##apply k-nearest neighbours:
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train , y_train)
acc = knn.score(x_test , y_test)
print("knn acc : " , acc)


##apply DecisionTreeRegressor
dtregression = DecisionTreeRegressor()
dtregression.fit(x_train , y_train)
acc = dtregression.score(x_test , y_test)
print("decision tree regression acc :" , acc)

##apply Random forest regression:
rf_regression = RandomForestRegressor()
rf_regression.fit(x_train , y_train)
acc = rf_regression.score(x_test, y_test)
print("random forest regression acc :" , acc)


##apply support vector machine regression:
svm_regression = SVR()
svm_regression.fit(x_train , y_train)
acc = svm_regression.score(x_test , y_test)
print("svm_regression acc :" , acc)


##svm by linear regressipn :
svmlinear_regression = LinearSVR()
svmlinear_regression.fit(x_train , y_train)
acc = svmlinear_regression.score(x_test , y_test)
print("svmlinear_regression acc :" , acc)


###    classifiers:   ####
###classifier alogrithm wont work becoz prices value are in continous:
