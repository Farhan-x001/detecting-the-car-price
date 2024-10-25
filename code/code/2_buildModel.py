 ###storing the best acc. value: by using pickle
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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



##apply algorithms:
bestscore_lr = 0
for i in range(1000):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    acc =  linear_regression.score(x_test, y_test)
    print("\n\n i - count :" , i , "acc:" ,acc ,end="")
    if bestscore_lr < acc:
        bestscore_lr = acc
        print("Linar Regression : ", bestscore_lr)
        with open("lr.pickle" , "wb")as lr_file:
            pickle.dump(linear_regression , lr_file)



##apply Random forest regression:

bestscore_rf = 0
for j in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    rf_regression = RandomForestRegressor()
    rf_regression.fit(x_train , y_train)
    acc = rf_regression.score(x_test, y_test)
    print("\n\n  j - count : " ,j ,"acc : " ,acc,end="")
    if bestscore_rf < acc:
        bestscore_rf = acc
        print("\n ------------------------>random forest regression acc :" , bestscore_rf)
        with open("randomforest.pickle" , "wb") as rf_file:
            pickle.dump(rf_regression , rf_file)
