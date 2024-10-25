# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

##import data to read
train_data = pd.read_csv('rawdataset/train-data.csv')
print("\n\n train data : \n" ,train_data)

##test data:
test_data = pd.read_csv('rawdataset/test-data.csv' )
print("\n\n test data : \n ",test_data)

###info. of data:
print("\n\n\n info. of data :\n ")
print(train_data.info())


##find mean values:
print("\n\n\n describe data : \n" , train_data.describe())

###no. of rows and columns:
print("\n\n\n rows , columns : " ,train_data.shape)

###column names :
print("\n\n column names : \n " , train_data.columns)


###kilometers number of types count:
print("\n\n kilometer counts : \n ",train_data['Kilometers_Driven'].value_counts())


# Looking at the unique values of Categorical Features
print(train_data['Location'].unique())
print(train_data['Fuel_Type'].unique())
print(train_data['Transmission'].unique())
print(train_data['Owner_Type'].unique())

##find null values:
print("\n\n null values : \n ",train_data.isnull().sum())

##Let's Drop sum Rows which contains NULL values.
##ignored New_Price as there are many cell which contains NULL value in this column.

print("Shape of train data Before dropping any Row: ",train_data.shape)
train_data = train_data[train_data['Mileage'].notna()]
print("Shape of train data After dropping Rows with NULL values in Mileage: ",train_data.shape)
train_data = train_data[train_data['Engine'].notna()]
print("Shape of train data After dropping Rows with NULL values in Engine : ",train_data.shape)
train_data = train_data[train_data['Power'].notna()]
print("Shape of train data After dropping Rows with NULL values in Power  : ",train_data.shape)
train_data = train_data[train_data['Seats'].notna()]
print("Shape of train data After dropping Rows with NULL values in Seats  : ",train_data.shape)


###reset the index values after dropping columns:
train_data = train_data.reset_index(drop=True)

for i in range(train_data.shape[0]):
    train_data.at[i, 'Company'] = train_data['Name'][i].split()[0]
    train_data.at[i, 'Mileage(km/kg)'] = train_data['Mileage'][i].split()[0]
    train_data.at[i, 'Engine(CC)'] = train_data['Engine'][i].split()[0]
    train_data.at[i, 'Power(bhp)'] = train_data['Power'][i].split()[0]

##convert to float type
train_data['Mileage(km/kg)'] = train_data['Mileage(km/kg)'].astype(float)
train_data['Engine(CC)'] = train_data['Engine(CC)'].astype(float)


##Power(bhp) to float an error occured (Can't convert str to float : null)
print(train_data['Power'][76])

x = 'n'
count = 0
position = []
for i in range(train_data.shape[0]):
    if train_data['Power(bhp)'][i]=='null':
        x = 'Y'
        count = count + 1
        position.append(i)
print("\n\n dislplay x :"  , x)
print("\n\n count : " ,count)
print("\n\n position :" ,position)

###reset index:
train_data = train_data.drop(train_data.index[position])
train_data = train_data.reset_index(drop=True)

##data rows and colummn:
print("\n\n rows and column : " ,train_data.shape)

##convert power(bhp) to float:

train_data['Power(bhp)'] = train_data['Power(bhp)'].astype(float)

###display data head:
print("\n\n\n data head : \n " ,train_data.head())

####full new_price to new_car_price:
for i in range(train_data.shape[0]):
    if pd.isnull(train_data.loc[i,'New_Price']) == False:
        train_data.at[i,'New_car_Price'] = train_data['New_Price'][i].split()[0]

##convert str to float:
train_data['New_car_Price'] = train_data['New_car_Price'].astype(float)


###drop columns:
train_data.drop(["Name"],axis=1,inplace=True)
train_data.drop(["Mileage"],axis=1,inplace=True)
train_data.drop(["Engine"],axis=1,inplace=True)
train_data.drop(["Power"],axis=1,inplace=True)
train_data.drop(["New_Price"],axis=1,inplace=True)
train_data.drop(["Unnamed: 0"],axis=1,inplace=True)
###to avoid variations in data drop Company column
train_data.drop(["Company"],axis=1,inplace=True)
train_data.drop(["Location"],axis=1,inplace=True)
print(train_data.columns)
# Data Visualization
###visulization on price
f, ax = plt.subplots(figsize=(10,6))
sns.distplot(train_data['Price'])
plt.xlim([0,160])
plt.show()


##Fuel Type
var = 'Fuel_Type'
data = pd.concat([train_data['Price'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x=var, y="Price", data=data)
fig.axis(ymin=0, ymax=165);
plt.show()

###owner type:
var ='Owner_Type'
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.stripplot(x = var, y ='Price', data = train_data)
plt.show()

##data info:
print("\n\n data info : \n" , train_data.info())


###Working with Categorical Data

#Fuel_Type
#Transmission
#Owner_Type
#converting to labelencoding:

##Working for Fuel_Type:
print("\n\n fuel type value count :\n" ,train_data['Fuel_Type'].value_counts())
##onehot code:
Fuel_t = train_data['Fuel_Type']
Fuel_t = pd.get_dummies(Fuel_t,drop_first=True)
print("\n\n fuel type: \n ",Fuel_t.head())

### Working with Transmission
print("\n\n transmissoin value count : \n" ,train_data['Transmission'].value_counts())

##by applying one hot coding:
Transmission_ = train_data['Transmission']
Transmission_ = pd.get_dummies(Transmission_,drop_first=True)
print("\n\n transmission columns :\n " ,Transmission_.head())

### Working with Owner_Type
print("\n\n owner values counts : \n" , train_data['Owner_Type'].value_counts())
##by applyin label encode:
train_data.replace({"First":1,"Second":2,"Third": 3,"Fourth & Above":4},inplace=True)

###for check display data:
print(train_data['Owner_Type'].head())

###final dataset:
final_train = pd.concat([train_data,Fuel_t,Transmission_],axis=1)
print("\n\n final data :\n " ,final_train.head())

##drop clumns:
final_train.drop(["Fuel_Type","Transmission" ,"New_car_Price"],axis=1,inplace=True)
final_train.head()

###shape of final data:
print("\n\n final data rows ,column : \n ", final_train.shape)

###export to new csv file:
from pandas import DataFrame
DataFrame(final_train.to_csv("finaldataset/updatedataset.csv", index=False,header=True))

print("successfully data updated")


