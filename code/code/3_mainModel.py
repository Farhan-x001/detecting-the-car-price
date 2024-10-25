##import pickle lib. for pickle file upload
import pandas as pd
import numpy as np
import pickle

pred_ = pd.read_csv("finaldataset/pred.csv")
print("\n\n display data : " ,pred_.head())

##make data into array format:
x= np.array(pred_)
print("\n\n x np array : \n" , x)

##for predictoin upload algorithm:

pickellr = open("lr.pickle" , "rb")
pickle_lr = pickle.load(pickellr)

print("\n\n prediction by linear regression: \n" , pickle_lr.predict(x))


picklerf = open("randomforest.pickle" ,"rb")
pickle_rf = pickle.load(picklerf)

print("\n\n\n predictoin by random forest :\n ",pickle_rf.predict(x))