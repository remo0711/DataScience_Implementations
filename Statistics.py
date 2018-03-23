#Author: Karthik

import pandas as pd
import numpy as np
import matplotlib as plt
import scipy as sp
import sklearn as sk
from sklearn import datasets

        ############################
        ##### Loading the file #####
        ############################
iris = pd.read_csv("D:\MSBA\Python_Codes\iris.csv",header="infer")
iris.head()

#Indexing - through label
iris["sep_len"]
iris.loc[0:2,"sep_len"]
#Indexing - through location
iris.iloc[0:2,0]

#Groupby
iris.groupby('class').mean()

#Creating new column
a = np.array([0]*len(iris))
iris['sep_len_ind'] = a
iris['sep_len_ind'][iris['sep_len']>=iris['sep_len'].mean()] = 1

#New column using list comprehension and new line
iris['sep_wid_ind'] = [1 if iris['sep_wid'][i]>=iris['sep_wid'].mean() \
else 0 for i in range(len(iris))]

#getting original dataset
iris_data = iris.iloc[:,0:5]

        ############################
        ##### Basic Statistics #####
        ############################
iris_data.describe()
