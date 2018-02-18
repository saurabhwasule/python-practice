# creating panda series from list
import pandas as pd
icecream=["Chocolate","Vanilla","Butter Scotch","Tuti Fruity"]
pd.Series(icecream)

lottery =[2,4,5,8,6,28]
s_lottery= pd.Series(lottery,name="lottery")

registration=[True,False,True,True]
s_bool=pd.Series(registration)

# creating panda series from dictionary
results = {'maths' : 23 ,'science':22, 'english' : 20 , 'history' : 10}
pd.Series(results)

# Attributes
print(s_lottery.values) # Returns values of the series
print(s_lottery.index) # Returns about index info  of the series
print(s_lottery.dtype) # Returns series type
print(s_lottery.size) # Returns length of series
print(s_lottery.name) # to get the name of the series declared on line 8
print(s_lottery.is_unique) # Returns true if series has unique value otherwise false

# Methods
print('Sum of Series is %s'%s_lottery.sum())
print('Product of Series is %s'%s_lottery.product())
print('Mean of Series is %s'%s_lottery.mean())

# Parameters and Arguments

weekdays=["Monday","Tuesday","Wednesday","Thursday","Friday"]
num1=[1,2,3,4,5]
print(pd.Series(data=weekdays,index=num1))

# read_csv Method

s=pd.read_csv("pandas//pokemon.csv",usecols=["Pokemon"],squeeze=True) #  squeeze parameter's argument to import the data as a Series object instead of a DataFrame.
s1=pd.read_csv("pandas//google_stock_price.csv",squeeze=True)

# Head and tail Method
pd.Series(s.head(n=10)) # Create a new series with 10 element
s_google_tail5=pd.Series(s1.tail()) # Create a new series with default 5 element
print(s_google_tail5)

# Python Built-In Functions

print(len(s1))
print(type(s1))
print(list(s_google_tail5)) # convert series to list
print(dict(s_google_tail5)) # convert series to dictionary
print(min(s))
print(max(s1))

#