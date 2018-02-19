import pandas as pd
# # creating panda series from list

# # icecream=["Chocolate","Vanilla","Butter Scotch","Tuti Fruity"]
# # pd.Series(icecream)
# #
# # lottery =[2,4,5,8,6,28]
# # s_lottery= pd.Series(lottery,name="lottery")
# #
# # registration=[True,False,True,True]
# # s_bool=pd.Series(registration)
# #
# # # creating panda series from dictionary
# # results = {'maths' : 23 ,'science':22, 'english' : 20 , 'history' : 10}
# # pd.Series(results)
# #
# # # Attributes
# # print(s_lottery.values) # Returns values of the series
# # print(s_lottery.index) # Returns about index info  of the series
# # print(s_lottery.dtype) # Returns series type
# # print(s_lottery.size) # Returns length of series
# # print(s_lottery.name) # to get the name of the series declared on line 8
# # print(s_lottery.is_unique) # Returns true if series has unique value otherwise false
# #
# # # Methods
# # print('Sum of Series is %s'%s_lottery.sum())
# # print('Product of Series is %s'%s_lottery.product())
# # print('Mean of Series is %s'%s_lottery.mean())
# #
# # # Parameters and Arguments
# #
# # weekdays=["Monday","Tuesday","Wednesday","Thursday","Friday"]
# # num1=[1,2,3,4,5]
# # print(pd.Series(data=weekdays,index=num1))
# #
# read_csv Method

pokemon=pd.read_csv("practice_csv//pokemon.csv",usecols=["Pokemon"],squeeze=True) #  squeeze parameter's argument to import the data as a Series object instead of a DataFrame.
pokemon1=pd.read_csv("practice_csv//pokemon.csv",index_col="Pokemon",squeeze=True)
google=pd.read_csv("practice_csv//google_stock_price.csv",squeeze=True)

# # Head and tail Method
# pd.Series(pokemon.head(n=10)) # Create a new series with 10 element
# s_google_tail5=pd.Series(google.tail()) # Create a new series with default 5 element
# print(s_google_tail5)
#
# # Python Built-In Functions
#
# print(len(google))
# print(type(google))
# print(list(s_google_tail5)) # convert series to list
# print(dict(s_google_tail5)) # convert series to dictionary
# print(min(pokemon))
# print(max(google))
#
# # Sort_value method
#
# pd.Series(s_google_tail5).sort_values(ascending=False) # Returns new series in ascending order
# google.sort_values(ascending=True,inplace=True)  # place parameter on a Series method to permanently modify the object it is called on
# # or
# google=google.sort_values(ascending=False) # or we cane reassigning the new object to the same variable to permanently modify the object it is called on
#
# # Sort_index method
#
# google.sort_index() # sort by index and if we again sort the google variable by index it will go to it's original state
# google.sort_index(ascending=False,inplace=True) # overwrite  the object and stored desc order series
#
# # in keyword
# # to check if a value exists in either the values or index of a Series.
# # If the .index or .values attribute is not included, pandas will default to searching among the Series index.
#
# print("Pikachu" in pokemon.values) # return true if the element exits in series otherwise false
#
# print(100 in pokemon.index) # return true if the index  exits in series otherwise false
#
# # Extract Series Values by Index Position
# # Use bracket notation to extract Series values by their index position.
#
# print(pokemon[[200,1,14]])  # return series element from 200,1,14 index location
# print(pokemon[10:20])  # return  element from 10 to 20
# print(pokemon[:10])  # return element till 10
# print(pokemon[-10:])  # return  element 10 element from last
#
# # Extract Series Values by Index Label
#
# print(pokemon1.head())
#
# print(pokemon1[[100,134]])# still can extract ement by index postion
#
# print(pokemon1[["Electrode","Charmander"]])# extract element using index postion
#
# print(pokemon1["Bulbasaur":"Charmeleon"])# extract element from Bulbasaur to Charmeleon
# # in the results both the element will be inclusive

# Extract Series Values by .get value

# pokemon1.sort_index(inplace=True) # sorted the list for better performance
# print(pokemon1.get(key=["Electrode","Charmander"])) # extract element using index postion

print(pokemon1.get(key=["Electrode","Charmand"]))# in Case element is not present NaN and future warning message will be returened