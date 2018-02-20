import pandas as pd

#Series is 1 D
#dataframe is 2 dimension

nba=pd.read_csv("practice_csv/nba.csv")

# # shared Method and attribute
#
# print(nba.head(5))
# print(nba.tail(5))
#
# print(nba.index)
# print(nba.values)
# print(nba.shape)
# print(nba.dtypes)
# print(nba.columns)
# print(nba.axes) # bundle index and dtype together
# print(nba.info)# summary of dataframe
# print(nba.get_dtype_counts()) # give the summary of different data type in a dataframe

# difference  between shared methods

rev=pd.read_csv("practice_csv/revenue.csv",index_col="Date")
print(rev)
s=pd.Series([1,2,3])
print(s.sum()) # will show sum of series

print(rev.sum()) # sum by column wise or  index wise

print(rev.sum(axis="columns"))



# # .value_counts method review
#
# print((nba["Position"].value_counts().head(1))) # Most popular postion in NBA team
# print((nba["Weight"].value_counts().tail())) # Most common weight in NBA team
# print(nba["Salary"].value_counts().head(5)) # Most popular salary in NBA team
#
# # Add new column in panda dataframe
# # assign scalar value to sport column
#
# nba["Sports"]="Basketball" # add new column at the last
#
# print(nba.head(3))
# # Another way to add new column in any postion
# nba.insert(3,column="League",value="National Basketball association")
#
# print(nba.head(3))
#
# # Broadcasting Operation
#
#
#
