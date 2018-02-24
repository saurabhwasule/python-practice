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
# print(rev)
# s=pd.Series([1,2,3])
# print(s.sum()) # will show sum of series
#
# print(rev.sum()) # sum by column wise or  index wise
#
# print(rev.sum(axis="columns"))

# select one column from data frame

# print(nba["Name"])  # to select one column
# print(type(nba["Name"]))  # single column in dataframe is series
#
# print(nba[["Name","Salary"]].head(5))  # to select more than one  column
#
# select_col=["Name","Salary"]
# print(nba[select_col].tail(5))  # to select more than one column in better way


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
# # Broadcasting Operation -A broadcasting operation performs an operation on all values within a pandas object
# we'll apply mathematical operations to values in a DataFrame column (i.e. a Series) including the .add(), .sub(), .mul() and .div() methods as well as operator to do same function

# print(nba["Age"].add(5).head(3))
# print(nba["Age"].head(3)+5)  # addition operation using + operator
#
# print(nba["Age"].sub(5).head(3))
# print(nba["Age"].head(3)-5)
#
# print(nba["Weight"].mul(.453592).head(3))  #display Weight in Kg
# print(nba["Weight"].head(3)*.453592)
#
# print(nba["Salary"].div(1000000).head(3))  # display salary in Million
# print(nba["Salary"].head(3)/.1000000)
#
# nba.insert(7,column="Weight in Kg",value=nba["Weight"]*.453592)  # add derived column to existing data frame
# print(nba.head(3))

# # .value_counts method review
#
# print((nba["Position"].value_counts().head(1))) # Most popular postion in NBA team
# print((nba["Weight"].value_counts().tail())) # Most common weight in NBA team
# print(nba["Salary"].value_counts().head(5)) # Most popular salary in NBA team

# # Drop rows with null
#
# print(nba.head(3))
# print(nba.tail(3))
# print(nba.dropna().tail()) # will remove the row if any rows which contain null or NaN
# print(nba.dropna(how="all")) # will remove the row if all rows contain null or NaN
#
# print(nba.dropna(axis="columns")) # will remove the coluns if any of the columns value is null or NaN
#
# print(nba.dropna(subset=["Salary"])) # remove the row if there is null value in salary column

#Fill in null value

print(nba.fillna(0)) # fill every null value in data frame by 0
# in above cas all the null value in salary and college column are replace by 0 which is not good
# to solve this issue we cn do as below
nba.dropna(how="all",inplace=True)
nba["Salary"].fillna(0,inplace=True) # replace salary null with 0
nba["College"].fillna("None",inplace=True) # replace null in College column with "No College" String
print(nba)

# .astype() Method
# prerequisite is no null should be removed

print(nba.head(6))

print(nba.info())
nba["Salary"]=nba["Salary"].astype("int")  # convert float to int
nba["Number"]=nba["Number"].astype("int")  # convert float to int
nba["Age"]=nba["Age"].astype("int") # convert float to int
print(nba["Position"].nunique())
print(nba["Team"].nunique())
nba["Position"]=nba["Position"].astype("category") # columns with less cardinality to assign each distinct value thru pinter
nba["Team"]=nba["Team"].astype("category")
print(nba.info())
print(nba.head(6))


# Sort data frame by sort_values method

print(nba.sort_values(["Team","Name"],ascending=[True,False],inplace=True))  # sort the dataframe by team(ascending) and Name(decending) column in Ascending order

print(nba.head(3))

print(nba.sort_values("Salary",na_position="last"))  # sorting the column containing null

# Sort data frame by index using sort_index method


print(nba.sort_index(ascending=True))



