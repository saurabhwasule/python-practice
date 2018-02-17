# # Print and input 1
# var=input("Enter a number to print:") #Get the input from the user
# print('Entered number is %s'%(var)) #Print the output of the variable
# print('You have entered'+var) #Print using addition of strings.
# print('You have entered %s'%(var)) #Print using % method
# print('You have entered {}'.format(var)) #Print using format method

# # Print and input 2
# num1=int(input("Enter the first number:"))#Get 2 number input from the user. Store the numbers in variable named 'num1' and 'num2'
# num2=int(input("Enter the second number:"))
# print('The input are %s and %s'%(num1,num2)) #Print the outputs of the variables in same line
# print('Multiplication of '+str(num1)+' and '+ str(num2) +' is ' +str(num1*num2)) #Print using addition of strings.
# print('Multiplication of %s and %s is %s'%(num1,num2,num1*num2)) #Print using % method
# print('Multiplication of {0} and {1} is {2}'.format(num1,num2,num1*num2)) #Print using format method


# #Strings - Part 1
# string1=input("Enter a string:")
# print('Length of %s is %s'%(string1,len(string1)))#1.Length of the string.
# print('First letter is %s and %s of the %s'%(string1[0],string1[-1],string1))#2.First letter and last letter of the string.
# print('Upper case of %s is %s'%(string1,string1.upper()))#3.Upper case of the string.
# print('Lower case of %s is %s'%(string1,string1.lower()))#4.Lower case of the string.
# print('First two words is %s and Last words is %s of the %s'%(string1[0:1],string1[-1:-2],string1)) #5.Slice the first two and last two words of the string.

# #Strings - Part 2
#
# var1 = 'Hello\nworld' #input with escape characters and store in variable
#
#
# print('The first variable is')
# print(var1) #1.Print the input which consist of escape characters.(on 1st input)
# print('The Second variable is')
# print(r'Hello\nworld')#2.Print the input like you typed.(on 2nd input)

# # List -Part 1
#
# string1='Saurabh Wasule'
#
# list1=str(string1)# 1.Convert the string to list.
# print('First element and last element of list is %s and %s'%(list1[0],list1[-1]) )# 2.Display the first element and last element of list.
# print ('Element except fist and last element is %s'%(list1[1:len(list1)-1]))# 3.Display the elements except first and last elements in the list.
# print('Length of the %s is %s'%(string1,len(string1)))# 4.Display the length of the list.
# string1=list(list1)
# print(string1)# 5.Convert the list back to string and display them.
# # 6.Convert the string to list with respect to spaces i.e ['hello','world'] for input 'hello world'
# string2='hello world'
# list2=string2.split(' ')
# print(list2)

# # List -Part 2
# list1=[1,2,3]
# list2=[2,3,4]
#
# list3=list1+list2
# print(list3)# 1.Add two lists and print them.
# print('max and min in list is %s and %s respectively'%(max(list3),min(list3)))# 2.Find the maximum and minimum of third list.
# print('The sorted of list3 is %s'%(sorted(list3)))# 3.Sort the third list and display them.
# rev=sorted(list3,reverse=True)
# print('Reverse of list is %s '%(rev))# 4.Reverse the third list and display them.
# list3.remove(1)
# print(list3)# 5.Remove the element '2' in the list and print them.
# var=list3.pop(-1)
# print(var)# 6.Pop the last element from the list and print them


# # Tuples
#
# # 1.Create two tuples, 1st with values 12,3,4,5,5,7 and 2nd with 100
#
# tup1 = (12,3,4,5,5,7)
# tup2 = (100,)
# tup3 = tup1 + tup2# 2.Create another tuple assign it to concat of two tuples
# print(tup3)
# print('Max and min of tuple is %s and %s '%(max(tup3),min(tup3)))# 3.Display the max and min of the tuple.
# print(tup3[2:5])# 4. Do slicing in tuple and display them.
# #we cannot delete elemnt from tuple
# print('Length of tuple is %s'%(len(tup3)))# 5.Delete 3 from the tuple and display its length.
# tup3[0]=44  #'tuple' object does not support item assignment
# # 6.Try to replace 12 from it.

# # Set
#
# set1={1,2,3}# 1.Create a set.
# print(set1)
# list1=[1,2,3,2,5,1]
# set2=set(list1)
# print(set2)# 2.Remove the duplicates in the list.
# # print(set2[1:3])# Slicing in set is not permitted 3.Try slicing and indexing in the set
# # 4.Convert list to tuples and tuples to sets and display them.
# tup2=tuple(list1)
# set3=set(tup2)
# print(set3)
# # 5.Convert sets to tuples and tuples to list and display them .
# tup3=tuple(set2)
# list2=list(tup3)
# print(list2)
# # 6.convert string to tuples and set and reverse them back to string.
# string1="Hello world"
# tup4=tuple(string1)
# print(tup4)
# set4=set(string1)
# print(set4)
# string2="".join(tup4)
# string3="".join(set4)
# print(string2)
# print(string3)
# # add element in set
# set4.update('z')
# print(set4)

# #dictionary 1
#
# # Create a dictionary which consist of Item(keys) and Quantity(values) of items in the shop.
# dict1={'soap':10,'bread':5,'shampoo':8}
#
# #Create another dictionary which consist of Item(Keys) and Price(values) of the items in the shop
# dict2={'soap':20.50,'bread':25.99,'shampoo':80}
#
# # Display the item with quantity and the cost of item in a single line like
#
# print('The shop have %s quantities of soap which cost %s USD each'%(dict1['soap'],dict2['soap']))
# print('The shop have %s quantities of bread which cost %s USD each'%(dict1['bread'],dict2['bread']))
# print('The shop have %s quantities of shampoo which cost %s USD each'%(dict1['shampoo'],dict2['shampoo']))

# #dictionary 2
#
# dict1 = {'maths' : 23 ,'science':22, 'english' : 20 , 'history' : 10}
# print(dict1.keys())# 1.Display only the subjects that are in dict1.
# print(dict1.values())# 2.Display only the marks scored by student.
# del dict1['english']
# print(dict1)# 3.Delete the subject in which he got low marks and display the dictionary.
# dict1={'english':24} # 4.Change the mark scored in English to 24 and display the dictionary.
# # 5.Now try to get the value of the history if not available print proper message.
# print(dict1.get('history','history is not available'))

# dictionary 3

# 1. Add two dictionary and assign it to dict3 Display the dict3
dict1={'soap':10,'bread':5,'shampoo':8}
dict2={'tea':130,'sugar':40}
dict3={}
dict3.update(dict1)
dict3.update(dict2)
print('the value of dict3 %s'%(dict3))
# 2. Clear the dictionary dict2 Display the dict2
dict2.clear()
print(dict2)
del dict1# 3. Delete the dictionary dict1.
val = input('Enter subject to check ')
output = dict3.get(val.lower(),'The item is not in dictionary')
print(output)
# 4.Get input from the user and print the score in that subject if not available print proper message. (user input can be of any case.)

