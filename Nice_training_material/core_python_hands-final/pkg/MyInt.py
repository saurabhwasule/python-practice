import functools 

@functools.total_ordering
class MyInt:
    def __init__(self,v):
        self.value = v
    def __add__(self, other):
        z = self.value + other.value
        return MyInt(z)
    def __str__(self):
        return "MyInt(%d)" % (self.value,)
    def __sub__(self, other):
        z = self.value - other.value
        return MyInt(z)
    def __eq__(self,other):
        return self.value == other.value 
    def __lt__(self,other):
        return self.value < other.value 
        
##################################
#from pkg.MyInt import Fraction 
#a = Fraction(1,2)
#b = Fraction(2,3)
#c = a + b 
#print(c) #Fraction(7,6)
#n1,d1 
#n2, d2
#result_n = n1*d2 + d1*n2 
#result_d = d1*d2 
#
#from pkg.MyInt import MyInt 
#a = MyInt(2)
#b = MyInt(3)
#c = a + b 
#print(c) # MyInt(5)

##################################
class Fraction:
    def __init__(self,n,d):
        self.n = n 
        self.d = d
    def __add__(self, other):
        result_n = self.n*other.d + self.d*other.n
        result_d = self.d * other.d 
        return Fraction(result_n,result_d)
    def __str__(self):
        return "Fraction(%d,%d)" % (self.n,self.d)
