import functools

#object.__new__(cls[, ...])
#Called to create a new instance of class cls.
#Calls __init__() with new instance and other args from __new__

class A:
	def __new__(cls, v):
		self = object.__new__(cls)
		cls.v = v     #class variable
		return self   #auto calls __init__
	def __init__(self, v):  #v from __new__
		self.v = v*v      #instance variable 


#inheritance



class Base:
		def __init__(self, arg):
			self.arg = arg          #can only be accessed as instance.arg
		def __str__(self):
			return "Base"

		
#Py3.0 no notion of unbound methods.
#Python 3.0 will pass along an instance to methods only for through-instance calls. 
#When calling through a class, you must pass an instance manually only if the method expects one
		

@functools.total_ordering
class D(Base) :             #class var - all instance of same class share this, each derivered hierarchy gets own copy
	class_var = 1           #can be accessed as D.class_var or instance.class_var, Any self.class_var hides this
	def __init__(self, arg, arg1):
		super().__init__(arg)  # valid for new style, or can be done like Base.__init__(self, arg) , valid for all style
		self.new = arg1
		print("Inside D")
	def meth(self, arg):          #instance.meth(10) or D.meth(instance, 10)
		self.new = arg
		x = 1		# local var, not instance
	@classmethod
	def cmeth(cls, arg):       	#D.cmeth(10)
		cls.class_var = arg
	@staticmethod 
	def smeth(arg):            #instance.smeth(10) or D.smeth(10)
		print(arg)	
	#Property notation
	def p_get(self):
		return self._prop
	def p_set(self, val):
		self._prop = val
	def p_del(self):
		del self._prop	
	prop = property(p_get, p_set, p_del, None)   #instance.prop
	#Other methods
	def __str__(self):
		return "D derived from Base"
	def __eq__(self, other):
		if type(self) is type(other):
			return ((self.arg, self.new) == (other.arg, other.new))
		return NotImplemented
	def __lt__(self, other):
		if type(self) is type(other):
			return ((self.arg, self.new) < (other.arg, other.new))
		return NotImplemented
	def __hash__(self):
		return hash(tuple(sorted(self.__dict__.items())))	

	
	
#extending bulitin
class MyList(list):
	def __init__(self, lst):
		list.__init__(self,lst)
	def freq(self):
		return {ch: self.count(ch) for ch in self}
		