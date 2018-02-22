### Install - for python3 , pip3 
$ pip2 install Django


###STEP 0: Python 2.7 , it's python.exe must be in PATH
#django-admin is installed in Scripts dir, must be in Path (> django-1.7) 
# < django-1.7 , use django-admin.py which is installed in Lib\site-packages\django\bin

#check help 
$ django-admin help startproject
$ django-admin help startapp

###STEP 1.1: Creating project 
django-admin startproject examplesite

###STEP 1.2 : check dir structure 

D:.
¦   db.sqlite3            #sqlite db file 
¦   manage.py
¦
+---examplesite
        settings.py       #setting files 
        settings.pyc
        urls.py           #route file 
        urls.pyc
        wsgi.py           #Web Server Gateway Interface file  exposing 'application'
        wsgi.pyc
        __init__.py
        __init__.pyc
        
        
###STEP 1.3  : check quickly 
$ cd examplesite
$ python manage.py help runserver
$ python manage.py runserver  8000 #<<port>>

#Open  http://127.0.0.1:8000/


###STEP 1.4 : check settings at examplesite\settings.py 

    DEBUG configuration 
    ALLOWED_HOSTS
    INSTALLED_APPS for default apps 
    MIDDLEWARE_CLASSES for pre/post processing of requests/response 
    ROOT_URLCONF name
    TEMPLATES
    WSGI_APPLICATION name 
    Database
    Password validation
    Internationalization
    Static files (CSS, JavaScript, Images)
    
#To get attributes programitically 
from django.conf import settings #check dir(settings)
settings.STATIC_URL  #'/static/'

###STEP 1.5: Quick add one route in examplesite\urls.py 

1. When Django starts , it check root url file from examplesite\settings.py 
    ROOT_URLCONF = 'examplesite.urls'

2. examplesite\urls.py file is processed , urlpatterns is [url, url, ...] 
    django.conf.urls.url(regex, view, kwargs=None, name=None)
    view : callable object 
    kwargs : allows to pass additional arguments to the view function 
    name: used for performing  URL reversing ie go to view callable from 'name' (name must be unique)
            for example 
            •In templates: Using the url template tag.
            •In Python code: Using the reverse() function.
            •In higher level code related to handling of URLs of Django model instances: The get_absolute_url() method.
         

3.  Add following 
#examplesite/urls.py

from . import views
urlpatterns = [                #previously it is patterns('', url...)
    #....
    url(r'^hello/', views.index),   #matches  http://address:port/hello and calls index method of views module 
    url(r'^hello2/', include('dummy.urls')), #match .../hello2/ and pass the remaining strings to urls.py of module dummy 
]
#Create examplesite/view.py 

from django.http import HttpResponse

def index(request):  #all methods take first arg as HttpRequest and returns HttpResponse 
    return HttpResponse("Hello, world")


#create module dummy under root dir 
D:.
¦   ...
+---dummy
¦       urls.py
¦       views.py
¦       __init__.py
¦
+---examplesite
        ...
    
#dummy/view.py 
from django.http import HttpResponse
def index(request, name):           #comes from url's (\w+)
    return HttpResponse("Hello " + name)


#dummy/urls.py 
from django.conf.urls import url, include 
from . import views
urlpatterns = [    
    url(r'^(\w+)/$',  views.index, name = "hello2index"),    #views 's index method has 2nd arg as (\w+)
]

4. Important attributes of django.http.HttpRequest 
HttpRequest.body        raw HTTP request body as a byte string
HttpRequest.path        example: /music/bands/the_beatles/
HttpRequest.get_full_path()  example: "/music/bands/the_beatles/?print=true"
HttpRequest.method      'GET' or 'POST' or others 
HttpRequest.content_type
HttpRequest.GET          QueryDict ,  dictionary-like object , access param as ['param']
HttpRequest.POST         QueryDict ,  dictionary-like object 
HttpRequest.FILES       {'name':  UploadedFile_instance}  , Must be : enctype="multipart/form-data" and <form> contains <input type="file" name="" />    
HttpRequest.META        All headers in dictionary-like object

#Example of django.http.request.QueryDict - in general immutable or use mutable=True in ctor
#create a shell 
$ python manage.py shell 
from django.http.request import QueryDict
q = QueryDict('a=1&a=2&c=3')  #<QueryDict: {'a': ['1', '2'], 'c': ['3']}>
q['a']          # u'2'  #last item 
q.getlist('a')  #[u'1', u'2']  #all items as list 
q.lists()       #[(u'a', [u'1', u'2']), (u'c', [u'3'])]
'a' in q        # True
q.urlencode()   # u'a=1&a=2&c=3'

#urldecode is automatic  
q = QueryDict('a=1+3&a=2&c=3') #<QueryDict: {u'a': [u'1 3', u'2'], u'c': [u'3']}>
q = QueryDict('a=1%203&a=2&c=3') #<QueryDict: {u'a': [u'1 3', u'2'], u'c': [u'3']}>



###STEP 1.6: Then create a application, inside examplesite

$ python manage.py startapp books 

#dir structure 
D:.
¦   db.sqlite3
¦   manage.py
¦
+---books
¦   ¦   admin.py
¦   ¦   apps.py
¦   ¦   models.py
¦   ¦   tests.py
¦   ¦   views.py
¦   ¦   __init__.py
¦   ¦
¦   +---migrations
¦           __init__.p
¦
+---dummy
¦       urls.py
¦       views.py
¦       __init__.py
¦
+---examplesite
        settings.py
        settings.pyc
        urls.py
        urls.pyc
        views.py
        views.pyc
        wsgi.py
        wsgi.pyc
        __init__.py
        __init__.pyc


###STEP2.0: Mysql setup
#installing  MySQLdb #for Py3.x - use Mysqlclient, from  http://www.lfd.uci.edu/~gohlke/pythonlibs/
pip2 install mysql-python 

#check by
import MySQLdb

#start MySQL and check 
cygwin$ /usr/bin/mysqld_safe &
#or run it services.msc
#shutting down
mysqladmin.exe -h 127.0.0.1 -u root   --connect-timeout=5 shutdown
#mysql admin #  default port 3306, 
mysql -u root    -h 127.0.0.1 
#few commands
show databases;
create database django;
use django;
show tables;
create table employes ( id INT PRIMARY KEY, first_name VARCHAR(20), last_name VARCHAR(20), hire_date  DATE);
desc employes;
insert into employes values (3, "das", "das", '1999-03-30');
select * from employes; 


#examplesite/settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'django',
        'USER': 'root',
        'PASSWORD': '',
        'HOST': '127.0.0.1',
    }
}

#Check by
$ python manage.py shell

from django.db import connection
cursor = connection.cursor()
cursor.execute("select * from employes")

results = cursor.fetchall()
results
for result in results:
    print(result[0], result[1], result[2])


cursor.close()



###STEP3.0: MVC 
#Update books/models.py, books/view.py and html template file (books/templates/) for your site in books
#also update admin module 


###STEP3.1: MVC - Admin module - you can create user, groups and your apps 
#1. check required settings in settings.py , TEMPLATES::OPTIONS::context_processors
#MIDDLEWARE_CLASSES and INSTALLED_APPS
#2. create ModelAdmin and register, check examplesite\books\admin.py 
#3. Hook to <<project>>\urls.py, by url(r'^admin/', admin.site.urls) (default), (note <Django-1.9, url(r'^admin/', include(admin.site.urls)) )


#To override an admin template for a specific app, 
#copy and edit the template from the django/contrib/admin/templates/admin directory, 
#and save it to <<project>>\templates\<<app>>\
#Note your app must come before 'django.contrib.admin' in settings.INSTALLED_APPS

#To enable above dir , use 'django.template.loaders.filesystem.Loader' in settings.TEMPLATES.OPTIONS.loaders
#and update 'DIRS': [os.path.join(BASE_DIR, 'templates')] in  settings.TEMPLATES.DIRS

#Note you have to create superuser to use the admin module 

#file books\admin.py :
from django.contrib import admin


from books.models import Book #note import is always from root, hence .\books\models.py

#ModelAdmin options - many options for customizing the interface. 
#All options are defined on the ModelAdmin subclass as class field 
#https://docs.djangoproject.com/en/1.10/ref/contrib/admin/#modeladmin-options

class BookAdmin(admin.ModelAdmin):
	fields = ['pub_date', 'name']  # reorder

admin.site.register(Book, BookAdmin)  # register your models and ModelAdmin instances with instance of django.contrib.admin.sites.AdminSite created by django.contrib.admin.site 
                                      #Customize the AdminSite for custom behaviour 

#By default following would do if there is no customization of BookAdmin
#admin.site.register(Book)




###STEP3.2: MVC- Model - the database tables
#books/models.py
# models.py (the database tables)
from django.db import models

class Book(models.Model):                   #two field  
	name = models.CharField(max_length=50)
	pub_date = models.DateField()
	
	def __unicode__(self):              # __str__ on Python 3, __unicode__ in Python2
		return self.name

###STEP3.3: MVC - Views - (the business logic)
#books/views.py
from django.shortcuts import render_to_response
from models import Book

def latest_books(request):
	book_list = Book.objects.order_by('-pub_date')
	return render_to_response('latest_books.html', {'book_list': book_list})

    
###STEP3.4: MVC - view template 

#books/templates/latest_books.html : {% %} for template code, {{ }} for var name 
{% load staticfiles %}
<!DOCTYPE>
<html>
    <head>
        <title>DJANGO</title>
        <link rel="stylesheet" type="text/css" href="{% static 'style.css' %}" />
    </head>
    <body>
        {% for book in book_list %} 
            <h1>{{ book.name }}  {{ book.pub_date}}</h1>
        {% endfor %} 
    </body>
</html>

###STEP3.5: MVC - view -  Addition of Static files 
#For any user defined static files, add settings.STATICFILES_DIRS

STATICFILES_DIRS = (
    "books/static",
    "books/static/images",
   )

#<<app>>/static/style.css
body {
    background: white url("images/background.gif") no-repeat right bottom;
}

h1 {
    color: red;
}


###STEP3.6: MVC - test  Adding testing of Models 
#books/tests.py
from django.test import TestCase
import datetime
from django.test import Client
from books.models import Book

class BookTests(TestCase):

	def setUp(self):
		# django uses separate table test_books in DB  and removes after test
		# hence populate few data in that else test would fail
		Book.objects.create(name="first book", pub_date=datetime.date.today())
		Book.objects.create(name="first book 12", pub_date=datetime.date.today())
		self.c =  Client()

		
	def test_sample(self):
		"""
		Something should be present
		"""
		response = self.c.get('http://127.0.0.1:8000/latest/')
		# we can call unittest methods
		self.assertRegexpMatches(response.content, "first book")

	def test_not_empty(self):
			"""
			Something should be present
			"""
			book_list = Book.objects.order_by('-pub_date')
			# we can call unittest methods
			self.assertTrue(book_list)






###STEP4.1:Update examplesite/urls.py to include url to be handled


import books.views

urlpatterns = [
    url(r'latest/$', books.views.latest_books),
    #...
]


###STEP4.2: Modify INSTALLED_APPS in examplesite/settings.py to include books application
##Note order is significant , first means apps' template file would override all below's
INSTALLED_APPS = {
...
'books',
}

###STEP5.0: Activating models

#Notify django that models is changed 
#Must when you change model ***
$ python manage.py makemigrations books

#(creates migrations/0001_initial.py) and create migration code. 
$ cat books/migrations/0001_initial.py

###STEP5.1: Check the db creation command by  
#(0001 is parameter which denotes which migration point to display)

$ python manage.py sqlmigrate books 0001


#Note- check DB tablename, must be in lowercase. eg books_book
#If you would like to change DB table name use Meta options
#eg: in models.py/Book class

class Book(models.Model):
	name = models.CharField(max_length=50)
	pub_date = models.DateField()
	class Meta:
		db_table = 'books'
		


###STEP 5.2:Create the Databse tables now
$ python manage.py migrate

#Check from SQL , note many auth_* and django_* tables are created along with books_book
show tables;

### STEP 5.3:Check everything is OK
$ python manage.py shell


from books.models import Book  

Book.objects.all() # []

# Create New
# for DateTimeField use  timezone.now() from django.utils import timezone
import datetime
b1 = Book(name="first book", pub_date=datetime.date.today())

# Save the object into the database. You have to call save() explicitly.
b1.save()

#Get object
Book.objects.all()
Book.objects.filter(id=1)
Book.objects.filter(name__startswith='first')
Book.objects.filter(name__endswith='12') #[]


###STEP 6.1:Testing
# Django uses test DB , hence  populate test DB at first (during setUp phase)
#-v is verbose level 
$ python manage.py test -v 3  books

###STEP 6.2 : Run the server by
$ python manage.py runserver

#If below error
Error: [Errno 10013] An attempt was made to access a socket in a way forbidden b
#change port
python manage.py runserver 8080

#Then check at   http://127.0.0.1:8000/admin  or  http://127.0.0.1:8000/latest
#admin password: admin/admin

#Note: must create superuser 
#Create  superuser eg  admin/adminadminadmin
$ python manage.py createsuperuser 

#change passowrd
$ python manage.py changepassword <user_name>

#To give a normal user privileges, open a shell with python manage.py shell and try:
from django.contrib.auth.models import User
user = User.objects.get(username='normaluser')
user.is_superuser = True
user.save()

#Iterate users/superusers
from django.contrib.auth.models import User
User.objects.filter(is_superuser=True)

#then change password
usr = User.objects.get(username='your username')
usr.set_password('raw password')
usr.save()







###STEP 7.0:Under apache httpd with mod_wsgi - enable mod_wsgi in http.conf and add below
#copy examplesite to C:/indigoampp/apache-2.2.15/wsgi-bin/django/
WSGIScriptAlias  /first "C:/indigoampp/apache-2.2.15/wsgi-bin/django/examplesite/examplesite/wsgi.py"
WSGIPythonPath   "C:/indigoampp/apache-2.2.15/wsgi-bin/django/examplesite/"

<Directory "C:/indigoampp/apache-2.2.15/wsgi-bin/django/examplesite/examplesite">
	Order allow, deny
   Allow from all
</Directory>

# Then check 
http://localhost/first/latest

#NOTE:
If multiple Django sites are run in a single mod_wsgi process, 
all of them will use the settings of whichever one happens to run first. 
#This can be solved by changing:
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{{ project_name }}.settings")
#in wsgi.py, to:
os.environ["DJANGO_SETTINGS_MODULE"] = "{{ project_name }}.settings"



###STEP 7.1: Using static files under apache

#In setings.py, set
# where static files should be loaded
STATIC_ROOT = 'C:/indigoampp/apache-2.2.15/htdocs/dstatic/'



###STEP 7.2:Configure your web server to serve the files in STATIC_ROOT under the URL STATIC_URL. 
#For example, in httpd.conf
Alias /static       "C:/indigoampp/apache-2.2.15/htdocs/dstatic/"
Alias /static/admin "C:/indigoampp/apache-2.2.15/htdocs/dstatic/admin"

<Directory "C:/indigoampp/apache-2.2.15/htdocs/dstatic">
	Order allow,deny
   Allow from all
</Directory>

###STEP 7.3 :Then execute to copy all static files to STATIC_ROOT (under windows command prompt)
$ python manage.py collectstatic

###STEP7.4: Final testing 
http://localhost/first/latest or   http://localhost/first/admin


#----------------------------------------------------------------------------

###Django - using django models from outside 
#dir structure 
D:.
¦   yourcode.py
¦   
¦
+---myapp
¦   ¦   settings.py
¦   ¦   models.py

#myapp/models.py  - all model code



#yourcode.py
import os
import django
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "myapp.settings"
)
django.setup()
from myapp.models import *
#now use all ur models 

#myapp/settings.py
from django.conf import settings
settings.configure(
    DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'django',
        'USER': 'root',
        'PASSWORD': '',
        'HOST': '127.0.0.1',
     },
    },
    TIME_ZONE='America/Montreal',
)

###Advanced Models-Example
$ python manage.py startapp modelex  

#Include modelex in INSTALLED_APPS in myproject/settings.py  (ie examplesite/settings.py)

#file : modelex/models.py

from django.db import models

class Publisher(models.Model):
	name = models.CharField(max_length=30)
	address = models.CharField(max_length=50)
	city = models.CharField(max_length=60)
	state_province = models.CharField(max_length=30)
	country = models.CharField(max_length=50)
	website = models.URLField()

class Author(models.Model):
	salutation = models.CharField(max_length=10)
	first_name = models.CharField(max_length=30)
	last_name = models.CharField(max_length=40)
	email = models.EmailField()
	headshot = models.ImageField(upload_to='/tmp')

class NewBook(models.Model):
	title = models.CharField(max_length=100)
	authors = models.ManyToManyField(Author)
	publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)
	publication_date = models.DateField()
    serial_no  = models.IntegerField()



# with migrations 
$ python manage.py makemigrations modelex
$ python manage.py sqlmigrate modelex 0001
 
##Go through the sql commands
#Note PRIMARY KEY id is generated automatically for each table 
#Note that ManyToMany relation generates a new table with forien keys of the relations

#Create the Databse tables now
$ python manage.py migrate

#Note: any change in models.py must be accompanied by 
$ python manage.py makemigrations modelex
$ python manage.py migrate

### Django - Python3 and > django-1.5

#working with same code in Python3 and Python2 - Unicode literals
#add below in each module 
from __future__ import unicode_literals 
#then , Removing the u prefix before unicode strings;

#working with same code in Python3 and Python2 - String handling
Python 2’s unicode type was renamed str in Python 3, 
str() was renamed bytes in Py 3

#working with same code in Python3 and Python2 - __str__() and  __unicode__() methods

In Python 2,  __str__() and  __unicode__() methods return str (bytes) and unicode (text) 
In Python 3, there’s __str__(), which must return str (text)

Use python_2_unicode_compatible decorator to use seamlessly in Py2 and Py3

#example 
from __future__ import unicode_literals
from django.utils.encoding import python_2_unicode_compatible

@python_2_unicode_compatible
class MyClass(object):
    def __str__(self):
        return "Instance of my class"
        
        
        
###Models - Details of Model Layer

from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)

#Creates, SQL=>
CREATE TABLE myapp_person (
    "id" serial NOT NULL PRIMARY KEY,
    "first_name" varchar(30) NOT NULL,
    "last_name" varchar(30) NOT NULL
);


##Another Example

from django.db import models

class Musician(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    instrument = models.CharField(max_length=100)

class Album(models.Model):
    artist = models.ForeignKey(Musician)
    name = models.CharField(max_length=100)
    release_date = models.DateField()
    num_stars = models.IntegerField()


## DELATION of a table
$ python manage.py dbshell
# in dbshell
show tables;		# for sqllite SELECT * FROM sqlite_master WHERE type='table';
DROP TABLE appname_modelname;
exit; 			# for sqlite, .exit

##Modifying a column in table
#correct your models.py file with the desired data types 
python manage.py makemigrations myapp
python manage.py migrate


#Each field in model should be an instance of the appropriate Field class
#Django uses the field class types to determine:
•The column type, which tells the database what kind of data to store (e.g. INTEGER, VARCHAR, TEXT).
•The default HTML widget to use when rendering a form field (e.g. <input type="text">, <select>).
•The minimal validation requirements, used in Django’s admin and in automatically-generated forms


###Models - Field options - All are optional

#null
If True, Django will store empty values as NULL in the database. Default is False.


#blank
If True, the field is allowed to be blank. Default is False.
Note that this is different than null. null is purely database-related, 
whereas blank is validation-related. 
If a field has blank=True, form validation will allow entry of an empty value. 
If a field has blank=False, the field will be required.


#choices
An iterable (e.g., a list or tuple) of 2-tuples to use as choices for this field. 
If this is given, the default form widget will be a select box 

The first element in each tuple is the value that will be stored in the database, 
the second element will be displayed by the default form widget or in a ModelChoiceField.

the display value for a choices field can be accessed using the get_FOO_display()
where FOO is field name  

#put below in models.py, and do makemigrations,migrate combinations
from django.db import models

class Person(models.Model):
    SHIRT_SIZES = (
        ('S', 'Small'),
        ('M', 'Medium'),
        ('L', 'Large'), )
    name = models.CharField(max_length=60)
    shirt_size = models.CharField(max_length=1, choices=SHIRT_SIZES)
    
from  modelex.models import Person   
p = Person(name="Fred Flintstone", shirt_size="L")
p.save()
>>> p.shirt_size
u'L'
>>> p.get_shirt_size_display()
u'Large'


#default
The default value for the field. This can be a value or a callable object. 
If callable it will be called every time a new object is created.

#help_text
Extra “help” text to be displayed with the form widget. 


#primary_key
If True, this field is the primary key for the model (instead of default auto generated id)
primary_key=True implies null=False and unique=True. 
Only one primary key is allowed on an object.
Workaround is
Use Model Meta option unique_together combining all columns 
and don't set any Primary_key (ie django would create one for you)

If you change the value of the primary key on an existing object and then save it, 
a new object will be created alongside the old one. 

#put below in models.py, and do makemigrations,migrate combinations
from django.db import models

class Fruit(models.Model):
	name = models.CharField(max_length=100, primary_key=True)
  
import modelex.models import Fruit  
fruit = Fruit.objects.create(name='Apple')
fruit.name = 'Pear'
fruit.save()
>>> Fruit.objects.values_list('name', flat=True)
['Apple', 'Pear']

#unique
If True, this field must be unique throughout the table.

#db_column
The name of the database column to use for this field. 
By default Django will use the field’s name.


#db_index
If True, a database index will be created for this field.


#db_tablespace
The name of the database tablespace to use for this field’s index, if this field is indexed. 
The default is the project’s DEFAULT_INDEX_TABLESPACE setting, if set, 
or the db_tablespace of the model 

#unique_for_date, unique_for_month, unique_for_year
For example, if you have a field 'title' that has unique_for_date="pub_date", 
then Django wouldn’t allow the entry of two records with the same title and pub_date.
Similarly for month or year (of pub_date) for unique_for_month, unique_for_year


#verbose_name
A human-readable name for the field. 
By default Django will automatically create it using the field’s attribute name, 
converting underscores to spaces. This can be first arg to Field ctor 
#For example  the verbose name is "person's first name":
first_name = models.CharField("person's first name", max_length=30)
#For example , the verbose name is "first name":
first_name = models.CharField(max_length=30)
#For ForeignKey, ManyToManyField and OneToOneField,
poll = models.ForeignKey(Poll, verbose_name="the related poll")
sites = models.ManyToManyField(Site, verbose_name="list of sites")
place = models.OneToOneField(Place, verbose_name="related place")
The convention is not to capitalize the first letter of the verbose_name. 
#Django will automatically capitalize the first letter where it needs to.


#editable
If False, the field will not be displayed in the admin or any other ModelForm. 
They are also skipped during model validation. Default is True.


#error_messages
The error_messages argument lets you override the default messages that the field will raise. 
Pass in a dictionary with below keys matching the error messages you want to override.
keys are null, blank, invalid, invalid_choice, unique, and unique_for_date. 


#validators
A list of validators to run for this field. 
A validator is a callable(def of class with __call__) that takes a value and raises a ValidationError if it doesn’t meet some criteria. 
#example validator that only allows even numbers:

from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _

def validate_even(value):
    if value % 2 != 0:
        raise ValidationError( _('%(value)s is not an even number'), params={'value': value}, )


class MyModel(models.Model):
    even_field = models.IntegerField(validators=[validate_even, ])
    
    
#then in form 
from django import forms
from modelex.models import validate_even

class MyForm(forms.Form):
    even_field = forms.IntegerField(validators=[validate_even])

#Built-in validators class  with __call__(), hence instance is callable
django.core.validators.RegexValidator([regex=None, message=None, code=None, inverse_match=None, flags=0])
regex – Can be a regular expression string or a pre-compiled regular expression

django.core.validators.EmailValidator(message=None, code=None, whitelist=None)

django.core.validators.URLValidator([schemes=None, regex=None, message=None, code=None])[source]
schemes : URL/URI scheme list to validate against. 
        If not provided, the default list is ['http', 'https', 'ftp', 'ftps'].
        As a reference, the IANA Web site provides a full list of valid URI schemes.

        
django.core.validators.MaxValueValidator(max_value)
Raises a ValidationError with a code of 'max_value' if value is greater than max_value.

django.core.validators.MinValueValidator(min_value)
Raises a ValidationError with a code of 'min_value' if value is less than min_value.

django.core.validators.MaxLengthValidator(max_length)
Raises a ValidationError with a code of 'max_length' if the length of value is greater than max_length.

django.core.validators.MinLengthValidator(min_length)
Raises a ValidationError with a code of 'min_length' if the length of value is less than min_length.

#Built-in validators instances 

django.core.validators.validate_email
An EmailValidator instance that ensures a value looks like an email address.

django.core.validators.validate_slug
A RegexValidator instance that ensures a value consists of only letters, numbers, underscores or hyphens.

django.core.validators.validate_ipv4_address
A RegexValidator instance that ensures a value looks like an IPv4 address.

django.core.validators.validate_ipv6_address
Uses django.utils.ipv6 to check the validity of an IPv6 address.

django.core.validators.validate_ipv46_address[source]
Uses both validate_ipv4_address and validate_ipv6_address to ensure a value is either a valid IPv4 or IPv6 address.

django.core.validators.validate_comma_separated_integer_list
A RegexValidator instance that ensures a value is a comma-separated list of integers.




###Models - Many-to-one relationships - use django.db.models.ForeignKey on 'Many' class 

#Example: Many-to-one relationships with querying 
#Many - Reporter, One - Article, use django.db.models.ForeignKey(Reporter) on Article 

#put below in models.py, and do makemigrations,migrate combinations
from __future__ import unicode_literals
from django.db import models
from django.utils.encoding import python_2_unicode_compatible

#all reporter objects ,  Reporter.objects. (this has add, get, filter, all methods)
#all articles of a reporter instance , r.article_set. (this has add, get, filter, all methods)
# instance of below class has create, save, delete methods and field names as attributes 
@python_2_unicode_compatible
class Reporter(models.Model):         #One Reporter has Many Article, Each Article belongs to One Reporter 
        first_name = models.CharField(max_length=30)
        last_name = models.CharField(max_length=30)
        email = models.EmailField()
        def __str__(self):              
            return "%s %s" % (self.first_name, self.last_name)

#All Article objects , Article.objects. (this has add, get, filter, all methods)
#Reporter instance of an article, a.reporter.  (this has add, get, filter, all methods)
#ForeignKey only on 'Many' side , 'One' side is automated by django
# instance of below class has create, save, delete methods and field names as attributes

@python_2_unicode_compatible
class Article(models.Model):
        headline = models.CharField(max_length=100)
        pub_date = models.DateField()
        reporter = models.ForeignKey(Reporter)
        def __str__(self):              
            return self.headline
        class Meta:
            ordering = ('headline',)

#Usages - create reporters 
$ python manage.py shell 

from modelex.models import *

r = Reporter(first_name='John', last_name='Smith', email='john@example.com')
r.save()

r2 = Reporter(first_name='Paul', last_name='Jones', email='paul@example.com')
r2.save()

Reporter.objects.all()  # from __str__(), [<Reporter: John Smith>, <Reporter: Paul Jones>]

#Usages - Create an Article linking a Reporter 

from datetime import date
a = Article(id=None, headline="This is a test", pub_date=date(2005, 7, 27), reporter=r)
a.save()

a.reporter.id  #1
a.reporter      #<Reporter: John Smith>

#Article objects have access to their related Reporter objects 
r = a.reporter
r.first_name, r.last_name   #('John', 'Smith')

#Usages - Create an Article via the Reporter object
#Each Reporter(Many side) automatically exposes <<One_side_name>>_set ie article_set
#which has create() method to create a 'One' side of Many to One relation
#has add() method to add a One , all() to list, check via dir(r.article_list)

new_article = r.article_set.create(headline="John's second story", pub_date=date(2005, 7, 29))
>>> new_article
<Article: John's second story>
>>> new_article.reporter
<Reporter: John Smith>
>>> new_article.reporter.id
1

#Usages - Create a new article, and add it to the article set::

new_article2 = Article(headline="Paul's story", pub_date=date(2006, 1, 17))
r.article_set.add(new_article2)
new_article2.reporter           #<Reporter: John Smith>
new_article2.reporter.id        #1
r.article_set.all()             #[<Article: John's second story>, <Article: Paul's story>, <Article: This is a test>]

#Add the same article to a different article set - check that it moves::

r2.article_set.add(new_article2)
new_article2.reporter.id        #2
new_article2.reporter           #<Reporter: Paul Jones>

#Adding an object of the wrong type raises TypeError::

r.article_set.add(r2)       #TypeError: 'Article' instance expected

r.article_set.all()         #[<Article: John's second story>, <Article: This is a test>]
r2.article_set.all()        #[<Article: Paul's story>]

r.article_set.count()       #2
r2.article_set.count()      #1


#Relations support field lookups as well.
#Use double underscores to separate relationship from search criteria 

#ariticle_set is set of Article, hence each field of Article can be checked 
r.article_set.filter(headline__startswith='This') #[<Article: This is a test>]

# Find all Articles for any Reporter whose first name is "John"
#Exact match is implied here
Article.objects.filter(reporter__first_name='John') #[<Article: John's second story>, <Article: This is a test>]

#Query twice over the related field. 
#This translates to an AND condition in the WHERE clause::

Article.objects.filter(reporter__first_name='John', reporter__last_name='Smith') #[<Article: John's second story>, <Article: This is a test>]

#For the related lookup , can supply a primary key value 
#or pass the related object explicitly::

Article.objects.filter(reporter__pk=1) #[<Article: John's second story>, <Article: This is a test>]
Article.objects.filter(reporter=1)     #[<Article: John's second story>, <Article: This is a test>]
Article.objects.filter(reporter=r)     #[<Article: John's second story>, <Article: This is a test>]

Article.objects.filter(reporter__in=[1,2]).distinct()  #[<Article: John's second story>, <Article: Paul's story>, <Article: This is a test>]
Article.objects.filter(reporter__in=[r,r2]).distinct() #[<Article: John's second story>, <Article: Paul's story>, <Article: This is a test>]

#use a queryset instead of a literal list of instances::

Article.objects.filter(reporter__in=Reporter.objects.filter(first_name='John')).distinct()

#Querying in the opposite direction
#'Many' side has 'One' Side even though it is not explicit in model 
#find all articles of a reporter 
r.article_set.all()                     #[<Article: This is a test>]

#operate on all reporters 
Reporter.objects.filter(article__pk=1)  #[<Reporter: John Smith>]
Reporter.objects.filter(article=1)      #[<Reporter: John Smith>]
Reporter.objects.filter(article=a)      #[<Reporter: John Smith>]

#nested filter criteria 
Reporter.objects.filter(article__headline__startswith='This')
Reporter.objects.filter(article__headline__startswith='This').distinct()

#Counting in the opposite direction works in conjunction with distinct()::
Reporter.objects.filter(article__headline__startswith='This').count()  #3
Reporter.objects.filter(article__headline__startswith='This').distinct().count() #1

#Queries can go round in circles::
Reporter.objects.filter(article__reporter__first_name__startswith='John')  #[<Reporter: John Smith>, <Reporter: John Smith>, <Reporter: John Smith>, <Reporter: John Smith>]
Reporter.objects.filter(article__reporter__first_name__startswith='John').distinct()#[<Reporter: John Smith>]
Reporter.objects.filter(article__reporter=r).distinct() #[<Reporter: John Smith>]

#If you delete a reporter, his articles will be deleted 
#(assuming that the ForeignKey was defined with :attr:`django.db.models.ForeignKey.on_delete` set to
#'CASCADE', which is the default)::

Article.objects.all()                       #[<Article: John's second story>, <Article: Paul's story>, <Article: This is a test>]
Reporter.objects.order_by('first_name')     #[<Reporter: John Smith>, <Reporter: Paul Jones>]  
r2.delete()
Article.objects.all()                       #[<Article: John's second story>, <Article: This is a test>]
Reporter.objects.order_by('first_name')     #[<Reporter: John Smith>]

#You can delete using a JOIN in the query::
Reporter.objects.filter(article__headline__startswith='This').delete()
Reporter.objects.all()  #[]
Article.objects.all()   #[]

##Many-to-one details 
To create a recursive relationship – 
an object that has a many-to-one relationship with itself – use models.ForeignKey('self').

you can use alos use string name of Model eg 
manufacturer = models.ForeignKey('Manufacturer')
manufacturer = models.ForeignKey('production.Manufacturer') #in another app 'production'

Disable index creation by setting db_index to False. 

Django appends "_id" to the field name to create its database column name. 

#Arguments ForeignKey.limit_choices_to
Sets a limit to the available choices for this field when this field is rendered 
using a ModelForm or the admin (by default, all objects in the queryset are available to choose). 
Either a dictionary, a Q object, or a callable returning a dictionary or Q object can be used.
# to list only User that have is_staff=True. 
staff_member = models.ForeignKey(User, limit_choices_to={'is_staff': True})
#or callable 
def limit_pub_date_choices():
    return {'pub_date__lte': datetime.date.utcnow()}

limit_choices_to = limit_pub_date_choices

#Arguments  ForeignKey.related_name
The name to use for the relation from the related object back to this one
blog = ForeignKey(Blog, on_delete=models.CASCADE, related_name='entries')
b = Blog.objects.get(id=1)
b.entries.all()     # Other side of ForeignKey , by default it is entry_set, now it is entries

#Arguments  ForeignKey.related_query_name
The name to use for the reverse filter name from the target model. 
Defaults to the value of related_name 

# Declare the ForeignKey with related_query_name
class Tag(models.Model):
    article = models.ForeignKey(Article, related_name="tags", related_query_name="tag")
    name = models.CharField(max_length=255)

# That's now the name of the reverse filter
Article.objects.filter(tag__name="important")


##Arguments  ForeignKey.to_field
The field on the related object that the relation is to. 
By default, Django uses the primary key of the related object.


#Arguments  ForeignKey.db_constraint
Controls whether or not a constraint should be created in the database for this foreign key. 
The default is True

#Arguments  ForeignKey.on_delete
When an object referenced by a ForeignKey is deleted, 
Django by default emulates the behavior of the SQL constraint ON DELETE CASCADE 
and also deletes the object containing the ForeignKey. 
user = models.ForeignKey(User, blank=True, null=True, on_delete=models.SET_NULL)
    • CASCADE
    Cascade deletes; the default.
    • PROTECT
    Prevent deletion of the referenced object by raising ProtectedError, a subclass of django.db.IntegrityError.
    • SET_NULL
    Set the ForeignKey null; this is only possible if null is True.
    • SET_DEFAULT
    Set the ForeignKey to its default value; a default for the ForeignKey must be set.
    • DO_NOTHING
    Take no action.
    • SET()
    Set the ForeignKey to the value passed to SET(), or if a callable is passed in, the result of calling it. 
 
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import models

def get_sentinel_user():
    return get_user_model().objects.get_or_create(username='deleted')[0]

class MyModel(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.SET(get_sentinel_user))





###Models - Many-to-many relationships- use ManyToManyField on either of side (not both)
#Generally, ManyToManyField instances should go in the object that’s going to be edited on a form


#Example: Many-to-many relationships
#an Article can be published in multiple Publication objects, 
#and a Publication has multiple Article objects:


from __future__ import unicode_literals
from django.db import models
from django.utils.encoding import python_2_unicode_compatible

#all publication objects ,  Publication.objects.  (this has add,  get, filter, all methods)
#all articles of a Publication instance , p.article_set.  (this has add, get, filter, all methods)
#Note article_set is created by Django, nothing is explicitly mentioned
#instance of below class has create, save, delete methods and field names as attributes
@python_2_unicode_compatible
class Publication(models.Model):
        title = models.CharField(max_length=30)

        def __str__(self):              # __unicode__ on Python 2
            return self.title

        class Meta:
            ordering = ('title',)
            
            
#All Article objects , Article.objects.  (this has add,  get, filter, all methods)
#publications of an article, a.publications. (from field name) (this has add, get, filter, all methods)
#ManyToManyField only on one side , other side is automated by django
# instance of below class has create, save, delete methods and field names as attributes

@python_2_unicode_compatible
class Article(models.Model):
        headline = models.CharField(max_length=100)
        publications = models.ManyToManyField(Publication)

        def __str__(self):              # __unicode__ on Python 2
            return self.headline

        class Meta:
            ordering = ('headline',)


#Create a couple of Publications
p1 = Publication(title='The Python Journal')
p1.save()
p2 = Publication(title='Science News')
p2.save()
p3 = Publication(title='Science Weekly')
p3.save()

#Create an Article
a1 = Article(headline='Django lets you build Web apps easily')

#You can't associate it with a Publication until it's been saved::

a1.publications.add(p1)    #ValueError: 'Article' instance needs to have a primary key value before a many-to-many relationship can be used.

a1.save()
a1.publications.add(p1)

#Create another Article, and set it to appear in both Publications::
a2 = Article(headline='NASA uses Python')
a2.save()
a2.publications.add(p1, p2)
a2.publications.add(p3)

#Adding a second time is OK::
a2.publications.add(p3)

#Adding an object of the wrong type raises :exc:`TypeError`::
a2.publications.add(a1)     #TypeError: 'Publication' instance expected

#Create and add a Publication to an Article in one step 
new_publication = a2.publications.create(title='Highlights for Children')

#Article objects have access to their related Publication objects::
a1.publications.all()   #[<Publication: The Python Journal>]

#Publication objects have access to their related Article objects::
p2.article_set.all()   #[<Article: NASA uses Python>]
Publication.objects.get(id=4).article_set.all() #[<Article: NASA uses Python>]

#Many-to-many relationships can be queried using __

Article.objects.filter(publications__id=1)  #[<Article: Django lets you build Web apps easily>, <Article: NASA uses Python>]
Article.objects.filter(publications__pk=1)  #[<Article: Django lets you build Web apps easily>, <Article: NASA uses Python>]
Article.objects.filter(publications=1)      #[<Article: Django lets you build Web apps easily>, <Article: NASA uses Python>]
Article.objects.filter(publications=p1)     #[<Article: Django lets you build Web apps easily>, <Article: NASA uses Python>]

Article.objects.filter(publications__title__startswith="Science") #[<Article: NASA uses Python>, <Article: NASA uses Python>]
Article.objects.filter(publications__title__startswith="Science").distinct() #[<Article: NASA uses Python>]

Article.objects.filter(publications__title__startswith="Science").count() #2
Article.objects.filter(publications__title__startswith="Science").distinct().count() #1

Article.objects.filter(publications__in=[1,2]).distinct()   #[<Article: Django lets you build Web apps easily>, <Article: NASA uses Python>]
Article.objects.filter(publications__in=[p1,p2]).distinct() #[<Article: Django lets you build Web apps easily>, <Article: NASA uses Python>]

#Reverse m2m queries are supported 
Publication.objects.filter(id=1)  #[<Publication: The Python Journal>]
Publication.objects.filter(pk=1)  #[<Publication: The Python Journal>]

Publication.objects.filter(article__headline__startswith="NASA") #[<Publication: Highlights for Children>, <Publication: Science News>, <Publication: Science Weekly>, <Publication: The Python Journal>]

Publication.objects.filter(article__id=1) #[<Publication: The Python Journal>]
Publication.objects.filter(article__pk=1) #[<Publication: The Python Journal>]
Publication.objects.filter(article=1)     #[<Publication: The Python Journal>]
Publication.objects.filter(article=a1)    #[<Publication: The Python Journal>]

Publication.objects.filter(article__in=[1,2]).distinct() #[<Publication: Highlights for Children>, <Publication: Science News>, <Publication: Science Weekly>, <Publication: The Python Journal>]
Publication.objects.filter(article__in=[a1,a2]).distinct() #[<Publication: Highlights for Children>, <Publication: Science News>, <Publication: Science Weekly>, <Publication: The Python Journal>]

#Excluding a related item 

Article.objects.exclude(publications=p2)  #[<Article: Django lets you build Web apps easily>]

#If we delete a Publication, its Articles won't be able to access it::

p1.delete()
Publication.objects.all()  #[<Publication: Highlights for Children>, <Publication: Science News>, <Publication: Science Weekly>]
a1 = Article.objects.get(pk=1) 
a1.publications.all()   #[]

#If we delete an Article, its Publications won't be able to access it::

a2.delete()
Article.objects.all() #[<Article: Django lets you build Web apps easily>]
p2.article_set.all()  #[]


#Adding via the 'other' end of an m2m::

a4 = Article(headline='NASA finds intelligent life on Earth')
a4.save()
p2.article_set.add(a4)
p2.article_set.all()  #[<Article: NASA finds intelligent life on Earth>]
a4.publications.all() #[<Publication: Science News>]

#Adding via the other end using keywords

new_article = p2.article_set.create(headline='Oxygen-free diet works wonders')
p2.article_set.all()  #[<Article: NASA finds intelligent life on Earth>, <Article: Oxygen-free diet works wonders>]
a5 = p2.article_set.all()[1]
a5.publications.all() #[<Publication: Science News>]

#Removing Publication from an Article - delinking , not hard deleting 

a4.publications.remove(p2)  #
p2.article_set.all()        #[<Article: Oxygen-free diet works wonders>]
a4.publications.all()       #[]

#And from the other end - delinking , not hard deleting 
p2.article_set.remove(a5)
p2.article_set.all()    #[]
a5.publications.all()   #[]

#Relation sets can be assigned. Assignment clears any existing set members::

a4.publications.all()   #[<Publication: Science News>]
a4.publications = [p3]  
a4.publications.all()   #[<Publication: Science Weekly>]

#Relation sets can be cleared - delinking , not hard deleting 

p2.article_set.clear()
p2.article_set.all()    #[]

# can clear from the other end- delinking , not hard deleting 
p2.article_set.add(a4, a5)
p2.article_set.all()    #[<Article: NASA finds intelligent life on Earth>, <Article: Oxygen-free diet works wonders>]
a4.publications.all()   #[<Publication: Science News>, <Publication: Science Weekly>]
a4.publications.clear()
a4.publications.all()   #[]
p2.article_set.all()    #[<Article: Oxygen-free diet works wonders>]

#Bulk add - Recreate the Article and Publication we have deleted::

p1 = Publication(title='The Python Journal')
p1.save()
a2 = Article(headline='NASA uses Python')
a2.save()
a2.publications.add(p1, p2, p3)

#Bulk delete some Publications - references to deleted publications should go::

Publication.objects.filter(title__startswith='Science').delete()
Publication.objects.all()   #[<Publication: Highlights for Children>, <Publication: The Python Journal>]
Article.objects.all()       #[<Article: Django lets you build Web apps easily>, <Article: NASA finds intelligent life on Earth>, <Article: NASA uses Python>, <Article: Oxygen-free diet works wonders>]
a2.publications.all()       #[<Publication: The Python Journal>]

#Bulk delete some articles - references to deleted objects should go::

q = Article.objects.filter(headline__startswith='Django')
print(q)    #[<Article: Django lets you build Web apps easily>]
q.delete()
print(q)    #[]
p1.article_set.all()    #[<Article: NASA uses Python>]

#An alternate to calling clear(),  is to assign the empty set::

p1.article_set = []
p1.article_set.all()    #[]

a2.publications = [p1, new_publication]
a2.publications.all()   #[<Publication: Highlights for Children>, <Publication: The Python Journal>]
a2.publications = []
a2.publications.all()   #[]



## Many-To_Many relation - Using intermediate model between both sides - use 'through'

from __future__ import unicode_literals
from django.db import models
from django.utils.encoding import python_2_unicode_compatible

@python_2_unicode_compatible
class Person(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):              # __unicode__ on Python 2
        return self.name

@python_2_unicode_compatible
class Group(models.Model):
    name = models.CharField(max_length=128)
    members = models.ManyToManyField(Person, through='Membership')  #Membership is another Model 

    def __str__(self):              # __unicode__ on Python 2
        return self.name

@python_2_unicode_compatible
class Membership(models.Model):
    person = models.ForeignKey(Person)    #Create link , must be only one ForeignKey using Many side class, else use ManyToManyField.through_fields
    group = models.ForeignKey(Group)      #Create link must be only one ForeignKey using Many side class, else ManyToManyField.through_fields
    date_joined = models.DateField()
    invite_reason = models.CharField(max_length=64)

#Usage - save intermediate model explicitly 
ringo = Person.objects.create(name="Ringo Starr")
paul = Person.objects.create(name="Paul McCartney")
beatles = Group.objects.create(name="The Beatles")
m1 = Membership(person=ringo, group=beatles, 
        date_joined=date(1962, 8, 16), invite_reason="Needed a new drummer.")
m1.save()
beatles.members.all()   #[<Person: Ringo Starr>]
ringo.group_set.all()   #[<Group: The Beatles>]
m2 = Membership.objects.create(person=paul, group=beatles,
        date_joined=date(1960, 8, 1), invite_reason="Wanted to form a band.")
beatles.members.all()   #[<Person: Ringo Starr>, <Person: Paul McCartney>]


#Must not use add, create, or assignment (i.e., beatles.members = [...]) to create relationships:

# THIS WILL NOT WORK
beatles.members.add(john)
# NEITHER WILL THIS
beatles.members.create(name="George Harrison")
# AND NEITHER WILL THIS
beatles.members = [john, paul, ringo, george]

#The remove() method is disabled for similar reasons. 
#the clear() method can be used to remove all many-to-many relationships for an instance:
beatles.members.clear()
# Note that this deletes the intermediate model instances
Membership.objects.all()  #[]

#Queries are used as normal many-to-many fields ignoring intermediate model
# Find all the groups with a member whose name starts with 'Paul'
Group.objects.filter(members__name__startswith='Paul') #[<Group: The Beatles>]


# query on intermediate model attributes
# Find all the members of the Beatles that joined after 1 Jan 1961
Person.objects.filter(
        group__name='The Beatles',
        membership__date_joined__gt=date(1961,1,1)) #[<Person: Ringo Starr]


#directly query the Membership model
ringos_membership = Membership.objects.get(group=beatles, person=ringo)
ringos_membership.date_joined   #datetime.date(1962, 8, 16)
ringos_membership.invite_reason #u'Needed a new drummer.'

#or query the many-to-many reverse relationship from a Person object:
ringos_membership = ringo.membership_set.get(group=beatles)
ringos_membership.date_joined   #datetime.date(1962, 8, 16)
ringos_membership.invite_reason #u'Needed a new drummer.'


##Many-to-Many details 
Django creates an intermediary join table to represent the many-to-many relationship. 
By default, this table name is generated using the name of the many-to-many field 
and the name of the table for the model that contains it. 

##Argument ManyToManyField.related_name
Same as ForeignKey.related_name.

##Argument ManyToManyField.related_query_name
Same as ForeignKey.related_query_name.

##Argument ManyToManyField.limit_choices_to
Same as ForeignKey.limit_choices_to.


##Argument ManyToManyField.symmetrical
Only used in the definition of ManyToManyFields on self. 

from django.db import models

class Person(models.Model):
    friends = models.ManyToManyField("self")


By default, Django does not add a 'person_set' attribute to the Person class for 'self' type 
set symmetrical to False to get above attribute 

#Argument ManyToManyField.db_table
The name of the table to create for storing the many-to-many data.


##Argument ManyToManyField.through_fields
Only used when a custom intermediary model is specified
and intermidiate table has many ForeignKey on same class
Defines which ForeignKey is to be used by Django

from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=50)

class Group(models.Model):
    name = models.CharField(max_length=128)
    members = models.ManyToManyField(Person, through='Membership', 
            through_fields=('group', 'person'))

class Membership(models.Model):
    group = models.ForeignKey(Group)
    person = models.ForeignKey(Person)
    inviter = models.ForeignKey(Person, related_name="membership_invites")
    invite_reason = models.CharField(max_length=64)









###Models -  One-to-one relationships, use OneToOneField either one side 

from __future__ import unicode_literals
from django.db import models
from django.utils.encoding import python_2_unicode_compatible

# .restaurant attribute is automatically created 
@python_2_unicode_compatible
class Place(models.Model):
        name = models.CharField(max_length=50)
        address = models.CharField(max_length=80)

        def __str__(self):              # __unicode__ on Python 2
            return "%s the place" % self.name

@python_2_unicode_compatible
class Restaurant(models.Model):
        place = models.OneToOneField(Place, primary_key=True)
        serves_hot_dogs = models.BooleanField(default=False)
        serves_pizza = models.BooleanField(default=False)

        def __str__(self):              # __unicode__ on Python 2
            return "%s the restaurant" % self.place.name

@python_2_unicode_compatible
class Waiter(models.Model):
        restaurant = models.ForeignKey(Restaurant)
        name = models.CharField(max_length=50)

        def __str__(self):              # __unicode__ on Python 2
            return "%s the waiter at %s" % (self.name, self.restaurant)


#Create a couple of Places::
p1 = Place(name='Demon Dogs', address='944 W. Fullerton')
p1.save()
p2 = Place(name='Ace Hardware', address='1013 N. Ashland')
p2.save()

#Create a Restaurant. Pass the ID of the "parent" object as this object's ID::
r = Restaurant(place=p1, serves_hot_dogs=True, serves_pizza=False)
r.save()

#A Restaurant can access its place::
r.place         #<Place: Demon Dogs the place>

#A Place can access its restaurant, if available::
p1.restaurant   #<Restaurant: Demon Dogs the restaurant>

#p2 doesn't have an associated restaurant::
from django.core.exceptions import ObjectDoesNotExist
try:
    p2.restaurant
except ObjectDoesNotExist:
    print("There is no restaurant here.")


#or use hasattr to avoid the need for exception catching::
hasattr(p2, 'restaurant')   #False

#Set the place using assignment .
#Because place is the primary key on Restaurant, the save will create a new restaurant::
r.place = p2
r.save()
p2.restaurant   #<Restaurant: Ace Hardware the restaurant>
r.place         #<Place: Ace Hardware the place>

#Set the place back again, using assignment in the reverse direction::
p1.restaurant = r
p1.restaurant       #<Restaurant: Demon Dogs the restaurant>

Restaurant.objects.all()    #returns the Restaurants, not the Places
Place.objects.all()   #returns all Places, regardless of whether they have Restaurants::

Place.objects.order_by('name')      #[<Place: Ace Hardware the place>, <Place: Demon Dogs the place>]

#query 
Restaurant.objects.get(place=p1)    #<Restaurant: Demon Dogs the restaurant>
Restaurant.objects.get(place__pk=1) #<Restaurant: Demon Dogs the restaurant>
Restaurant.objects.filter(place__name__startswith="Demon")  #[<Restaurant: Demon Dogs the restaurant>]
Restaurant.objects.exclude(place__address__contains="Ashland")  #[<Restaurant: Demon Dogs the restaurant>]

#This  works in reverse::
Place.objects.get(pk=1)     #<Place: Demon Dogs the place>
Place.objects.get(restaurant__place=p1) #<Place: Demon Dogs the place>
Place.objects.get(restaurant=r)         #<Place: Demon Dogs the place>
Place.objects.get(restaurant__place__name__startswith="Demon")  #<Place: Demon Dogs the place>

#Add a Waiter to the Restaurant::
w = r.waiter_set.create(name='Joe')
w.save()
w   #<Waiter: Joe the waiter at Demon Dogs the restaurant>

#Query the waiters::
Waiter.objects.filter(restaurant__place=p1) #[<Waiter: Joe the waiter at Demon Dogs the restaurant>]
Waiter.objects.filter(restaurant__place__name__startswith="Demon")  #[<Waiter: Joe the waiter at Demon Dogs the restaurant>]



###Models  - models across files
#It’s OK to relate a model to one from another app. 

from django.db import models
from geography.models import ZipCode

class Restaurant(models.Model):
    # ...
    zip_code = models.ForeignKey(ZipCode)



###Models - Field name restrictions
1. A field name cannot be a Python reserved word
class Example(models.Model):
    pass = models.IntegerField() # 'pass' is a reserved word!

2. A field name cannot contain more than one underscore in a row
class Example(models.Model):
    foo__bar = models.IntegerField() # 'foo__bar' has two underscores!
    
#SQL reserved words, such as join, where or select, are allowed as model field names, 
#because Django escapes all database table names and column names in every underlying SQL query. It uses the quoting syntax of your particular database engine.




###Models -  Meta options - any options applicable for whole table 

from django.db import models

class Ox(models.Model):
    horn_length = models.IntegerField()

    class Meta:
        ordering = ["horn_length"]
        verbose_name_plural = "oxen"


## Available Meta options
#abstract
If abstract = True, this model will be an abstract base class.

#app_label
If a model exists outside of the standard locations (models.py or a models package in an app), 
the model must define which app it is part of

#db_table
The name of the database table to use for the model: db_table = 'music_album'
Use lowercase table names for MySQL

#db_tablespace
The name of the database tablespace to use for this model. 
The default is the project’s DEFAULT_TABLESPACE setting, if set. 

#get_latest_by
The name of an orderable field in the model, a DateField, DateTimeField, or IntegerField. 
This specifies the default field to use in  Manager’s latest() and earliest() methods.
get_latest_by = "order_date"

#managed
Defaults to True, meaning Django will create the appropriate database tables in migrate 
or as part of migrations and remove them as part of a flush management command. 
Use it when Model is used with existing DB table, no other impacts to Querying 

#order_with_respect_to
Marks this object as “orderable” with respect to the given field. 
order_with_respect_to adds an additional field/database column named _order
hence use createmigrations/migrate combination

#Example 
from django.db import models

#.answer_set. is created automatically 
class Question(models.Model):           #this is Many side 
    text = models.TextField()
    # ...

class Answer(models.Model):                #this is One side 
    question = models.ForeignKey(Question) #Takes Many side 
    # ...

    class Meta:
        order_with_respect_to = 'question'

#When order_with_respect_to is set, two additional methods are provided to retrieve 
#and to set the order of the related objects: 
#get_RELATED_order() and set_RELATED_order(), where RELATED is the lower cased model name. 

question = Question.objects.get(id=1)
question.get_answer_order()  #[1, 2, 3]
question.set_answer_order([3, 1, 2])


#The One side  has two methods, get_next_in_order() and get_previous_in_order(), 
#which can be used to access those objects in their proper order. Assuming the Answer objects are ordered by id:
answer = Answer.objects.get(id=2)
answer.get_next_in_order()      #<Answer: 3>
answer.get_previous_in_order()  #<Answer: 1>


#ordering
The default ordering for the object, for use when obtaining lists of objects
Ordering is not a free operation
ordering = ['-order_date']  #- means descending order
ordering = ['pub_date']     #ascending order 
#To order by pub_date descending, then by author ascending, use this:
ordering = ['-pub_date', 'author']


#permissions
Extra permissions to enter into the permissions table when creating this object. 
Add, delete and change permissions are automatically created for each model. 
#This example specifies an extra permission, can_deliver_pizzas:
permissions = (("can_deliver_pizzas", "Can deliver pizzas"),) #(permission_code, human_readable_permission_name)


#default_permissions
Defaults to ('add', 'change', 'delete'). 
You may customize this list, for example, by setting this to an empty list 

#proxy
If proxy = True, a model which subclasses another model will be treated as a proxy model.


#select_on_save
The default is False.
Determines if Django will use the pre-1.6 django.db.models.Model.save() algorithm. 


#unique_together
Sets of field names that, taken together, must be unique
is enforced at the database level
ManyToManyField cannot be included in unique_together. 
unique_together = (("driver", "restaurant"),)
unique_together = ("driver", "restaurant")  #if only single set


#index_together
Sets of field names that, taken together, are indexed:
index_together = [
    ["pub_date", "deadline"],
]
index_together = ["pub_date", "deadline"] #if only single set 

#verbose_name
A human-readable name for the object, singular:
verbose_name = "pizza"
By default Django will use - 'CamelCase becomes camel case'

#verbose_name_plural
The plural name for the object:
verbose_name_plural = "stories"
If this isn’t given, Django will use verbose_name + "s".



###Models - Field types

#AutoField
An IntegerField that automatically increments according to available IDs. 


#BigIntegerField
A 64 bit integer,The default form widget for this field is a TextInput.


#BinaryField
A field to store raw binary data. It only supports bytes assignment
Dont store file here 

#BooleanField
A true/false field.
The default form widget for this field is a CheckboxInput.
If you need to accept null values then use NullBooleanField instead.
The default value of BooleanField is None when Field.default isn’t defined.


#CharField(max_length)
A string field, for small- to large-sized strings.
For large amounts of text, use TextField.
The default form widget for this field is a TextInput.
    #CharField.max_length
    The maximum length (in characters) of the field. 
    The max_length is enforced at the database level and in Django’s validation.




#CommaSeparatedIntegerField(max_length=None[, **options])
A field of integers separated by commas. 


#DateField([auto_now=False, auto_now_add=False, **options])
A date, represented in Python by a datetime.date instance. 
The default form widget for this field is a TextInput. 
The options auto_now_add, auto_now, and default are mutually exclusive.
    #DateField.auto_now
    Automatically set the field to now every time the object is saved. 
    #DateField.auto_now_add
    Automatically set the field to now when the object is first created.
 

#DateTimeField([auto_now=False, auto_now_add=False, **options])
A date and time, represented in Python by a datetime.datetime instance. 
Similar to DateField

#DecimalField(max_digits=None, decimal_places=None[, **options])
A fixed-precision decimal number, represented in Python by a Decimal instance. 
The default form widget for this field is a TextInput.
models.DecimalField(..., max_digits=5, decimal_places=2)


#DurationField
A field for storing periods of time - modeled in Python by timedelta. 
Arithmetic, comparing  with DurationField works other than PostgreSQL, 

#EmailField([max_length=254, **options])
A CharField that checks that the value is a valid email address. 
It uses EmailValidator to validate the input.

#FileField([upload_to=None, max_length=100, **options])
A file-upload field.
FileField instances are created in your database as varchar columns with a default max length of 100 characters
File uploading depends on two settings.py , configure both in apache like static files 
MEDIA_ROOT eg "/var/www/example.com/media/,
MEDIA_URL eg  "/media/" 
FileField.upload_to is the subdir of MEDIA_ROOT where files are uploaded 

class MyModel(models.Model):
    # file will be uploaded to MEDIA_ROOT/uploads
    upload = models.FileField(upload_to='uploads/')
    # or can  contain strftime() formatting
    # file will be saved to MEDIA_ROOT/uploads/2015/01/30 
    upload = models.FileField(upload_to='uploads/%Y/%m/%d/')
    
#FileField.upload_to can be callable 
#instance = Model instance where FileField field is present 
#filename = upload file name 
def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return 'user_{0}/{1}'.format(instance.user.id, filename)

class MyModel(models.Model):
    upload = models.FileField(upload_to=user_directory_path)

#Example - accessing ImageField or FileField 
from django.db import models

class Car(models.Model):
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=5, decimal_places=2)
    photo = models.ImageField(upload_to='cars')

#car instance 
car = Car.objects.get(name="57 Chevy")
car.photo       #<ImageFieldFile: chevy.jpg>,  FieldFile or ImageFieldFile object , has name, size, url, open(mode), close(), delete() attributes
car.photo.name  #'cars/chevy.jpg'
car.photo.path  #'/media/cars/chevy.jpg'
car.photo.url   #'http://media.example.com/cars/chevy.jpg'

#Saving to different place 
import os
from django.conf import settings
initial_path = car.photo.path
car.photo.name = 'cars/chevy_ii.jpg'
new_path = settings.MEDIA_ROOT + car.photo.name
# Move the file on the filesystem
os.rename(initial_path, new_path)
car.save()
car.photo.path #'/media/cars/chevy_ii.jpg'
car.photo.path == new_path #True



#FilePathField(path=None[, match=None, recursive=False, max_length=100, **options])
A CharField whose choices are limited to the filenames(allow_files=True) or dirs (allow_folders=True)
matched with 'match' regex in director 'path'. Might be recursive 
FilePathField(path="/home/images", match="foo.*", recursive=True)

#FloatField
A floating-point number represented in Python by a float instance.
The default form widget for this field is a TextInput.


#ImageField([upload_to=None, height_field=None, width_field=None, max_length=100, **options])
Inherits all attributes and methods from FileField(+ height_field, width_field attributes)
but validates that the uploaded object is a valid image.
Requires the Pillow library.
The default form widget for this field is a ClearableFileInput.


#IntegerField
An integer.
The default form widget for this field is a TextInput.


#IPAddressField
An IP address, in string format (e.g. “192.0.2.30”). 
The default form widget for this field is a TextInput.


#GenericIPAddressField([protocol=both, unpack_ipv4=False, **options])
An IPv4 or IPv6 address, in string format (e.g. 192.0.2.30 or 2a02:42fe::4). 
The default form widget for this field is a TextInput.

#NullBooleanField
Like a BooleanField, but allows NULL as one of the options. 
Use this instead of a BooleanField with null=True. 
The default form widget for this field is a NullBooleanSelect.


#PositiveIntegerField
Like an IntegerField, but must be either positive or zero (0). 


#PositiveSmallIntegerField
Like a PositiveIntegerField, but only allows values under a certain (database-dependent) point. 


#SlugField([max_length=50, **options])
Slug is a newspaper term. 
A slug is a short label for something, containing only letters, numbers, underscores or hyphens. 
They’re generally used in URLs.


#SmallIntegerField
Like an IntegerField, but only allows values under a certain (database-dependent) point.


#TextField
A large text field. 
The default form widget for this field is a Textarea.


#TimeField([auto_now=False, auto_now_add=False, **options])
A time, represented in Python by a datetime.time instance. 
Accepts the same auto-population options as DateField.
The default form widget for this field is a TextInput. 


#URLField([max_length=200, **options])
A CharField for a URL.
The default form widget for this field is a TextInput.


#UUIDField
A field for storing universally unique identifiers. 
Uses Python’s UUID class. 



### Models - django.db.models.Model 

#attribute  Model.objects
django.db.models.manager.Manager instance, used for accessing DB 
Managers are only accessible via model classes, not the model instances.

#Model methods
Model methods are used for custom “row-level” functionality of DB table  
Manager methods are used for custom “table-wide” functionality

#Example 
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    birth_date = models.DateField()

    def baby_boomer_status(self):		# Model Methods, self is a row ie instance of Person
        "Returns the person's baby-boomer status."
        import datetime
        if self.birth_date < datetime.date(1945, 8, 1):
            return "Pre-boomer"
        elif self.birth_date < datetime.date(1965, 1, 1):
            return "Baby boomer"
        else:
            return "Post-boomer"

    def _get_full_name(self):
        "Returns the person's full name."
        return '%s %s' % (self.first_name, self.last_name)
    full_name = property(_get_full_name)


##Model methods that must be overridden in subclass
    
__str__() (Python 3)        Python 3 equivalent of __unicode__().
__unicode__() (Python 2)    Stringify a Model name
get_absolute_url()          This tells Django how to calculate the URL for an object. 
                            Any object that has a URL that uniquely identifies it should define this method


##Overriding predefined model methods eg create, save etc 
#Overridden model methods are not called on bulk operations

#Example - custom save 

from django.db import models

class Blog(models.Model):
    name = models.CharField(max_length=100)
    tagline = models.TextField()

    def save(self, *args, **kwargs):
        do_something()
        super(Blog, self).save(*args, **kwargs) # Call the "real" save() method.
        do_something_else()


#Example -  prevent saving:

from django.db import models

class Blog(models.Model):
    name = models.CharField(max_length=100)
    tagline = models.TextField()

    def save(self, *args, **kwargs):
        if self.name == "Yoko Ono's blog":
            return # Yoko shall never have her own blog!
        else:
            super(Blog, self).save(*args, **kwargs) # Call the "real" save() method.


##Model inheritance-creating abstract Model 
#Abstract class does not get DB table 

from django.db import models

class CommonInfo(models.Model):
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()

    class Meta:
        abstract = True

class Student(CommonInfo):
    home_group = models.CharField(max_length=5)

#If a child class does not declare its own Meta class, 
#it will inherit the parent’s Meta but makes abstract = False 

#Or use explicit Meta subclass 

from django.db import models

class CommonInfo(models.Model):
    # ...
    class Meta:
        abstract = True
        ordering = ['name']

class Student(CommonInfo):
    # ...
    class Meta(CommonInfo.Meta):
        db_table = 'student_info'    #inherits ordering


#Be careful with ForeignKey.related_name or ManyToManyField.related_name 
#when used with Abstract Class as all subclass derives the same related_name , resulting conflicts 

#Solution is to use "%(app_label)s_%(class)s" in related_name
•'%(class)s' is replaced by the lower-cased name of the child class that the field is used in.
•'%(app_label)s' is replaced by the lower-cased name of the app the child class is contained within. 

#Example 
from django.db import models

class Base(models.Model):
    m2m = models.ManyToManyField(OtherModel, related_name="%(app_label)s_%(class)s_related")

    class Meta:
        abstract = True

class ChildA(Base):
    pass

class ChildB(Base):
    pass

Along with another app rare/models.py:

from common.models import Base

class ChildB(Base):
    pass

#The reverse name of the common.ChildA.m2m field will be common_childa_related, 
#the reverse name of the common.ChildB.m2m field will be common_childb_related, 
#the reverse name of the rare.ChildB.m2m field will be rare_childb_related. 


##Model inheritance-Multi-table inheritance
# Each base model class corresponds to its own database table and can be queried and created individually. 

from django.db import models

class Place(models.Model):
    name = models.CharField(max_length=50)
    address = models.CharField(max_length=80)

class Restaurant(Place):
    serves_hot_dogs = models.BooleanField(default=False)
    serves_pizza = models.BooleanField(default=False)


#All of the fields of Place will also be available in Restaurant,
#although the data will reside in a different database table. 

Place.objects.filter(name="Bob's Cafe") #Place.objects contain all Restaurant objects as well
Restaurant.objects.filter(name="Bob's Cafe") #but not vice versa 

#If you have a Place that is also a Restaurant, 
#Get Restaurant instance from lower case restaurant

p = Place.objects.get(id=12)
# If p is a Restaurant object, this will give the child class:
p.restaurant #<Restaurant: ...> 
#or raise a Restaurant.DoesNotExist exception if p is not Restaurant 


## Meta and multi-table inheritance
#child model does not inherit parent’s Meta class. 
#except  ordering  or a get_latest_by attribute

#To disable parent's ordering 
class ChildModel(ParentModel):
    # ...
    class Meta:
        # Remove parent's ordering effect
        ordering = []


##Inheritance and reverse relations in multi table inheritance 
#uses an implicit OneToOneField to link the child and the parent, 
#hence it’s possible to move from the parent down to the child

#However, this uses the default related_name value for ForeignKey and ManyToManyField relations.
#Use explicit related_name else Django will raise a validation error.

#ERROR
class Supplier(Place):
    customers = models.ManyToManyField(Place)
#Correct 
models.ManyToManyField(Place, related_name='provider').






###Models - Proxy model inheritance
#To change the Python behaviour of a model –eg  to change the default manager, 
#or add a new method etc without altering originals Fields 

#Proxy model does not create new DB table 
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)

class MyPerson(Person):
    class Meta:
        proxy = True

    def do_something(self):
        # ...
        pass

#MyPerson class operates on the same database table as its parent Person class. 
#any new instances of Person will also be accessible through MyPerson and vice-versa

#QuerySets still return the model that was requested
#There is no way to get a MyPerson object whenever you query for Person objects. 

p = Person.objects.create(first_name="foobar")
MyPerson.objects.get(first_name="foobar")  #<MyPerson: foobar>

#Can use a proxy model to define a different default ordering on a model. 
class OrderedPerson(Person):
    class Meta:
        ordering = ["last_name"]
        proxy = True

#Person queries will be unordered and OrderedPerson queries will be ordered by last_name.

#A proxy model must inherit from exactly one non-abstract model class. 

#Proxy model managers
#If you don’t specify any model managers on a proxy model, 
#it inherits the managers from its model parents. 

#If you define a manager on the proxy model, it will become the default, 
#although any managers defined on the parent classes will still be available.


from django.db import models

class NewManager(models.Manager):
    # ...
    pass

class MyPerson(Person):
    objects = NewManager()

    class Meta:
        proxy = True

#If you wanted to add a new manager to the Proxy, 
#without replacing the existing default, 
# Create an abstract class for the new manager.

class ExtraManagers(models.Model):
    secondary = NewManager()

    class Meta:
        abstract = True

class MyPerson(Person, ExtraManagers):
    class Meta:
        proxy = True


###Models - Multiple inheritance
#Same as Python multiple inheritances , but Field name “hiding” is not permitted
#For Meta -  if multiple parents contain a Meta class, only the first one is going to be used

class Article(models.Model):
    headline = models.CharField(max_length=50)
    body = models.TextField()

class Book(models.Model):
    title = models.CharField(max_length=50)

class BookReview(Book, Article):
    pass

#Note the below usage caveat because of same id inheritances 
article = Article.objects.create(headline='Some piece of news.')
review = BookReview.objects.create(
	headline='Review of Little Red Riding Hood.',
	title='Little Red Riding Hood')
    
>>> assert Article.objects.get(pk=article.pk).headline == article.headline
Traceback (most recent call last):
  File "<console>", line 1, in <module>
AssertionError
# the "Some piece of news." headline has been overwritten.
Article.objects.get(pk=article.pk).headline
'Review of Little Red Riding Hood.'

#Solution is to use explicit AutoField in the base models:

class Article(models.Model):
    article_id = models.AutoField(primary_key=True)
    ...

class Book(models.Model):
    book_id = models.AutoField(primary_key=True)
    ...

class BookReview(Book, Article):
    pass

#Or use a common ancestor to hold the AutoField:

class Piece(models.Model):
    pass

class Article(Piece):
    ...

class Book(Piece):
    ...

class BookReview(Book, Article):
    pass




###Models - django.db.models.manager.Manager - usef for Query operation 
#By default, Django adds a Manager with the name objects to every Django model class. 

#To use some other names 

from django.db import models

class Person(models.Model):
    #...
    people = models.Manager()

#Person.people.all() will provide a list of all Person objects.


#Custom Managers
# Usecase : to add extra Manager methods to add “table-level” functionality
#and to modify the initial QuerySet the Manager returns.

#Adding extra Manager methods
#Manager methods can access self.model to get the model class to which they’re attached.

#Example - offers extra method with_counts()

from django.db import models

class PollManager(models.Manager):
    def with_counts(self):
        from django.db import connection
        cursor = connection.cursor()
        cursor.execute("""
            SELECT p.id, p.question, p.poll_date, COUNT(*)
            FROM polls_opinionpoll p, polls_response r
            WHERE p.id = r.poll_id
            GROUP BY p.id, p.question, p.poll_date
            ORDER BY p.poll_date DESC""")
        result_list = []
        for row in cursor.fetchall():
            p = self.model(id=row[0], question=row[1], poll_date=row[2])
            p.num_responses = row[3]
            result_list.append(p)
        return result_list

class OpinionPoll(models.Model):
    question = models.CharField(max_length=200)
    poll_date = models.DateField()
    objects = PollManager()

class Response(models.Model):
    poll = models.ForeignKey(OpinionPoll)
    person_name = models.CharField(max_length=50)
    response = models.TextField()


OpinionPoll.objects.with_counts() #returns list of OpinionPoll objects with num_responses attributes.

#Modifying initial Manager QuerySets by overriding the Manager.get_queryset()
#By default Manager’s base QuerySet returns all objects in the system. 

class DahlBookManager(models.Manager):
    def get_queryset(self):
        return super(DahlBookManager, self).get_queryset().filter(author='Roald Dahl') #can use any method of QuerySets

# Then hook it into the Book model explicitly.
class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=50)
    objects = models.Manager() # The default manager.
    dahl_objects = DahlBookManager() # The Dahl-specific manager.

Book.objects.all() #return all books in the database, 
Book.dahl_objects.all() #return the ones written by Roald Dahl.


#You can attach as many Manager() instances to a model as you’d like.
# This is an easy way to define common “filters” for your models.

#the first Manager Django encounters (in the order in which they’re defined in the model)
#is called  “default” Manager which is used in many places inside Django

class AuthorManager(models.Manager):
    def get_queryset(self):
        return super(AuthorManager, self).get_queryset().filter(role='A')

class EditorManager(models.Manager):
    def get_queryset(self):
        return super(EditorManager, self).get_queryset().filter(role='E')

class Person(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    role = models.CharField(max_length=1, choices=(('A', _('Author')), ('E', _('Editor'))))
    people = models.Manager()
    authors = AuthorManager()
    editors = EditorManager()

#usage   
Person.authors.all()
Person.editors.all()
Person.people.all()




#Calling custom QuerySet methods (after sub classing QuerySet) from the Manager

class PersonQuerySet(models.QuerySet):
    def authors(self):
        return self.filter(role='A')

    def editors(self):
        return self.filter(role='E')

class PersonManager(models.Manager):
    def get_queryset(self):
        return PersonQuerySet(self.model, using=self._db)

    def authors(self):
        return self.get_queryset().authors()

    def editors(self):
        return self.get_queryset().editors()

class Person(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    role = models.CharField(max_length=1, choices=(('A', _('Author')), ('E', _('Editor'))))
    people = PersonManager()
    
#Usage 
Person.people.authors()
Person.people.editors()

#OR use like below via as_manager() method without creating explicit Manager subclass 
class Person(models.Model):
    ...
    people = PersonQuerySet.as_manager()

    
#Methods are copied from PersonQuerySet to it's Manager  according to the following rules:
•Public methods are copied by default.
•Private methods (starting with an underscore) are not copied by default.
•Methods with a queryset_only attribute set to False are always copied.
•Methods with a queryset_only attribute set to True are never copied.

class CustomQuerySet(models.QuerySet):
    # Available on both Manager and QuerySet.
    def public_method(self):
        return

    # Available only on QuerySet.
    def _private_method(self):
        return

    # Available only on QuerySet.
    def opted_out_public_method(self):
        return
    opted_out_public_method.queryset_only = True

    # Available on both Manager and QuerySet.
    def _opted_in_private_method(self):
        return
    _opted_in_private_method.queryset_only = False



##Custom managers and model inheritance
1. Managers defined on non-abstract base classes are not inherited by child classes. 
   redeclare it explicitly on the child class.
2. Managers from abstract base classes are always inherited by the child class, 
3. The default manager on a class is either the first manager declared on the class, 
   or the default manager of the first abstract base class in the parent hierarchy, 
   If no default manager is explicitly declared, Django’s normal default manager is used.

class AbstractBase(models.Model):
    # ...
    objects = CustomManager()

    class Meta:
        abstract = True

class ChildA(AbstractBase):
    # ...
    # This class has CustomManager as the default manager(objects)
    pass

class ChildB(AbstractBase):
    # ...
    # An explicit default manager.
    default_manager = OtherManager()

#Example of getting specific manager as default manager 
class ExtraManager(models.Model):
    extra_manager = OtherManager()

    class Meta:
        abstract = True

class ChildC(AbstractBase, ExtraManager):
    # ...
    # Default manager is CustomManager, but OtherManager is
    # also available via the "extra_manager" attribute.
    pass

#Note below when using Manager via abstract class 
ClassA.objects.do_something()
is legal, but:
AbstractBase.objects.do_something()
will raise an exception. 

#Note - shallow copy on Manager must be possible as Djago relies on that internally 
#normal method addition is OK 
import copy
manager = MyManager()
my_copy = copy.copy(manager)


#Using use_for_related_fields field  - for Automatic manager for Django's internal use 
If this attribute is set on the default manager for a model 
Django will use that class whenever it needs to automatically create a manager for the class. 
Otherwise, it will use django.db.models.Manager.
#NOTE: Must not over ride get_queryset when use_for_related_fields is used 

class MyManager(models.Manager):
    use_for_related_fields = True  #must be set at class level 
    # ...




###Models - Model instance reference

##Creating objects 
m = MyModel( all_fields_as_keywords)
m.save()   #creates DB row 

##Hooking create mechanisms 
#Option-1: Add a classmethod on the model class:

from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)

    @classmethod
    def create(cls, title):
        book = cls(title=title)
        # do something with the book
        return book

book = Book.create("Pride and Prejudice")

#Option-2:  Add a method on a custom manager (usually preferred):

class BookManager(models.Manager):
    def create_book(self, title):
        book = self.create(title=title)
        # do something with the book
        return book

class Book(models.Model):
    title = models.CharField(max_length=100)

    objects = BookManager()

book = Book.objects.create_book("Pride and Prejudice")


## Validating objects - Model.full_clean(exclude=None, validate_unique=True)
#call model.full_clean(), which does 
1.Validate the model fields - Model.clean_fields(exclude=None) #exclude is list of fields not to be validated 
2.Validate the model as a whole - Model.clean()
3.Validate the field uniqueness - Model.validate_unique(exclude=None)

#for  ModelForm, is_valid() performs these steps 
#save() does not call full_clean() automatically 

from django.core.exceptions import ValidationError
try:
    article.full_clean()
except ValidationError as e:
    # Do something based on the errors contained in e.message_dict.
    # Display them to a user, or handle them programmatically.
    pass


#Custom model validation - override Model.clean()
# Example : To provide a value for a field, 
#or to do validation that requires access to more than a single field:

import datetime
from django.core.exceptions import ValidationError
from django.db import models

class Article(models.Model):
    #...
    def clean(self):
        # Don't allow draft entries to have a pub_date.
        if self.status == 'draft' and self.pub_date is not None:
            raise ValidationError('Draft entries may not have a publication date.')
        # Set the pub_date for published items if it hasn't been set already.
        if self.status == 'published' and self.pub_date is None:
            self.pub_date = datetime.date.today()

#To access validation errors from ValidationError exception , use e.message_dict[NON_FIELD_ERRORS]
from django.core.exceptions import ValidationError, NON_FIELD_ERRORS
try:
    article.full_clean()
except ValidationError as e:
    non_field_errors = e.message_dict[NON_FIELD_ERRORS]

    
#To assign exceptions to a specific field of Model, 
#instantiate the ValidationError with a dictionary, where the keys are the field names.
#for example: to assign the error to the pub_date field:

class Article(models.Model):
    ...
    def clean(self):
        # Don't allow draft entries to have a pub_date.
        if self.status == 'draft' and self.pub_date is not None:
            raise ValidationError({'pub_date': 'Draft entries may not have a publication date.'})
        ...

##Saving objects - Model.save([force_insert=False, force_update=False, using=DEFAULT_DB_ALIAS, update_fields=None])

#Auto-incrementing primary keys
#If a model has an AutoField — an auto-incrementing primary key — 
#save() would create that field 

b2 = Blog(name='Cheddar Talk', tagline='Thoughts on cheese.')
b2.id     # Returns None, because b doesn't have an ID yet.
b2.save()
b2.id     # Returns the ID of your new object.

#The pk property - Model.pk
- alias of primary key 

#Explicitly specifying auto-primary-key values
#If key exists, it's modification of old row, else creation of new row 

b3 = Blog(id=3, name='Cheddar Talk', tagline='Thoughts on cheese.')
b3.id     # Returns 3.
b3.save()
b3.id     # Returns 3.
#override 
b4 = Blog(id=3, name='Not Cheddar', tagline='Anything but cheese.')
b4.save()  # Overrides the previous blog with ID=3!


##save() process 
 
1. Emit a pre-save signal, django.db.models.signals.pre_save 
2. Pre-process the data for each field , eg DateField with auto_now=True
3. Prepare the data for the database for each field eg DateField
4. Insert the data into the database by SQL 
5. Emit a post-save signal, django.db.models.signals.post_save 

## How Django knows to UPDATE vs. INSERT
#because same save() method for creating and changing objects.
• if primary key is a value other than None or the empty string, UPDATE.
• else It's INSERT 
# Forcing an INSERT or UPDATE
when calling save(), useforce_insert=True or force_update=True 

#Updating attributes based on existing fields eg increment 
#for example 
product = Product.objects.get(name='Venezuelan Beaver Cheese')
product.number_sold += 1
product.save()
#but below is better , by using F()
from django.db.models import F
product = Product.objects.get(name='Venezuelan Beaver Cheese')
product.number_sold = F('number_sold') + 1
product.save()

#Specifying which fields to save when saving only few fields - use update_fields parameter
product.name = 'Name changed again'
product.save(update_fields=['name'])
#An empty update_fields iterable will skip the save. 
#A value of None will perform an update on all fields.




##Deleting objects - Model.delete([using=DEFAULT_DB_ALIAS])
Issues an SQL DELETE for the object. 
This only deletes the object in the database; 
the Python instance will still exist and will still have data in its fields.
For customized deletion behavior, override the delete() method. 



## Other model instance methods
Model.__unicode__() Py2.x
Model.__str__()     Py3.x human readable string , Django uses it internally to many places 
Model.__eq__()      == with the same primary key value and the same concrete class 
Model.__hash__()    based on the instance’s primary key value. 

#Model.get_absolute_url()  
#tells Django how to calculate the canonical URL for an object.

def get_absolute_url(self):
    return "/people/%i/" % self.id
#but below is better , using reverse() on views 
def get_absolute_url(self):
    from django.core.urlresolvers import reverse
    return reverse('people.views.details', args=[str(self.id)])
#usage in template 
<!-- BAD template code. Avoid! -->
<a href="/people/{{ object.id }}/">{{ object.name }}</a>
<!-- GOOD template code. USE!!! -->
<a href="{{ object.get_absolute_url }}">{{ object.name }}</a>

#Model.get_FOO_display() - FOO is Field name 
For every field that has choices , get_FOO_display() prints human readable name 

from django.db import models
class Person(models.Model):
    SHIRT_SIZES = (
        (u'S', u'Small'),
        (u'M', u'Medium'),
        (u'L', u'Large'),
    )
    name = models.CharField(max_length=60)
    shirt_size = models.CharField(max_length=2, choices=SHIRT_SIZES)
    
p = Person(name="Fred Flintstone", shirt_size="L")
p.save()
p.shirt_size
u'L'
p.get_shirt_size_display()
u'Large'

#Model.get_next_by_FOO(**kwargs), Model.get_previous_by_FOO(**kwargs) - FOO is Field name 
For every DateField and DateTimeField that does not have null=True, 
returns the next and previous object with respect to the date field, 
raising a DoesNotExist exception when appropriate.
To perform  custom filtering, use keyword arguments





###Models - Making queries

from django.db import models

class Blog(models.Model):                      #One Blog has many Entry, One Entry belongs to One Blog,  'Many' side use ForeingKey
    name = models.CharField(max_length=100)
    tagline = models.TextField()

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Author(models.Model):
    name = models.CharField(max_length=50)
    email = models.EmailField()

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Entry(models.Model):
    blog = models.ForeignKey(Blog)       #Entry side has 'blog', Blog side has 'entry_set'
    headline = models.CharField(max_length=255)
    body_text = models.TextField()
    pub_date = models.DateField()
    mod_date = models.DateField()
    authors = models.ManyToManyField(Author) #Entry side has 'authors', Author side has 'entry_set'
    n_comments = models.IntegerField()
    n_pingbacks = models.IntegerField()
    rating = models.IntegerField()

    def __str__(self):              # __unicode__ on Python 2
        return self.headline



##Creating objects

from blog.models import Blog
b = Blog(name='Beatles Blog', tagline='All the latest Beatles news.')
b.save()    #performs an INSERT SQL statement

Blog.objects.create(name='Beatles Blog', tagline='All..') #create and save in single step 
 

##Updating changes to objects - using same Primary key 
b5.save()   # first time creation 
b5.name = 'New name'
b5.save()  #next time update

##Saving ForeignKey and ManyToManyField fields
#ForeignKey - Update through 'One' side of Many to One, 
from blog.models import Entry
entry = Entry.objects.get(pk=1)
cheese_blog = Blog.objects.get(name="Cheddar Talk")
entry.blog = cheese_blog  # One side of Many to One , use 'blog' attribute 
entry.save()

#ManyToManyField , use .add() of the field 
from blog.models import Author
joe = Author.objects.create(name="Joe")
entry.authors.add(joe)

#To add multiple records to a ManyToManyField in one go,
john = Author.objects.create(name="John")
paul = Author.objects.create(name="Paul")
george = Author.objects.create(name="George")
ringo = Author.objects.create(name="Ringo")
entry.authors.add(john, paul, george, ringo)


##Retrieving objects - construct a QuerySet via a Manager
Blog.objects            #<django.db.models.manager.Manager object at ...>

#Retrieving all objects, use .all() 
all_entries = Entry.objects.all()

#Retrieving specific objects with filters, use filter(**kwargs), exclude(**kwargs) (inverse of filter)
Entry.objects.filter(pub_date__year=2006)

#it is the same as:
Entry.objects.all().filter(pub_date__year=2006)

#Chaining filters - The result of refining a QuerySet is itself a QuerySet

Entry.objects.filter(
     headline__startswith='What'
 ).exclude(
    pub_date__gte=datetime.date.today()
 ).filter(
     pub_date__gte=datetime(2005, 1, 30)
 )

#Each time you refine a QuerySet, you get a brand-new QuerySet (independent of original)
# QuerySets are lazy
q = Entry.objects.filter(headline__startswith="What")
q = q.filter(pub_date__lte=datetime.date.today())
q = q.exclude(body_text__icontains="food")
print(q)  #it hits the database only once, at the last line (print(q)). 


#Retrieving a single object with get(any_query_expression)
one_entry = Entry.objects.get(pk=1)

#If there are no results that match the query, get() will raise a DoesNotExist exception. 
#Django raises MultipleObjectsReturned if more than one item matches the get() query. 

##Limiting QuerySets - use slice on QuerySet , Entry.objects.all()[-1] not supported 
Entry.objects.all()[:5]
Entry.objects.all()[5:10]

#Not supported below 
Entry.objects.all()[:10:2] #usage of step 
Entry.objects.all()[-1]  #negative index

#To retrieve a single object rather than a list 
Entry.objects.order_by('headline')[0]
#same as 
Entry.objects.order_by('headline')[0:1].get()


##Field lookups - syntax-  field__lookuptype=value
#it can be even on relation eg relatedModel__field__lookuptype=value 

Entry.objects.filter(pub_date__lte='2006-01-01')
#-> SELECT * FROM blog_entry WHERE pub_date <= '2006-01-01';

#for ForeignKey , specify with  _id. 
Entry.objects.filter(blog_id=4)

##Common lookups 
#exact          An “exact” match
Entry.objects.get(headline__exact="Man bites dog")
#-> SELECT ... WHERE headline = 'Man bites dog';

#exact is default when no __ is present in lookup 
Blog.objects.get(id__exact=14)  # Explicit form
Blog.objects.get(id=14)         # __exact is implied

#iexact         A case-insensitive match
Blog.objects.get(name__iexact="beatles blog")

#contains, icontains       Case-sensitive/insensitive containment test
Entry.objects.get(headline__contains='Lennon')
#-> SELECT ... WHERE headline LIKE '%Lennon%';

#startswith, endswith       Starts-with and ends-with search, respectively. 
#istartswith, endswith      Case-insensitive version

##Lookups that span relationships - SQL JOIN
#To span a relationship, use the field name of related fields across models, 
#separated by double underscores
Entry.objects.filter(blog__name='Beatles Blog')

#To refer to a “reverse” relationship, use lowercase name of the model.
Blog.objects.filter(entry__headline__contains='Lennon')

#To span across multiple relationships -use multiple __ 
Blog.objects.filter(entry__authors__name='Lennon')

#if there was no author associated with an entry, 
#no error is raised, but simply ignore that entry 

Blog.objects.filter(entry__authors__name__isnull=True)
#return Blog objects that have an empty name on the author 
#and also those which have an empty author on the entry. 
#If you don’t want those latter objects, use AND in filter 
Blog.objects.filter(entry__authors__isnull=False,     entry__authors__name__isnull=True)

##Spanning multi-valued relationships
#Note all args of filter() are AND of those lookup (ie one SQL with AND)

#Chaining of filter is also disjunction, but one after another(multiple SQL)
#but for multi-valued relations(ie accesing 'Many' from 'One'), 
#many filter() apply to any object linked to the primary model
#not necessarily those objects that were selected by an earlier filter() call
#resulting into OR 

#To select all blogs that contain entries with both “Lennon” in the headline 
#and that were published in 2008 (the same entry satisfying both conditions)
Blog.objects.filter(entry__headline__contains='Lennon',   entry__pub_date__year=2008)

#To select all blogs that contain an entry with “Lennon” in the headline 
#as well as an entry that was published in 2008 (OR for multivalued)

#Blog is One side of Entry, Many side (Blog contains many Entry, Each Entry belongs to One Blog)
#Entry is multi-valued , hence resulting into OR 
#Since We are filtering the Blog items with each filter statement, not the Entry items.
Blog.objects.filter(entry__headline__contains='Lennon').filter( entry__pub_date__year=2008)


##But for exclude() , multiple args in one exclude() is OR 
#To exclude blogs that contain either of criteria 
Blog.objects.exclude( entry__headline__contains='Lennon',  entry__pub_date__year=2008,)

#to Exclude blogs that contain both of these criteria , use like below 
Blog.objects.exclude(  entry=Entry.objects.filter(    headline__contains='Lennon',    pub_date__year=2008, ),)



##Filters can reference fields on the model - use F()
#To compare the value of a model field with another field on the same model

#to find a list of all blog entries that have had more comments than pingbacks, 
from django.db.models import F
Entry.objects.filter(n_comments__gt=F('n_pingbacks'))

#Django supports the use of addition, subtraction, multiplication, division, 
#modulo, and power arithmetic with F() objects, both with constants and with other F() 

#To find all the blog entries with more than twice as many comments as pingbacks
Entry.objects.filter(n_comments__gt=F('n_pingbacks') * 2)

#To find all the entries where the rating of the entry is less than the 
#sum of the pingback count and comment count
Entry.objects.filter(rating__lt=F('n_comments') + F('n_pingbacks'))

#Can also use the double underscore notation to span relationships in an F() object. 

#to retrieve all the entries where the author’s name is the same as the blog name,
Entry.objects.filter(authors__name=F('blog__name'))

#For date and date/time fields, can add or subtract a timedelta object with F() 

#all entries that were modified more than 3 days after they were published:
from datetime import timedelta
Entry.objects.filter(mod_date__gt=F('pub_date') + timedelta(days=3))

#The F() objects support bitwise operations by .bitand() and .bitor()
F('somefield').bitand(16)



##The pk lookup shortcut - alias of Primary Key 

#equivalent
Blog.objects.get(id__exact=14) 	# Explicit form
Blog.objects.get(id=14) 		# __exact is implied
Blog.objects.get(pk=14) 		# pk implies id__exact

#any query term can be combined with pk to perform a query on the primary key of a model:

# Get blogs entries with id 1, 4 and 7
Blog.objects.filter(pk__in=[1,4,7])

# Get all blog entries with id > 14
Blog.objects.filter(pk__gt=14)

#pk lookups also work across joins. 
#For example, these three statements are equivalent:

Entry.objects.filter(blog__id__exact=3) 	# Explicit form
Entry.objects.filter(blog__id=3)        	# __exact is implied
Entry.objects.filter(blog__pk=3)        	# __pk implies __id__exact


##Escaping percent signs and underscores in LIKE statements
#The field lookups that equate to LIKE SQL statements will automatically escape 
#the two special characters used in LIKE statements – %(multi char wildcard) and _ (single char wildcard)
#for example (iexact, contains, icontains, startswith, istartswith, endswith and iendswith) 

#to retrieve all the entries that contain a percent sign, 
#just use the percent sign as any other character:
Entry.objects.filter(headline__contains='%')
#-> SELECT ... WHERE headline LIKE '%\%%';



##Caching and QuerySets
#Each QuerySet contains a cache to minimize database access. 
#In a newly created QuerySet, the cache is empty.
#The first time a QuerySet is evaluated fully (not partially eg slice, index)
# – and, hence, a database query happens – cache is filled and used always 

#Below would execute two times the same query, but with different cache, increasing time 
print([e.headline for e in Entry.objects.all()])
print([e.pub_date for e in Entry.objects.all()])

#To avoid , save and reuse 
queryset = Entry.objects.all()
print([p.headline for p in queryset]) # Evaluate the query set.
print([p.pub_date for p in queryset]) # Re-use the cache from the evaluation.


#Querysets do not always cache their results eg when using slice and indexing on result 
#repeatedly getting a certain index in a queryset object will query the database each time:
queryset = Entry.objects.all()
print queryset[5] # Queries the database
print queryset[5] # Queries the database again


#if the entire queryset has already been evaluated, 
#the cache will be checked instead:

queryset = Entry.objects.all()
[entry for entry in queryset] 	# Queries the database
print queryset[5] 			# Uses cache
print queryset[5] 			# Uses cache


#other actions that will result in the entire queryset being evaluated 
#and therefore populate the cache:
[entry for entry in queryset]
bool(queryset)
entry in queryset
list(queryset)

#Simply printing the queryset will not populate the cache. 
#This is because the call to __repr__() only returns a slice of the entire queryset.



## Complex lookups with Q objects - using django.db.models.Q()
#Keyword argument queries – in filter(), etc. – are “AND”ed together. 
#If you need to execute more complex queries (for example, queries with OR statements), 
#you can use Q objects.

#it is used to encapsulate a collection of keyword arguments as in “Field lookups”

from django.db.models import Q
Q(question__startswith='What')   #single LIKE 

#Q objects can be combined using the & and | operators.
Q(question__startswith='Who') | Q(question__startswith='What')
#-> WHERE question LIKE 'Who%' OR question LIKE 'What%'

#Q objects can be negated using the ~ operator
Q(question__startswith='Who') | ~Q(pub_date__year=2005)


#Each lookup function that takes keyword-arguments 
#(e.g. filter(), exclude(), get()) can also be passed one or more Q objects 
#But thos would be “AND”ed 

Poll.objects.get( Q(question__startswith='Who'), Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6)))

#Lookup functions can mix the use of Q objects and keyword arguments 
#are “AND”ed 
#However, if a Q object is provided, it must precede the definition of any keyword arguments. 
Poll.objects.get(   Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6)),   question__startswith='Who')



##Comparing objects
#following two statements are equivalent:

some_entry == other_entry
some_entry.id == other_entry.id

#Comparisons will always use the primary key eg if name is Primary key 
some_obj == other_obj
some_obj.name == other_obj.name


#Deleting objects- delete()
#This method immediately deletes the object 
e.delete()

#delete objects in bulk (gets executed in SQL, instance delete() is not called)
#Manger does not have delete() ie Entry.objects.delete() not possible 
#but below is possible as delete() is on QuerySet '
Entry.objects.all().delete()
#this deletes all Entry objects with a pub_date year of 2005
Entry.objects.filter(pub_date__year=2005).delete()

#When Django deletes an object, by default it emulates ON DELETE CASCADE
#This cascade behavior is customizable via the on_delete argument to the ForeignKey.

b = Blog.objects.get(pk=1)
# This will delete the Blog and all of its Entry objects.
b.delete()

#Copying model instances
#no built in, but can be done as below 
blog = Blog(name='My blog', tagline='Blogging is easy')
blog.save() # blog.pk == 1

blog.pk = None
blog.save() # blog.pk == 2


#Due to how inheritance works, you have to set both pk and id to None:

class ThemeBlog(Blog):
    theme = models.CharField(max_length=200)

django_blog = ThemeBlog(name='Django', tagline='Django is easy', theme='python')
django_blog.save() # django_blog.pk == 3

django_blog.pk = None
django_blog.id = None
django_blog.save() # django_blog.pk == 4

#This process does not copy related objects. 
#To do that, execute code as below 
entry = Entry.objects.all()[0] # some previous entry
old_authors = entry.authors.all()
entry.pk = None
entry.save()
entry.authors = old_authors # saves new many2many relations


##Updating multiple objects at once - use QuerySet.update(), returns number of row updated 

# Update all the headlines with pub_date in 2007.
Entry.objects.filter(pub_date__year=2007).update(headline='Everything is the same')

#You can only set non-relation fields and ForeignKey fields using this method. 
#To update a non-relation field, provide the new value as a constant. 
#To update ForeignKey fields, set the new value to be the new model instance 

b = Blog.objects.get(pk=1)
# Change every Entry so that it belongs to this Blog.
Entry.objects.all().update(blog=b)

#The only restriction on the QuerySet that is updated is that 
#it can only access the model’s main table. 
#You can filter based on related fields, but you can only update columns 
#in the model’s main table. 

b = Blog.objects.get(pk=1)
# Update all the headlines belonging to this Blog.
Entry.objects.select_related().filter(blog=b).update(headline='Everything is the same')

#Be aware that the update() method is executed via  SQL statement. 
#It doesn’t run any save() methods or emit the pre_save or post_save signals
#To do that, call save() manually 
for item in my_queryset:
    item.save()


#Calls to update can also use F expressions 
Entry.objects.all().update(n_pingbacks=F('n_pingbacks') + 1)

#unlike F() objects in filter and exclude clauses, 
#you can’t introduce joins when you use F() objects in an update 
#– you can only reference fields local to the model being updated. 
# THIS WILL RAISE A FieldError
Entry.objects.update(headline=F('blog__name'))



###Related Objects
#Note:  all the models you’re using be defined in applications listed in INSTALLED_APPS.
#Otherwise, backwards relations may not work properly.

##One-to-many relationships
#Forward - use the attribute field name 

e = Entry.objects.get(id=2)
e.blog # Returns the related Blog object.

#get/set 
e = Entry.objects.get(id=2)
e.blog = some_blog
e.save()

#If a ForeignKey field has null=True set (i.e., it allows NULL values), 
#you can assign None to remove the relation. Example:

e = Entry.objects.get(id=2)
e.blog = None
e.save() # "UPDATE blog_entry SET blog_id = NULL ...;"

#Forward access to one-to-many relationships is cached the first time the related object is accessed. 

e = Entry.objects.get(id=2)
print(e.blog)  # Hits the database to retrieve the associated Blog.
print(e.blog)  # Doesn't hit the database; uses cached version.

#select_related() QuerySet method recursively prepopulates the cache of all 
#one-to-many relationships ahead of time
e = Entry.objects.select_related().get(id=2)
print(e.blog)  # Doesn't hit the database; uses cached version.
print(e.blog)  # Doesn't hit the database; uses cached version.


#Following relationships “backward” - lowercasedOtherSideModel_set - it's a Manager instance 
#Every addition, creation and deletion is immediately and automatically saved to the database.

b = Blog.objects.get(id=1)
b.entry_set.all() # Returns all Entry objects related to Blog.

# b.entry_set is a Manager that returns QuerySets.
b.entry_set.filter(headline__contains='Lennon')
b.entry_set.count()

#Customize lowercasedOtherSideModel_set by setting the related_name parameter in the ForeignKey 

blog = ForeignKey(Blog, related_name='entries')

b = Blog.objects.get(id=1)
b.entries.all() # Returns all Entry objects related to Blog.

# b.entries is a Manager that returns QuerySets.
b.entries.filter(headline__contains='Lennon')
b.entries.count()

##Using a custom reverse manager - use entry_set(newManager) 

from django.db import models

class Entry(models.Model):
    #...
    objects = models.Manager()  # Default Manager
    entries = EntryManager()    # Custom Manager

b = Blog.objects.get(id=1)
b.entry_set(manager='entries').all()  #use manager keyword 
                                      #all() would use EntryManger's get_queryset() method,

#Also, specifying a custom reverse manager also enables to call its custom methods:
b.entry_set(manager='entries').is_published() #is_published() is present in EntryManager

##Additional methods to handle related objects
add(obj1, obj2, ...)        Adds the specified model objects to the related object set.
create(**kwargs)            Creates a new object, saves it and puts it in the related object set. Returns the newly created object.
remove(obj1, obj2, ...)     Removes the specified model objects from the related object set.
clear()                     Removes all objects from the related object set.

#To assign the members of a related set in one step
b = Blog.objects.get(id=1)
b.entry_set = [e1, e2]


##Many-to-many relationships
#The model that defines the ManyToManyField uses the attribute name of that field itself,
#whereas the “reverse” model uses lowercasedOtherSideModel_set - it's a Manager instance 
#Customize lowercasedOtherSideModel_set by setting the related_name parameter in the ManyToManyField

e = Entry.objects.get(id=3)
e.authors.all() # Returns all Author objects for this Entry.
e.authors.count()
e.authors.filter(name__contains='John')

a = Author.objects.get(id=5)
a.entry_set.all() # Returns all Entry objects for this Author.




##One-to-one relationships
#The model that defines the OneToOneField uses attribute name 
#The other side use lowercasedOtherSideModel


class EntryDetail(models.Model):
    entry = models.OneToOneField(Entry)
    details = models.TextField()

ed = EntryDetail.objects.get(id=2)
ed.entry # Returns the related Entry object.

#reverse side 
e = Entry.objects.get(id=2)
e.entrydetail # returns the related EntryDetail object

#If no object has been assigned to this relationship, 
#Django will raise a DoesNotExist exception.

#Instances can be assigned to the reverse relationship
e.entrydetail = ed



##Queries over related objects - similar to normal case 
#the following three queries would be identical:
Entry.objects.filter(blog=b) # Query using object instance
Entry.objects.filter(blog=b.id) # Query using id from instance
Entry.objects.filter(blog=5) # Query using id directly


##Performing raw SQL queries - using  Manager.raw(raw_query, params=None, translations=None)
#returns a RawQuerySet object 
#While a RawQuerySet instance can be iterated over like a normal QuerySet, 
#RawQuerySet doesn’t implement all methods you can use with QuerySet. 
#If the query does not return rows, a (possibly cryptic) error will result.
class Person(models.Model):
    first_name = models.CharField(...)
    last_name = models.CharField(...)
    birth_date = models.DateField(...)
You could then execute custom SQL like so:

for p in Person.objects.raw('SELECT * FROM myapp_person'):
    print(p)



#Mapping query fields to model fields
#raw() automatically maps fields in the query to fields on the model.
#The order of fields in your query doesn’t matter

#both are same 
Person.objects.raw('SELECT id, first_name, last_name, birth_date FROM myapp_person')
Person.objects.raw('SELECT last_name, birth_date, first_name, id FROM myapp_person')

#Matching is done by name. 
#you can use SQL’s AS clauses to map fields in the query to model fields. 

Person.objects.raw('''SELECT first AS first_name,
                              last AS last_name,
                             bd AS birth_date,
                             pk AS id,
                     FROM some_other_table''')


#Or use translations keyword 
name_map = {'first': 'first_name', 'last': 'last_name', 'bd': 'birth_date', 'pk': 'id'}
Person.objects.raw('SELECT * FROM some_other_table', translations=name_map)

#raw() supports indexing, so if you need only the first result 
first_person = Person.objects.raw('SELECT * FROM myapp_person')[0]

#However, the indexing and slicing are not performed at the database level. 
#for efficient implementation do at SQL level for raw()
first_person = Person.objects.raw('SELECT * FROM myapp_person LIMIT 1')[0]


#Fields may also be left out in raw()
#in this , query returns deferred model instances 
#This means that the fields that are omitted from the query will be loaded on demand. 
#There is only one field that you can’t leave out - the primary key field

people = Person.objects.raw('SELECT id, first_name FROM myapp_person')

for p in Person.objects.raw('SELECT id, first_name FROM myapp_person'):
    print(p.first_name, # This will be retrieved by the original query
        p.last_name) # This will be retrieved on demand



##Adding annotations in raw()
#You can also execute queries containing fields that aren’t defined on the model. 
#For example, we could use PostgreSQL’s age() function to get a list of people with their ages calculated by the database:

people = Person.objects.raw('SELECT *, age(birth_date) AS age FROM myapp_person')
for p in people:
    print("%s is %s." % (p.first_name, p.age))


#Passing parameters into raw()- use the params argument to raw():
#params is a list or dictionary of parameters. 
#You’ll use %s placeholders in the query string for a list, 
#or %(key)s placeholders for a dictionary 



lname = 'Doe'
Person.objects.raw('SELECT * FROM myapp_person WHERE last_name = %s', [lname])

#Dictionary params are not supported with the SQLite backend; 
#with this backend, you must pass parameters as a list.



#Do not use string formatting on raw queries, prone to SQL injection attack 
# don't do below
query = 'SELECT * FROM myapp_person WHERE last_name = %s' % lname
Person.objects.raw(query)


##Performing raw SQL queries - Executing custom SQL directly by bypassing model layer
#note that the SQL statement in cursor.execute() uses placeholders, "%s", 
#not the "?" placeholder

from django.db import connection

def my_custom_sql(self):
    cursor = connection.cursor()
    cursor.execute("UPDATE bar SET foo = 1 WHERE baz = %s", [self.baz])
    cursor.execute("SELECT foo FROM bar WHERE baz = %s", [self.baz])
    row = cursor.fetchone()
    return row

#to include literal percent signs in the query, excape it by %
cursor.execute("SELECT foo FROM bar WHERE baz = '30%'")
cursor.execute("SELECT foo FROM bar WHERE baz = '30%%' AND id = %s", [self.id])

#If you are using more than one database, use django.db.connections 
from django.db import connections
cursor = connections['my_db_alias'].cursor()

#By default, the Python DB API will return results without their field names
#To get field name as dict key 

def dictfetchall(cursor):
    "Returns all rows from a cursor as a dict"
    desc = cursor.description
    return [ dict(zip([col[0] for col in desc], row))  for row in cursor.fetchall()    ]

#Here is an example of the difference between the two:
cursor.execute("SELECT id, parent_id FROM test LIMIT 2");
cursor.fetchall()
((54360982L, None), (54360880L, None))
#using dictfetchall
cursor.execute("SELECT id, parent_id FROM test LIMIT 2");
dictfetchall(cursor)
[{'parent_id': None, 'id': 54360982L}, {'parent_id': None, 'id': 54360880L}]


#Using a cursor as a context manager:

with connection.cursor() as c:
    c.execute(...)
    
#is equivalent to:
c = connection.cursor()
try:
    c.execute(...)
finally:
    c.close()




###Models - Querying Aggregation - use aggregate() or count() etc 
#to introduce new summary field , use annotate()

from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()

class Publisher(models.Model):
    name = models.CharField(max_length=300)
    num_awards = models.IntegerField()

class Book(models.Model):
    name = models.CharField(max_length=300)
    pages = models.IntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    rating = models.FloatField()
    authors = models.ManyToManyField(Author) #Many Author has Many Book
    publisher = models.ForeignKey(Publisher)  #Publisher may have many Book
    pubdate = models.DateField()

class Store(models.Model):
    name = models.CharField(max_length=300)
    books = models.ManyToManyField(Book)
    registered_users = models.PositiveIntegerField()

# Total number of books.
Book.objects.count() #2452

# Total number of books with publisher=BaloneyPress
Book.objects.filter(publisher__name='BaloneyPress').count() #73

# Average price across all books.
from django.db.models import Avg
Book.objects.all().aggregate(Avg('price'))  #{'price__avg': 34.35}

# Max price across all books.
from django.db.models import Max
Book.objects.all().aggregate(Max('price'))  #{'price__max': Decimal('81.20')}

# All the following queries involve traversing the Book<->Publisher
# many-to-many relationship backward

# Each publisher, each with a count of books as a "num_books" attribute.
from django.db.models import Count
pubs = Publisher.objects.annotate(num_books=Count('book'))
pubs  #[<Publisher BaloneyPress>, <Publisher SalamiPress>, ...]
pubs[0].num_books #73

# The top 5 publishers, in order by number of books.
pubs = Publisher.objects.annotate(num_books=Count('book')).order_by('-num_books')[:5]
pubs[0].num_books #1323

###aggregate() is a terminal clause for a QuerySet that, when invoked, 
#returns a dictionary of name-value pairs. 

Book.objects.aggregate(average_price=Avg('price')) #{'average_price': 34.35}

#more than one aggregate 
from django.db.models import Avg, Max, Min
Book.objects.aggregate(Avg('price'), Max('price'), Min('price'))
#{'price__avg': 34.35, 'price__max': Decimal('81.20'), 'price__min': Decimal('12.99')}


#Generating aggregates for each item in a QuerySet - use annotate()
#Each argument to annotate() describes an aggregate that is to be calculated. 

#For example, to annotate books with the number of authors:

# Build an annotated queryset
from django.db.models import Count
q = Book.objects.annotate(Count('authors'))
# Interrogate the first object in the queryset
q[0] #<Book: The Definitive Guide to Django>
q[0].authors__count #2
# Interrogate the second object in the queryset
q[1] #<Book: Practical Django Projects>
q[1].authors__count #1

#To give annotated filed name 
q = Book.objects.annotate(num_authors=Count('authors'))
q[0].num_authors #2
q[1].num_authors #1

#Unlike aggregate(), annotate() is not a terminal clause. 
#The output of the annotate() clause is a QuerySet; 
#this QuerySet can be modified using filter(), order_by(), or annotate().
#hence to get result of QuerySet(), use .get() after .annotate()

#combining multiple aggregations with annotate() will yield the wrong results, 
#as multiple tables are cross joined, resulting in duplicate row aggregations.

##*** In order to understand what happens in your query - use query attribute 

from django.contrib.auth.models import User
print User.objects.filter(last_name__icontains = 'ax').query

#if you have DEBUG = True, then all of your queries are logged
#get thos logs from 
from django.db import connections
connections['default'].queries

##Joins and aggregates - use __ for any field in annotate() and aggregate()

#to find the price range of books offered in each store

from django.db.models import Max, Min
Store.objects.annotate(min_price=Min('books__price'), max_price=Max('books__price'))

#This  retrieves the Store model, 
#join (through the many-to-many relationship) with the Book model, 
#and aggregate on the price field of the book model to produce a minimum and maximum value.

#to know the lowest and highest price of any book that is available for sale in a store, 

Store.objects.aggregate(min_price=Min('books__price'), max_price=Max('books__price'))


#Join chains can be as deep as you require. 
#to extract the age of the youngest author of any book available for sale
Store.objects.aggregate(youngest_age=Min('books__authors__age'))


##Following relationships backwards in aggregate() and annotate()
# The lowercase name of related models and double-underscores are used here too.

#for all publishers, annotated with their respective total book stock counters 
#(note how we use 'book' to specify the Publisher -> Book reverse foreign key hop):
#(Every Publisher in the resulting QuerySet will have attribute called book__count)

from django.db.models import Count, Min, Sum, Avg
Publisher.objects.annotate(Count('book'))

#for the oldest book of any of those managed by every publisher:
#(The resulting dictionary will have a key called 'oldest_pubdate'.
# If no such alias were specified, it would be the rather long 'book__pubdate__min'.)

Publisher.objects.aggregate(oldest_pubdate=Min('book__pubdate'))

#with many-to-many relations. 
#for every author, annotated with the total number of pages considering 
#all the books the author has (co-)authored 
#(note how we use 'book' to specify the Author -> Book reverse many-to-many hop):
#(Every Author in the resulting QuerySet will have attribute called total_pages. 
#If no such alias were specified, it would be the rather long book__pages__sum.)

Author.objects.annotate(total_pages=Sum('book__pages'))

#for the average rating of all the books written by author(s) we have on file:
#(The resulting dictionary will have a key called 'average__rating'. 
#If no such alias were specified, it would be the rather long 'book__rating__avg'.)

Author.objects.aggregate(average_rating=Avg('book__rating'))



## Aggregations and other QuerySet clauses-filter() and exclude()
#Aggregate/annotate  can also participate after filters. 

#When used with an annotate() clause, 
#a filter has the effect of constraining the objects for which an annotation is calculated. 

#To generate an annotated list of all books that have a title starting with “Django” using the query:

from django.db.models import Count, Avg
Book.objects.filter(name__startswith="Django").annotate(num_authors=Count('authors'))

#When used with an aggregate() clause, 
#a filter has the effect of constraining the objects over which the aggregate is calculated. For example, you can generate the average price of all books with a title that starts with “Django” using the query:

Book.objects.filter(name__startswith="Django").aggregate(Avg('price'))



##Filtering on annotations
#Annotated values can also be filtered. 

#to generate a list of books that have more than one author
Book.objects.annotate(num_authors=Count('authors')).filter(num_authors__gt=1)

##Order of annotate() and filter() clauses
#filter() and annotate() are not commutative operations 

#When an annotate() clause is applied to a query, 
#the annotation is computed over the state of the query up to the point 
#where the annotation is requested. 


#both are different 
Publisher.objects.annotate(num_books=Count('book')).filter(book__rating__gt=3.0)
Publisher.objects.filter(book__rating__gt=3.0).annotate(num_books=Count('book'))

#the annotation in the first query will provide the total number of all books published 
#by the publisher; 
#the second query will only include good books in the annotated count. 


##Annotations and order_by()
#Annotations can be used as a basis for ordering. 

#to order a QuerySet of books by the number of authors that have contributed to the book, you could use the following query:
Book.objects.annotate(num_authors=Count('authors')).order_by('num_authors')


##Annotationa and values() - with values() , annotate() behaves like groupBy 
#Ordinarily, annotations are generated on a per-object basis 
#- an annotated QuerySet will return one result for each object in the original QuerySet.
 
#With values(), An annotation is  provided for each unique group; 
#the annotation is computed over all members of the group.

#return one result for each author in the database, annotated with their average book rating.
Author.objects.annotate(average_rating=Avg('book__rating'))


#authors will be grouped by name, get an annotated result for each unique author name
Author.objects.values('name').annotate(average_rating=Avg('book__rating'))


##Order of annotate() and values() clauses
#If the values() clause precedes the annotate(), 
#the annotation will be computed using the grouping described by the values() clause.

#if the annotate() clause precedes the values() clause, 
#the annotations will be generated over the entire query set. 
#and the values() clause only constrains the fields that are generated on output.

#yield one unique result for each author; 
#however, only the author’s name and the average_rating annotation will be returned in the output data.

Author.objects.annotate(average_rating=Avg('book__rating')).values('name', 'average_rating')


##Interaction with default ordering or order_by()
#default ordering is always part of GroupBy when used with values() and annotate()


from django.db import models

class Item(models.Model):
    name = models.CharField(max_length=10)
    data = models.IntegerField()

    class Meta:
        ordering = ["name"]

        
# Warning: not quite correct!
#annotate() would happen on group by distinct (data, name) pairs
Item.objects.values("data").annotate(Count("id"))

#To reset default ordering , use order_by() at last method 
Item.objects.values("data").annotate(Count("id")).order_by()


## Aggregating annotations - aggregate on the result of an annotation

#to calculate the average number of authors per book you first annotate the set of books with the author count, 
#then aggregate that author count, referencing the annotation field:

from django.db.models import Count, Avg
Book.objects.annotate(num_authors=Count('authors')).aggregate(Avg('num_authors'))
#{'num_authors__avg': 1.66}









###Query - Lookup API reference

#A lookup expression consists of three parts(always separated by __) 
•Fields part (e.g. Book.objects.filter(author__best_friends__first_name...);
•Transforms part(could be many) (may be omitted) (e.g. __lower__first3chars__reversed); (instance of class Transform)
•A lookup (e.g. __icontains) that, if omitted, defaults to __exact  (instance of class  Lookup)

Django uses RegisterLookupMixin to give a class the interface to register lookups on itself
The query expression API is a common set of methods that classes define to be usable 
in query expressions to translate themselves into SQL expressions 
Note Field, Transform, Aggregate, Lookup etc  have get_lookup(), get_transform() methods from Query Language API 

•.filter(myfield__mylookup) will call myfield.get_lookup('mylookup').
•.filter(myfield__mytransform__mylookup) will call myfield.get_transform('mytransform'), 
  and then mytransform.get_lookup('mylookup').
•.filter(myfield__mytransform) will first call myfield.get_lookup('mytransform'), 
  which will fail, so it will fall back to calling myfield.get_transform('mytransform')
  and then mytransform.get_lookup('exact').


###Djano- List of Lookup 
#They’re specified as keyword arguments to the QuerySet methods 
#filter(), exclude() and get().

#prefix = i  , means case insensitive 

#exact, iexact
Exact match. 
If the value provided for comparison is None, it will be interpreted as an SQL NULL 

Entry.objects.get(id__exact=14)
Entry.objects.get(id__exact=None)
Blog.objects.get(name__iexact='beatles blog')

#SQL equivalents:
SELECT ... WHERE id = 14;
SELECT ... WHERE id IS NULL;
SELECT ... WHERE name ILIKE 'beatles blog';



#contains, icontains
Case-sensitive/insensitive containment test.

Entry.objects.get(headline__contains='Lennon')
#SQL equivalent:
SELECT ... WHERE headline LIKE '%Lennon%';

#in
In a given list.

Entry.objects.filter(id__in=[1, 3, 4])
#SQL equivalent:
SELECT ... WHERE id IN (1, 3, 4);

#Or can use another QuerySet 
inner_qs = Blog.objects.filter(name__contains='Cheddar')
entries = Entry.objects.filter(blog__in=inner_qs)

#Using with values(), value_list(), must result into single field 
#OK 
inner_qs = Blog.objects.filter(name__contains='Ch').values('name')
entries = Entry.objects.filter(blog__name__in=inner_qs)
#NOK
# Bad code! Will raise a TypeError.
inner_qs = Blog.objects.filter(name__contains='Ch').values('name', 'id')
entries = Entry.objects.filter(blog__name__in=inner_qs)

#Relation ship operator 
gt      Greater than
gte     Greater than or equal to.
lt      Less than.
lte     Less than or equal to.


Entry.objects.filter(id__gt=4)
#SQL equivalent:
SELECT ... WHERE id > 4;


#startswith, istartswith, endswith,iendswith
Case-sensitive/insensitive  starts-with/ends-with 

Entry.objects.filter(headline__iendswith='will')
#SQL equivalent:
SELECT ... WHERE headline ILIKE '%will'

#range
Range test (inclusive).
You can use range anywhere you can use BETWEEN in SQL — for dates, numbers and even characters
But note Generally speaking, you can’t mix dates and datetimes.


import datetime
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date(2005, 3, 31)
Entry.objects.filter(pub_date__range=(start_date, end_date))

#SQL equivalent:
SELECT ... WHERE pub_date BETWEEN '2005-01-01' and '2005-03-31';


#date
For datetime fields, casts the value as date. 
Allows chaining additional field lookups. Takes a date value.

Entry.objects.filter(pub_date__date=datetime.date(2005, 1, 1))
Entry.objects.filter(pub_date__date__gt=datetime.date(2005, 1, 1))

#year, month, day, week_day, hour, minute, second

For date and datetime fields, an exact year/month/day/week_day match. 
For datetime/time fields, an exact hour, minute, second match 
Allows chaining additional field lookups. 
Takes an integer year/month/day/week_day(1 (Sunday) to 7 (Saturday))/hour/minute/second

Entry.objects.filter(pub_date__year=2005)
Entry.objects.filter(pub_date__year__gte=2005)

Entry.objects.filter(pub_date__week_day=2)
Entry.objects.filter(pub_date__week_day__gte=2)

Event.objects.filter(timestamp__hour=23)
Event.objects.filter(time__hour=5)
Event.objects.filter(timestamp__hour__gte=12)

#isnull
Takes either True or False, which correspond to SQL queries of IS NULL and IS NOT NULL, respectively.

Entry.objects.filter(pub_date__isnull=True)

#search   - Deprecated since version 1.10: 
A boolean full-text search, taking advantage of full-text indexing


#Django1.10 
Replace it with a custom lookup:


from django.db import models

class Search(models.Lookup):
    lookup_name = 'search'

    def as_mysql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = lhs_params + rhs_params
        return 'MATCH (%s) AGAINST (%s IN BOOLEAN MODE)' % (lhs, rhs), params

models.CharField.register_lookup(Search)
models.TextField.register_lookup(Search)

#regex, iregex
Case-sensitive/insensitive  regular expression match

Entry.objects.get(title__regex=r'^(An?|The) +')
Entry.objects.get(title__iregex=r'^(an?|the) +')




###Djano- List of Aggregate functions (can be used inside annotate() as well)
#module -  django.db.models 

#SQLite can’t handle aggregation on date/time fields out of the box
#Aggregation functions(other than Count) return None (Count returns 0) when used with an empty QuerySet
#Almost all aggregates have the following parameters in common:
expression      A string that references a field on the model, or a query expression.
output_field    An optional argument that represents the model field of the return value



#Usage 
from django.db.models import Avg 
Book.objects.filter(name__startswith="Django").aggregate(Avg('price'))

from django.db.models import Count
Company.objects.annotate(
    managers_required=(Count('num_employees') / 4) + Count('num_managers'))




#Avg, Sum 
class django.db.models.Avg(expression, output_field=FloatField(), **extra)
class Sum(expression, output_field=None, **extra)[source]¶
Returns the mean value/sum of the given expression, which must be numeric 
unless you specify a different output_field.
•Default alias: <field>__avg, <field>__sum

#Count 
class django.db.models.Count(expression, distinct=False, **extra)
Returns the number of objects that are related through the provided expression.
•Default alias: <field>__count
•Return type: int
distinct    If distinct=True, the count will only include unique instances. 
            The default value is False.



#Max, Min 
class django.db.models.Max(expression, output_field=None, **extra)
class django.db.models.Min(expression, output_field=None, **extra)
Returns the maximum/minimum value of the given expression.
•Default alias: <field>__max, <field>__min



#StdDev, Variance
class StdDev(expression, sample=False, **extra)
class Variance(expression, sample=False, **extra)
Returns the standard deviation/variance of the data in the provided expression.
•Default alias: <field>__stddev, <field>__variance
sample      By default, StdDev returns the population standard deviation. 


#An aggregate expression is a special case of a Func() expression 
#Aggregate API 
class Aggregate(expression, output_field=None, **extra)

template
A class attribute, as a format string, that describes the SQL that is generated for this aggregate. 
Defaults to '%(function)s( %(expressions)s )'.

function
A class attribute describing the aggregate function that will be generated. 
Defaults to None.

expression
A string that references a field on the model, or a query expression.



#Creating your own Aggregate Functions
from django.db.models import Aggregate

class Count(Aggregate):
    # supports COUNT(distinct field)
    function = 'COUNT'
    template = '%(function)s(%(distinct)s%(expressions)s)'

    def __init__(self, expression, distinct=False, **extra):
        super(Count, self).__init__(
            expression,
            distinct='DISTINCT ' if distinct else '',
            output_field=IntegerField(),
            **extra )





###Djano- List of Database Functions(can be used inside Annotate())
#(some of them can be registered as Transform)
#Module - django.db.models.functions  


#Example 
class Author(models.Model):
    name = models.CharField(max_length=50)
    age = models.PositiveIntegerField(null=True, blank=True)
    alias = models.CharField(max_length=50, null=True, blank=True)
    goes_by = models.CharField(max_length=50, null=True, blank=True)

#Cast  (Django-1.10)
class django.db.models.functions.Cast(expression, output_field)
Forces the result type of expression to be the one from output_field.

#A Value() object represents a simple value eg 1, Django converts to Value(1)

from django.db.models import FloatField
from django.db.models.functions import Cast
Value.objects.create(integer=4)  #Value.objects only in Django1.10)
value = Value.objects.annotate(as_float=Cast('integer', FloatField)).get()
print(value.as_float)  #4.0


#Coalesce
class django.db.models.functions.Coalesce(*expressions, **extra)
Accepts a list of at least two field names or expressions and returns the first non-null 
value (note that an empty string is not considered a null value)

# Get a screen name from least to most public
from django.db.models import Sum, Value as V
from django.db.models.functions import Coalesce
Author.objects.create(name='Margaret Smith', goes_by='Maggie')
author = Author.objects.annotate(screen_name=Coalesce('alias', 'goes_by', 'name')).get()
print(author.screen_name) #Maggie

# Prevent an aggregate Sum() from returning None
aggregated = Author.objects.aggregate(
        combined_age=Coalesce(Sum('age'), V(0)),
        combined_age_default=Sum('age'))
print(aggregated['combined_age'])#0
print(aggregated['combined_age_default'])#None

#Concat
class django.db.models.functions.Concat(*expressions, **extra)
Accepts a list of at least two text fields or expressions and returns the concatenated text

# Get the display name as "name (goes_by)"
from django.db.models import CharField, Value as V
from django.db.models.functions import Concat
Author.objects.create(name='Margaret Smith', goes_by='Maggie')
author = Author.objects.annotate(
            screen_name=Concat('name', V(' ('), 'goes_by', V(')'),
            output_field=CharField())).get()
print(author.screen_name) #Margaret Smith (Maggie)



#Greatest, Least (Django-1.9)
class django.db.models.functions.Greatest(*expressions, **extra)
class django.db.models.functions.Least(*expressions, **extra)
Accepts a list of at least two field names or expressions 
and returns the greatest/least value

#Exmaple 
comments = Comment.objects.annotate(last_updated=Greatest('modified', 'blog__modified'))
annotated_comment = comments.get()


#Length (can be used as Transform, register at first)
class django.db.models.functions.Length(expression, **extra)
Accepts a single text field or expression and returns the number of characters the value has

Author.objects.create(name='Margaret Smith')
author = Author.objects.annotate(
        name_length=Length('name'),
        goes_by_length=Length('goes_by')).get()
print(author.name_length, author.goes_by_length) #(14, None)

#Usage as transform - register at first 

from django.db.models import CharField
from django.db.models.functions import Length
CharField.register_lookup(Length, 'length')
# Get authors whose name is longer than 7 characters
authors = Author.objects.filter(name__length__gt=7)


#Lower, Upper (can be used as Transform, register at first )
class django.db.models.functions.Lower(expression, **extra)
class django.db.models.functions.Upper(expression, **extra)

from django.db.models.functions import Lower
Author.objects.create(name='Margaret Smith')
author = Author.objects.annotate(name_lower=Lower('name')).get()
print(author.name_lower) #margaret smith


#Substr
class django.db.models.functions.Substr(expression, pos, length=None, **extra)
Returns a substring of length length from the field or expression 
starting at position pos. The position is 1-indexed

# Set the alias to the first 5 characters of the name as lowercase
from django.db.models.functions import Substr, Lower
Author.objects.create(name='Margaret Smith')
Author.objects.update(alias=Lower(Substr('name', 1, 5))) #1
print(Author.objects.get(name='Margaret Smith').alias) #marga

#Now (Django 1.9.)
class django.db.models.functions.Now
Returns the database server’s current date and time when the query is executed, 
typically using the SQL CURRENT_TIMESTAMP.

from django.db.models.functions import Now
Article.objects.filter(published__lte=Now()) #<QuerySet [<Article: How to Django>]>


#Date Functions(Django 1.10)
#Extract (can be used as Transform, no need to register)
class django.db.models.functions.Extract(expression, lookup_name=None, tzinfo=None, **extra)
Extracts a component of a date as a number

class django.db.models.functions.ExtractYear(expression, tzinfo=None, **extra)
lookup_name = 'year'

class django.db.models.functions.ExtractMonth(expression, tzinfo=None, **extra)
ookup_name = 'month'

class django.db.models.functions.ExtractDay(expression, tzinfo=None, **extra)
lookup_name = 'day'

class django.db.models.functions.ExtractWeekDay(expression, tzinfo=None, **extra)
lookup_name = 'week_day'

Use ExtractYear(...) rather than Extract(..., lookup_name='year').
Each class is also a Transform registered on DateField and DateTimeField 
as __(lookup_name), e.g. __year.

#datetime 
class django.db.models.functions.ExtractHour(expression, tzinfo=None, **extra)
lookup_name = 'hour'

class django.db.models.functions.ExtractMinute(expression, tzinfo=None, **extra)
lookup_name = 'minute'

class django.db.models.functions.ExtractSecond(expression, tzinfo=None, **extra)
lookup_name = 'second'

These are logically equivalent to Extract('datetime_field', lookup_name). 
Each class is also a Transform registered on DateTimeField as __(lookup_name), 
e.g. __minute.


#Given the datetime 2015-06-15 23:30:01.000321+00:00, the built-in lookup_names return:
•“year”: 2015
•“month”: 6
•“day”: 15
•“week_day”: 2
•“hour”: 23
•“minute”: 30
•“second”: 1


#Example 
from datetime import datetime
from django.utils import timezone
from django.db.models.functions import (
    ExtractYear, ExtractMonth, ExtractDay, ExtractWeekDay
    )
start_2015 = datetime(2015, 6, 15, 23, 30, 1, tzinfo=timezone.utc)
end_2015 = datetime(2015, 6, 16, 13, 11, 27, tzinfo=timezone.utc)
Experiment.objects.create(
        start_datetime=start_2015, start_date=start_2015.date(),
        end_datetime=end_2015, end_date=end_2015.date())
Experiment.objects.annotate(
        year=ExtractYear('start_date'),
        month=ExtractMonth('start_date'),
        day=ExtractDay('start_date'),
        weekday=ExtractWeekDay('start_date'),
        ).values('year', 'month', 'day', 'weekday').get(
            end_date__year=ExtractYear('start_date'),
            )
#{'year': 2015, 'month': 6, 'day': 15, 'weekday': 2}


#Trunc(can be used as Transform, but register at first, but Note!!)
class django.db.models.functions.Trunc(expression, kind, output_field=None, tzinfo=None, **extra)
Truncates a date up to a significant component
use TruncYear(...) rather than Trunc(..., kind='year').

#DateTimeField truncation
class django.db.models.functions.TruncDate(expression, **extra)
kind = 'date'
class django.db.models.functions.TruncDay(expression, output_field=None, tzinfo=None, **extra)
kind = 'day'
class django.db.models.functions.TruncHour(expression, output_field=None, tzinfo=None, **extra)
kind = 'hour'
class django.db.models.functions.TruncMinute(expression, output_field=None, tzinfo=None, **extra)
kind = 'minute'
class django.db.models.functions.TruncSecond(expression, output_field=None, tzinfo=None, **extra)
kind = 'second'

#DateField truncation
class django.db.models.functions.TruncYear(expression, output_field=None, tzinfo=None, **extra)
kind = 'year'
class django.db.models.functions.TruncMonth(expression, output_field=None, tzinfo=None, **extra)
kind = 'month'

The subclasses are all defined as transforms, 
but they aren’t registered with any fields, Register it 
but names are already reserved by the Extract subclasses.


#Given the datetime 2015-06-15 14:30:50.000321+00:00, the built-in kinds return:
•“year”: 2015-01-01 00:00:00+00:00
•“month”: 2015-06-01 00:00:00+00:00
•“day”: 2015-06-15 00:00:00+00:00
•“hour”: 2015-06-15 14:00:00+00:00
•“minute”: 2015-06-15 14:30:00+00:00
•“second”: 2015-06-15 14:30:50+00:00

#Usage 
from datetime import datetime
from django.db.models import Count, DateTimeField
from django.db.models.functions import Trunc
Experiment.objects.create(start_datetime=datetime(2015, 6, 15, 14, 30, 50, 321))
Experiment.objects.create(start_datetime=datetime(2015, 6, 15, 14, 40, 2, 123))
Experiment.objects.create(start_datetime=datetime(2015, 12, 25, 10, 5, 27, 999))
experiments_per_day = Experiment.objects.annotate(
    start_day=Trunc('start_datetime', 'day', output_field=DateTimeField())
    ).values('start_day').annotate(experiments=Count('id'))
for exp in experiments_per_day:
        print(exp['start_day'], exp['experiments'])



###Djano- List of conditional expression 
#Conditional expressions can be used in annotations, aggregations, lookups, and updates.
#They can also be combined and nested with other expressions. 

#Examples 
from django.db import models

class Client(models.Model):
    REGULAR = 'R'
    GOLD = 'G'
    PLATINUM = 'P'
    ACCOUNT_TYPE_CHOICES = (
        (REGULAR, 'Regular'),
        (GOLD, 'Gold'),
        (PLATINUM, 'Platinum'),
    )
    name = models.CharField(max_length=50)
    registered_on = models.DateField()
    account_type = models.CharField(
        max_length=1, choices=ACCOUNT_TYPE_CHOICES, default=REGULAR, )

#When
class When(condition=None, then=None, **lookups)

A When() object is used to encapsulate a 'condition' 
and its result 'then' for use in the conditional expression
The condition can be specified using field lookups or Q objects

from django.db.models import When, F, Q
# String arguments refer to fields; the following two examples are equivalent:
When(account_type=Client.GOLD, then='name')
When(account_type=Client.GOLD, then=F('name'))
# You can use field lookups in the condition
from datetime import date
When(registered_on__gt=date(2014, 1, 1),
        registered_on__lt=date(2015, 1, 1),
        then='account_type')
# Complex conditions can be created using Q objects
When(Q(name__startswith="John") | Q(name__startswith="Paul"),
        then='name')



#Case
class Case(*cases, **extra)
cases are When() objects 

#Example 
from datetime import date, timedelta
from django.db.models import CharField, Case, Value, When
Client.objects.create(
        name='Jane Doe',
        account_type=Client.REGULAR,
        registered_on=date.today() - timedelta(days=36))
Client.objects.create(
        name='James Smith',
        account_type=Client.GOLD,
        registered_on=date.today() - timedelta(days=5))
Client.objects.create(
        name='Jack Black',
        account_type=Client.PLATINUM,
        registered_on=date.today() - timedelta(days=10 * 365))
# Get the discount for each Client based on the account type
Client.objects.annotate(
        discount=Case(
            When(account_type=Client.GOLD, then=Value('5%')),
            When(account_type=Client.PLATINUM, then=Value('10%')),
            default=Value('0%'),
            output_field=CharField(),
        ),
    ).values_list('name', 'discount')
#[('Jane Doe', '0%'), ('James Smith', '5%'), ('Jack Black', '10%')]


#Usage - Conditional update

#to change the account_type for our clients to match their registration dates. 


a_month_ago = date.today() - timedelta(days=30)
a_year_ago = date.today() - timedelta(days=365)
# Update the account_type for each Client from the registration date
Client.objects.update(
        account_type=Case(
            When(registered_on__lte=a_year_ago,
                then=Value(Client.PLATINUM)),
            When(registered_on__lte=a_month_ago,
                then=Value(Client.GOLD)),
            default=Value(Client.REGULAR)
        ),
    )
Client.objects.values_list('name', 'account_type') #[('Jane Doe', 'G'), ('James Smith', 'R'), ('Jack Black', 'P')]



#Usage - Conditional aggregation

#to find out how many clients there are for each account_type?

# Create some more Clients first so we can have something to count
Client.objects.create(
        name='Jean Grey',
        account_type=Client.REGULAR,
        registered_on=date.today())
Client.objects.create(
        name='James Bond',
        account_type=Client.PLATINUM,
        registered_on=date.today())
Client.objects.create(
        name='Jane Porter',
        account_type=Client.PLATINUM,
        registered_on=date.today())
# Get counts for each value of account_type
from django.db.models import IntegerField, Sum
Client.objects.aggregate(
        regular=Sum(
            Case(When(account_type=Client.REGULAR, then=1),
                output_field=IntegerField())
        ),
        gold=Sum(
            Case(When(account_type=Client.GOLD, then=1),
                output_field=IntegerField())
        ),
        platinum=Sum(
            Case(When(account_type=Client.PLATINUM, then=1),
                output_field=IntegerField())
        )
    )
#{'regular': 2, 'gold': 1, 'platinum': 3}







###Djano- List of QuerySet methods 

##Methods that return new QuerySets
##filter()
##exclude()
##annotate()
##order_by()
By default, results returned by a QuerySet are ordered by the ordering tuple given 
by the ordering option in the model’s Meta. 
OR override default by 
Entry.objects.filter(pub_date__year=2005).order_by('-pub_date', 'headline')
# ordered by pub_date descending, then by headline ascending.
 
#To use random ordering, use ?
Entry.objects.order_by('?')

#With other related table 
Entry.objects.order_by('blog__name', 'headline')

#To order by a field that is a relation to another model, 
#Django will use the default ordering on the related model, 
#or order by the related model’s primary key if there is no Meta.ordering specified. 

#For example, since the Blog model has no default ordering specified:
Entry.objects.order_by('blog')

#...is identical to:
Entry.objects.order_by('blog__id')


#It is also possible to order a queryset by a related field, 
#without incurring the cost of a JOIN, by referring to the _id of the related field:

# No Join
Entry.objects.order_by('blog_id')

# Join
Entry.objects.order_by('blog__id')


#You can also order by query expressions by calling asc() or desc() on the expression:
Entry.objects.order_by(Coalesce('summary', 'headline').desc())

#You can order by a field converted to lowercase with Lower 
#which will achieve case-consistent ordering:
Entry.objects.order_by(Lower('headline').desc())

#If you don’t want any ordering to be applied to a query, call 
.order_by() 

#Each order_by() call will clear any previous ordering. 
#For example, this query will be ordered by pub_date and not headline:
Entry.objects.order_by('headline').order_by('pub_date')



##reverse()
#Use the reverse() method to reverse the order in which a queryset’s elements are returned. 
#Calling reverse() a second time restores the ordering back to the normal direction.

#To retrieve the “last” five items in a queryset, you could do this:
my_queryset.reverse()[:5]



##distinct(*fields)  #*fields are applicable only On PostgreSQL 
Returns a new QuerySet that uses SELECT DISTINCT in its SQL query. 
This eliminates duplicate rows from the query results.

#if your query spans multiple tables, it’s possible to get duplicate results 
#when a QuerySet is evaluated. use distinct().

#order_by() with distinct() - be careful
Any fields used in an order_by() (or default ordering via Meta) 
are included in the SQL SELECT columns. This might affect distinct.

if you use a values() query to restrict the columns selected, 
the columns used in any order_by() (or default model ordering) will still be involved 
and may affect uniqueness of the results.


#When you specify field names, you must provide an order_by() in the QuerySet, 
#and the fields in order_by() must start with the fields in distinct(), in the same order.

#Examples (those after the first will only work on PostgreSQL):
Author.objects.distinct() #[...]  
Entry.objects.order_by('pub_date').distinct('pub_date') #[...]



##values(*fields)
Returns a QuerySet that returns dictionaries, rather than model instances, 
when used as an iterable

# This list contains a Blog object.
Blog.objects.filter(name__startswith='Beatles') #<QuerySet [<Blog: Beatles Blog>]>

# This list contains a dictionary.
Blog.objects.filter(name__startswith='Beatles').values() #<QuerySet [{'id': 1, 'name': 'Beatles Blog', 'tagline': 'All the latest Beatles news.'}]>

Blog.objects.values() #<QuerySet [{'id': 1, 'name': 'Beatles Blog', 'tagline': 'All the latest Beatles news.'}]>
Blog.objects.values('id', 'name') #<QuerySet [{'id': 1, 'name': 'Beatles Blog'}]>

#If you have a field called foo that is a ForeignKey, 
#the default values() call will return a dictionary key called foo_id
Entry.objects.values() #<QuerySet [{'blog_id': 1, 'headline': 'First Entry', ...}, ...]>

Entry.objects.values('blog') #<QuerySet [{'blog': 1}, ...]>
Entry.objects.values('blog_id') #<QuerySet [{'blog_id': 1}, ...]>


#note that you can call filter(), order_by(), etc. after the values() call, 
#that means that these two calls are identical:
Blog.objects.values().order_by('id')
Blog.objects.order_by('id').values()


#You can also refer to fields on related models with reverse relations 
#through OneToOneField, ForeignKey and ManyToManyField attributes:

Blog.objects.values('name', 'entry__headline')
#<QuerySet [{'name': 'My blog', 'entry__headline': 'An entry'},{'name': 'My blog', 'entry__headline': 'Another entry'}, ...]>




##values_list(*fields, flat=False)¶
This is similar to values() except that instead of returning dictionaries, 
it returns tuples when iterated over. 

Entry.objects.values_list('id', 'headline') #[(1, 'First entry'), ...]

#with flat 
Entry.objects.values_list('id').order_by('id') #[(1,), (2,), (3,), ...]
Entry.objects.values_list('id', flat=True).order_by('id') #[1, 2, 3, ...]


#It is an error to pass in flat when there is more than one field.
#values_list() returns all the fields in the model, in the order they were declared.

#A common need is to get a specific field value of a certain model instance. 
#To achieve that, use values_list() followed by a get() call:
Entry.objects.values_list('headline', flat=True).get(pk=1) #'First entry'



##dates(field, kind, order='ASC')
#Truncates date only to 'kind'
#kind should be either "year", "month" or "day". 

Entry.objects.dates('pub_date', 'year')     #[datetime.date(2005, 1, 1)]
Entry.objects.dates('pub_date', 'month')    #[datetime.date(2005, 2, 1), datetime.date(2005, 3, 1)]
Entry.objects.dates('pub_date', 'day')      #[datetime.date(2005, 2, 20), datetime.date(2005, 3, 20)]
Entry.objects.dates('pub_date', 'day', order='DESC') #[datetime.date(2005, 3, 20), datetime.date(2005, 2, 20)]
Entry.objects.filter(headline__contains='Lennon').dates('pub_date', 'day') #[datetime.date(2005, 3, 20)]



##datetimes(field_name, kind, order='ASC', tzinfo=None)
#Truncates datetime only to 'kind'
#kind should be either "year", "month", "day", "hour", "minute" or "second


##none()
Calling none() will create a queryset that never returns any objects 
and no query will be executed when accessing the results. 

Entry.objects.none()  #<QuerySet []>


##all()
##select_related(*fields)
Returns a QuerySet that will “follow” foreign-key relationships, 
selecting additional related-object data when it executes its query. 
This is a performance booster 


# Hits the database.
e = Entry.objects.get(id=5)
# Hits the database again to get the related Blog object.
b = e.blog

#And here’s select_related lookup:
# Hits the database.
e = Entry.objects.select_related('blog').get(id=5)
# Doesn't hit the database, because e.blog has been prepopulated
# in the previous query.
b = e.blog

#The order of filter() and select_related() chaining isn’t important. 
#These querysets are equivalent:
Entry.objects.filter(pub_date__gt=timezone.now()).select_related('blog')
Entry.objects.select_related('blog').filter(pub_date__gt=timezone.now())


##prefetch_related(*lookups)
Returns a QuerySet that will automatically retrieve, in a single batch, 
related objects for each of the specified lookups.


##extra(select=None, where=None, params=None, tables=None, order_by=None, select_params=None)
#To specifify SQL select, where, table directly 
#Specify one or more of params, select, where or tables. 

#each Entry object will have an extra attribute, is_recent, a boolean representing 
Entry.objects.extra(select={'is_recent': "pub_date > '2006-01-01'"})


Blog.objects.extra(
    select={
        'entry_count': 'SELECT COUNT(*) FROM blog_entry WHERE blog_entry.blog_id = blog_blog.id'
    },
)

#To pass parameters to 'select=', use  select_params parameter
#you should use a collections.OrderedDict for the select value

Blog.objects.extra(
    select=OrderedDict([('a', '%s'), ('b', '%s')]),
    select_params=('one', 'two'))

#To define explicit SQL WHERE clauses 
Entry.objects.extra(where=["foo='a' OR bar = 'a'", "baz = 'a'"])

#to order the resulting queryset 
q = Entry.objects.extra(select={'is_recent': "pub_date > '2006-01-01'"})
q = q.extra(order_by = ['-is_recent'])

#The params argument is a list of any extra parameters to be substituted.

Entry.objects.extra(where=['headline=%s'], params=['Lennon'])




##defer(*fields)
A queryset that has deferred fields will still return model instances. 
Each deferred field will be retrieved from the database if you access that field 
(one at a time, not all the deferred fields at once).

Entry.objects.defer("headline", "body")


##only(*fields)
You call it with the fields that should not be deferred when retrieving a model

Person.objects.defer("age", "biography")
Person.objects.only("name")


##using(alias)
This method is for controlling which database the QuerySet will be evaluated 
against if you are using more than one database. 

# queries the database with the 'default' alias.
Entry.objects.all()
# queries the database with the 'backup' alias
Entry.objects.using('backup')



##select_for_update()
##raw()

##Methods that do not return QuerySets 
##get(**kwargs)
##create(**kwargs)
#for creating an object and saving it all in one step
p = Person.objects.create(first_name="Bruce", last_name="Springsteen")


##get_or_create(defaults=None, **kwargs)¶
for looking up an object with the given kwargs or creating one if necessary.
Returns a tuple of (object, created), where object is the retrieved or created object 
and created is a boolean specifying whether a new object was created.

#For example 
try:
    obj = Person.objects.get(first_name='John', last_name='Lennon')
except Person.DoesNotExist:
    obj = Person(first_name='John', last_name='Lennon', birthday=date(1940, 10, 9))
    obj.save()
#OR
obj, created = Person.objects.get_or_create(
    first_name='John',
    last_name='Lennon',
    defaults={'birthday': date(1940, 10, 9)},
)

#For MySQL,  use the READ COMMITTED isolation level rather than REPEATABLE READ (the default), otherwise you may see cases where get_or_create will raise an IntegrityError but the object won’t appear in a subsequent get() call.

##update_or_create()
Returns a tuple of (object, created), where object is the created or updated object 
and created is a boolean specifying whether a new object was created

#For example:
try:
    obj = Person.objects.get(first_name='John', last_name='Lennon')
    for key, value in updated_values.iteritems():
        setattr(obj, key, value)
    obj.save()
except Person.DoesNotExist:
    updated_values.update({'first_name': 'John', 'last_name': 'Lennon'})
    obj = Person(**updated_values)
    obj.save()


#OR
obj, created = Person.objects.update_or_create(
    first_name='John', last_name='Lennon', defaults=updated_values)


##bulk_create(objs, batch_size=None)
This method inserts the provided list of objects into the database 
in an efficient manner (generally only 1 query, no matter how many objects there are):


Entry.objects.bulk_create([
    Entry(headline="Django 1.0 Released"),
    Entry(headline="Django 1.1 Announced"),
    Entry(headline="Breaking: Django is awesome")
    ])


#caveats 
•The model’s save() method will not be called, and the pre_save and post_save signals will not be sent.
•It does not work with child models in a multi-table inheritance scenario.
•If the model’s primary key is an AutoField it does not retrieve and set the primary key attribute, as save() does, unless the database backend supports it (currently PostgreSQL).
•It does not work with many-to-many relationships.


##count()
##in_bulk(id_list=None)
Takes a list of primary-key values and returns a dictionary 
mapping each primary-key value to an instance of the object with the given ID. 
If a list isn’t provided, all objects in the queryset are returned.

Blog.objects.in_bulk([1])       #{1: <Blog: Beatles Blog>}
Blog.objects.in_bulk([1, 2])    #{1: <Blog: Beatles Blog>, 2: <Blog: Cheddar Talk>}
Blog.objects.in_bulk([])        #{}
Blog.objects.in_bulk()          #{1: <Blog: Beatles Blog>, 2: <Blog: Cheddar Talk>, 3: <Blog: Django Weblog>}


##iterator()
Evaluates the QuerySet (by performing the query) and returns an iterator over the results
A QuerySet typically caches its results internally 
so that repeated evaluations do not result in additional queries. 
In contrast, iterator() will read results directly, without doing any caching 
at the QuerySet 


##latest(field_name=None) , earliest(ield_name=None)
Returns the latest/earliest object in the table, by date, 
using the field_name provided as the date field.

Entry.objects.latest('pub_date')





##first() last()
Returns the first/last object matched by the queryset, 
or None if there is no matching object. 
If the QuerySet has no ordering defined, then the queryset is automatically ordered by the primary key.

p = Article.objects.order_by('title', 'pub_date').first()



##aggregate()
##exists()
Returns True if the QuerySet contains any results, and False if not

entry = Entry.objects.get(pk=123)
if some_queryset.filter(pk=entry.pk).exists():
    print("Entry contained in queryset")


##update(**kwargs)
Performs an SQL update query for the specified fields, 
and returns the number of rows matched (which may not be equal to the number of rows updated if some rows already have the new value).

#to turn comments off for all blog entries published in 2010, you could do this:
Entry.objects.filter(pub_date__year=2010).update(comments_on=False)
Entry.objects.filter(pub_date__year=2010).update(comments_on=False, headline='This is old')

#QuerySet that is updated is that it can only update columns in the model’s main table, 
#not on related models. You can’t do this, for example:
Entry.objects.update(blog__name='foo') # Won't work!

#Filtering based on related fields is still possible, though:
Entry.objects.filter(blog__id=1).update(comments_on=True)


#You cannot call update() on a QuerySet that has had a slice taken 
#or can otherwise no longer be filtered.

#Using update() also prevents a race condition wherein something might 
#change in your database in the short period of time between loading the object and calling save().



##delete()
Performs an SQL delete query on all rows in the QuerySet 
and returns the number of objects deleted and a dictionary with the number of deletions per object type.

The delete() is applied instantly. 
You cannot call delete() on a QuerySet that has had a slice taken or can otherwise no longer be filtered.

#to delete all the entries in a particular blog:
b = Blog.objects.get(pk=1)

# Delete all the entries belonging to this Blog.
Entry.objects.filter(blog=b).delete() #(4, {'weblog.Entry': 2, 'weblog.Entry_authors': 2})


##as_manager()




###Django -List of Query Expression API 
#Set of methods  that classes define to be usable in query expressions to translate 
#themselves into SQL expressions
#Direct field references, aggregates, and Transform are examples that follow this API
#Note Lookup has few extra functionalities 
#To be complaint of Query expression API, class must implement 
as_sql(self, compiler, connection)¶
get_lookup(lookup_name)
get_transform(transform_name)
output_field
....

##Built in Query Expression API compliant classes - module- django.db.models
F()                                                 for accessing Field in Query expression 
Aggregate(expression, output_field=None, **extra)   for Aggregation 
Value(value, output_field=None)                     Simple value eg 1 is converted to Value(1)

#Other classes 
#RawSQL(sql, params, output_field=None)              To use raw sql 
#DOn't use as might not be portable across DB engines
from django.db.models.expressions import RawSQL
queryset.annotate(val=RawSQL("select col from sometable where othercol = %s", (someparam,)))


#Func() expressions
Func() expressions are the base type of all expressions 
that involve database functions like COALESCE and LOWER, or aggregates like SUM.
Note Transform is subclass of Func 

#Usage 
from django.db.models import Func, F
queryset.annotate(field_lower=Func(F('field'), function='LOWER'))


# can be used to build a library of database functions:
class Lower(Func):
    function = 'LOWER'

queryset.annotate(field_lower=Lower('field'))

##Supported arithmetic
Django supports addition, subtraction, multiplication, division, modulo arithmetic, 
and the power operator on query expressions, using Python constants, variables, 
and even other expressions.

#Example 
from django.db.models import F, Count
from django.db.models.functions import Length, Upper, Value

# Find companies that have more employees than chairs.
Company.objects.filter(num_employees__gt=F('num_chairs'))

# Find companies that have at least twice as many employees
# as chairs. Both the querysets below are equivalent.
Company.objects.filter(num_employees__gt=F('num_chairs') * 2)
Company.objects.filter( num_employees__gt=F('num_chairs') + F('num_chairs'))

# How many chairs are needed for each company to seat all employees?
company = Company.objects.filter(
            num_employees__gt=F('num_chairs')).annotate(
            chairs_needed=F('num_employees') - F('num_chairs')).first()
company.num_employees #120
company.num_chairs #50
company.chairs_needed #70

# Create a new company using expressions.
company = Company.objects.create(name='Google', ticker=Upper(Value('goog')))
# Be sure to refresh it if you need to access the field.
company.refresh_from_db()
company.ticker  #'GOOG'

# Annotate models with an aggregated value. Both forms
# below are equivalent.
Company.objects.annotate(num_products=Count('products'))
Company.objects.annotate(num_products=Count(F('products')))

# Aggregates can contain complex computations also
Company.objects.annotate(num_offerings=Count(F('products') + F('services')))

# Expressions can also be used in order_by()
Company.objects.order_by(Length('name').asc())
Company.objects.order_by(Length('name').desc())





###Django- How to write custom Lookup and Transformer 

#custom Lookup  - ne 

Author.objects.filter(name__ne='Jack') #in SQL "author"."name" <> 'Jack'


#STEP-1:Lookup implementation , Lookup means  <lhs>__<lookup_name>=<rhs>.

from django.db.models import Lookup

class NotEqual(Lookup):
    lookup_name = 'ne'  #The name of this lookup, used to identify it on parsing query expressions

    #Responsible for producing the query string and parameters for the expression
    #Part of The Query Expression API
    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection) #Returns a tuple (lhs_string, lhs_params)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = lhs_params + rhs_params
        return '%s <> %s' % (lhs, rhs), params


#STEP-2: registration 

from django.db.models.fields import Field
Field.register_lookup(NotEqual)

#OR using decorator 

from django.db.models.fields import Field

@Field.register_lookup
class NotEqualLookup(Lookup):
    # ...


#We can now use foo__ne for any field foo. 

##Custom Transformer - abs  - absolute value 
Experiment.objects.filter(change__abs=27))
Experiment.objects.filter(change__abs__lt=27))

#STEP-1: Implementation - use the SQL function ABS() 
from django.db.models import Transform

class AbsoluteValue(Transform):
    lookup_name = 'abs'
    function = 'ABS'


#STEP-2: register it for IntegerField:


from django.db.models import IntegerField
IntegerField.register_lookup(AbsoluteValue)


#By using Transform instead of Lookup it means we are able to chain further lookups afterwards.

Experiment.objects.filter(change__abs__lt=27) 
#-> SELECT ... WHERE ABS("experiments"."change") < 27


#When looking for which lookups are allowable after the Transform has been applied, 
#Django uses the output_field attribute
#Above can also be written as below (not required because Transform has not changed output_field)

#Example, when ABS can be applied on FloatField as well 

from django.db.models import FloatField, Transform

class AbsoluteValue(Transform):
    lookup_name = 'abs'
    function = 'ABS'

    @property
    def output_field(self):
        return FloatField()

#Writing an efficient abs__lt lookup
#Suppose we want SQL to be like 
Experiment.objects.filter(change__abs__lt=27) 
#->SELECT .. WHERE "experiments"."change" < 27 AND "experiments"."change" > -27

from django.db.models import Lookup

class AbsoluteValueLessThan(Lookup):
    lookup_name = 'lt'

    def as_sql(self, compiler, connection):
        lhs, lhs_params = compiler.compile(self.lhs.lhs)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = lhs_params + rhs_params + lhs_params + rhs_params
        return '%s < %s AND %s > -%s' % (lhs, rhs, lhs, rhs), params

AbsoluteValue.register_lookup(AbsoluteValueLessThan)

# bilateral transformer example - use bilateral = True
The AbsoluteValue example we discussed previously is a transformation 
which applies to the left-hand side of the lookup. 
There may be some cases where you want the transformation to be applied 
to both the left-hand side and the right-hand side.

#Implementation 
from django.db.models import Transform

class UpperCase(Transform):
    lookup_name = 'upper'
    function = 'UPPER'
    bilateral = True


from django.db.models import CharField, TextField
CharField.register_lookup(UpperCase)
TextField.register_lookup(UpperCase)


Author.objects.filter(name__upper="doe") 
#-> SELECT ... WHERE UPPER("author"."name") = UPPER('doe')












###Django - AppConfig 

#Django contains a registry of installed applications 
#that stores configuration and provides introspection. 
#It also maintains a list of available models.


from django.apps import apps
apps.get_app_config('admin').verbose_name  #'Admin'

##Other methods of AppConfig 
#Configurable attributes
AppConfig.name          Full Python path to the application, e.g. 'django.contrib.admin'
AppConfig.label         Short name for the application, e.g. 'admin'
AppConfig.verbose_name  Human-readable name for the application, e.g. “Administration”.
                        This attribute defaults to label.title().
AppConfig.path          Filesystem path to the application directory, e.g. '/usr/lib/python3.4/dist-packages/django/contrib/admin'.

#Read-only attributes
AppConfig.module        Root module for the application, e.g. <module 'django.contrib.admin' from 'django/contrib/admin/__init__.pyc'>.
AppConfig.models_module Module containing the models, e.g. <module 'django.contrib.admin.models' from 'django/contrib/admin/models.pyc'>.

#Methods
AppConfig.get_models()  Returns an iterable of Model classes for this application.
AppConfig.get_model(model_name) Returns the Model with the given model_name
AppConfig.ready()       Subclasses can override this method to perform initialization tasks such as registering signals. It is called as soon as the registry is fully populated.

##Projects and applications
The term project describes a Django web application
created by 'django-admin startproject mysite' 
creates mysite package e with settings.py, urls.py, and wsgi.py.

A project’s root directory (the one that contains manage.py) 
is the container for all of a project’s applications which aren’t installed separately.

application 'django-admin startapp myapp'' describes a Python package 
that provides some set of features(having models, view etc)
Applications may be reused in various projects.



#Older way of configuring Application (subclassing AppConfig)
#when INSTALLED_APPS contains 'appname'
#Django checks for a 'default_app_config' variable in that module.
#If there is no default_app_config, Django uses the base AppConfig class.

#Example 
# rock_n_roll/apps.py

from django.apps import AppConfig

class RockNRollConfig(AppConfig):
    name = 'rock_n_roll'
    verbose_name = "Rock ’n’ roll"



# rock_n_roll/__init__.py

default_app_config = 'rock_n_roll.apps.RockNRollConfig'


##Newer way Configuring applications - by subclassing AppConfig
#and put the dotted path to that subclass in INSTALLED_APPS.

#Example 
# anthology/apps.py

from rock_n_roll.apps import RockNRollConfig

class JazzManoucheConfig(RockNRollConfig):
    verbose_name = "Jazz Manouche"

# anthology/settings.py

INSTALLED_APPS = [
    'anthology.apps.JazzManoucheConfig',
    # ...
]







f
###Django - URLConf - URL root




#How Django processes a request

1. Django determines the root URLconf module to use, settings.ROOT_URLCONF
   or 	incoming HttpRequest.urlconf (set by middleware request processing)

2. Django loads that Python module and urlpatterns variable 

3. Django runs through each URL pattern, in order, 
   and stops at the first one that matches the requested URL.
   
4. Once one of the regexes matches, Django imports and calls the given view(a method or class with __call__)
   The view gets passed the following arguments:
     -An instance of HttpRequest.
     -any  matched parameter from url either possitional or if named group, keyword way 

5. If no regex matches, or if an exception is raised, raise error handling view

#Example without named group

from django.conf.urls import patterns, url

from . import views

urlpatterns = [
    url(r'^articles/2003/$', views.special_case_2003),
    url(r'^articles/(\d{4})/$', views.year_archive),
    url(r'^articles/(\d{4})/(\d{2})/$', views.month_archive),
    url(r'^articles/(\d{4})/(\d{2})/(\d+)/$', views.article_detail),
]

• A request to /articles/2005/03/ would match the third entry in the list.
  calls views.month_archive(request, '2005', '03').
	
•  /articles/2005/3/ would not match any URL patterns

•  /articles/2003/ would match the first pattern in the list, as in order, it is first 

• /articles/2003 would not match any of these patterns

• /articles/2003/03/03/ would match the final pattern. 
  calls views.article_detail(request, '2003', '03', '03').



#Example of Named groups

from django.conf.urls import patterns, url

from . import views

urlpatterns = [         #earlier, it is patterns('',url(...))
    url(r'^articles/2003/$', views.special_case_2003),
    url(r'^articles/(?P<year>\d{4})/$', views.year_archive),
    url(r'^articles/(?P<year>\d{4})/(?P<month>\d{2})/$', views.month_archive),
    url(r'^articles/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/$', views.article_detail),
]

•  /articles/2005/03/ => views.month_archive(request, year='2005', month='03')
   instead of views.month_archive(request, '2005', '03')
   
• articles/2003/03/03/ => views.article_detail(request, year='2003', month='03', day='03').


##What the URLconf searches against - does not look at request method

 http://www.example.com/myapp/,  URLconf looks myapp/
 http://www.example.com/myapp/?page=3, URLconf looks myapp/
 Hence appropriate Regex is required , 
 Note query parameter can be got from HttpRequest

#Captured arguments are always strings even if you match like (\d+)


##Specifying defaults for view arguments

# URLconf
from django.conf.urls import patterns, url

from . import views

urlpatterns = [
    url(r'^blog/$', views.page),     #same view with default num
    url(r'^blog/page(?P<num>\d+)/$', views.page), #same view 
]

# View (in blog/views.py)
def page(request, num="1"):
    # Output the appropriate page of blog entries, according to num.
    ...


## Error handling - when no match or any exception
# only below in root URLconf is valid 
#values of these variables should be callables 
#The variables are:
•handler404 – default is  django.conf.urls.handler404.
•handler500 – default is django.conf.urls.handler500.
•handler403 – default is django.conf.urls.handler403.
•handler400 – default isdjango.conf.urls.handler400.


##Passing strings instead of callable objects(might be removed in future)
from django.conf.urls import patterns, url

urlpatterns = [
    url(r'^archive/$', 'mysite.views.archive'),  #full python path 
    url(r'^about/$', 'mysite.views.about'),
    url(r'^contact/$', 'mysite.views.contact'),
]


#Django class based views must be imported

from django.conf.urls import patterns, url
from mysite.views import ClassBasedView

urlpatterns = patterns('',
    url(r'^myview/$', ClassBasedView.as_view()),
)

###Including other URLconfs
#inlcude() takes string of full path of another urls
#or takes another array of url()

from django.conf.urls import include, url

urlpatterns = [
    #No trailling $, hence all remaining URL goes to included URLConf
    url(r'^community/', include('django_website.aggregator.urls')),
    url(r'^contact/', include('django_website.contact.urls')),
    
]

##Another options - having patterns in same file 

from django.conf.urls import include, url

from apps.main import views as main_views
from credit import views as credit_views

extra_patterns = [
    url(r'^reports/$', credit_views.report),
    url(r'^reports/(?P<id>[0-9]+)/$', credit_views.report),
    url(r'^charge/$', credit_views.charge),
]
#/credit/reports/ => credit_views.report() 
urlpatterns = [
    url(r'^$', main_views.homepage),
    url(r'^help/', include('apps.help.urls')),
    url(r'^credit/', include(extra_patterns)), #after credit/, remaining URL 
]

#remove redundancy from URLconfs 
#where a single pattern prefix is used repeatedly
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/history/$', views.history),
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/edit/$', views.edit),
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/discuss/$', views.discuss),
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/permissions/$', views.permissions),
]

#OR
urlpatterns = [
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/', include([
        url(r'^history/$', views.history),
        url(r'^edit/$', views.edit),
        url(r'^discuss/$', views.discuss),
        url(r'^permissions/$', views.permissions),
    ])),
]


##An included URLconf receives any captured parameters from parent URLconfs

#the captured "username" variable is passed to the included URLconf
# In settings/urls/main.py
from django.conf.urls import include, url

urlpatterns = [
    url(r'^(?P<username>\w+)/blog/', include('foo.urls.blog')),
]

# In foo/urls/blog.py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.blog.index),   #index gets username keyword 
    url(r'^archive/$', views.blog.archive), #archive gets username keyword 
]



##Nested arguments - possible as in regex 


#Example 
from django.conf.urls import url

urlpatterns = [
    url(r'blog/(page-(\d+)/)?$', blog_articles),                  # bad
    url(r'comments/(?:page-(?P<page_number>\d+)/)?$', comments),  # good
]

#usage 
blog/page-2/ => blog_articles(request, "page-2/", "2") 
comments/page-2/ => comments(request, page_number= "2") #outer is non-capturing argument (?:...)

#Note for URL reversing, blog_articles needs two args, one outer and one inner 



##Passing extra options to view functions - by third arg of url as dict 

from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^blog/(?P<year>[0-9]{4})/$', views.year_archive, {'foo': 'bar'}),
]


/blog/2005/ => views.year_archive(request, year='2005', foo='bar').

##Passing extra options to include() - by third arg of include  as dict

#OPTION-1
# main.py
from django.conf.urls import include, url

urlpatterns = [
    url(r'^blog/', include('inner'), {'blogid': 3}), 
]

# inner.py
from django.conf.urls import url
from mysite import views
#each below line gets {'blogid': 3} from main.py 
urlpatterns = [
    url(r'^archive/$', views.archive),
    url(r'^about/$', views.about),
]


#OR OPTION-2

# main.py
from django.conf.urls import include, url
from mysite import views

urlpatterns = [
    url(r'^blog/', include('inner')),
]

# inner.py
from django.conf.urls import url

urlpatterns = [
    url(r'^archive/$', views.archive, {'blogid': 3}),
    url(r'^about/$', views.about, {'blogid': 3}),
]



##Reverse resolution of URLs (given view name, get URL)
•In templates: Using the url template tag.
•In Python code: Using the reverse() function.
•Handling of URLs of Django model instances: The get_absolute_url() method.

#In order to perform URL reversing, use named URL patterns 
#and Name should be unique 
#Put prefix to URL name to reduce collisions of names 

#Example 

from django.conf.urls import url

from . import views

urlpatterns = [
    #...
    url(r'^articles/([0-9]{4})/$', views.year_archive, name='news-year-archive'),
    #...
]

#Template code , 
#note news-year-archive takes one arg because of captured group

<a href="{% url 'news-year-archive' 2012 %}">2012 Archive</a>
{# Or with the year in a template context variable: #}
<ul>
{% for yearvar in year_list %}
<li><a href="{% url 'news-year-archive' yearvar %}">{{ yearvar }} Archive</a></li>
{% endfor %}
</ul>


#Or in Python code:

from django.urls import reverse
from django.http import HttpResponseRedirect

def redirect_to_year(request):
    # ...
    year = 2006
    # ...
    return HttpResponseRedirect(reverse('news-year-archive', args=(year,)))


##URL namespaces
URL namespaces allow you to uniquely reverse named URL patterns 
even if different applications use the same URL names

#A URL namespace comes in two parts, both of which are strings:
application namespace - comes  urls.py  app_name, all instances share this  
instance namespace    - a specific instance of app , default one is same as app_name 
                        Can be set via namespace agument of include()
                        Note instance means here same view is used with 
                        two URL root in URLConf
                        
#Namespaced URLs are specified using the ':' operator. 
'admin:index' indicates a namespace of 'admin', and a named URL of 'index'.

#Namespaces can also be nested. 
'sports:polls:index' -> 'index' in the namespace 'polls' 
                       that is itself defined within the top-level namespace 'sports'.


##Example
#same polls.urls rooted at two URL differentiated by namespace(ie instance part of namespace)

#urls.py

from django.conf.urls import include, url

urlpatterns = [
    url(r'^author-polls/', include('polls.urls', namespace='author-polls')),
    url(r'^publisher-polls/', include('polls.urls', namespace='publisher-polls')),
]

#polls/urls.py

from django.conf.urls import url

from . import views

app_name = 'polls'   #application part of namespace 
urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'), #name for named URL 
    url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),
    ...
]


#Using this setup, the following lookups are possible:

•If 'author-polls' instance (ie root URL of '/author-polls') is active 
'polls:index' will resolve to the index page of the 'author-polls' ie "/author-polls/".
#for example in 
reverse('polls:index', current_app=self.request.resolver_match.namespace)
#or  in the template:
{% url 'polls:index' %}

•If there is no current instance (ie no root URL as given above)-
'polls:index' will resolve to the last registered instance of polls ie 'publisher-polls'

•'author-polls:index' will always resolve to the index page of 'author-polls'
 (and likewise for 'publisher-polls') .




##Application namespaces in  included URLconfs
#OPTION-1 : set an app_name attribute

#polls/urls.py

from django.conf.urls import url

from . import views

app_name = 'polls'  #<--- Application namespaces
urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),
    ...
]

#urls.py

from django.conf.urls import include, url

urlpatterns = [
    url(r'^polls/', include('polls.urls')),
]

#OPTION-2,
In include() , pass a 2-tuple containing:
(<list of url() instances>, <application namespace>)

from django.conf.urls import include, url

from . import views

polls_patterns = ([
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),
], 'polls') #<-- Application namespaces

urlpatterns = [
    url(r'^polls/', include(polls_patterns)),
]

#In <Django 1.9 
specify both the application namespace and the instance namespace in a single place, 
either by passing them as parameters to include() 
or by including a 3-tuple containing  in urlpatterns 
(<list of url() instances>, <application namespace>, <instance namespace>).













###Django - Views
https://docs.djangoproject.com/en/1.10/topics/http/views/
#Writing views


from django.http import HttpResponse
import datetime

def current_datetime(request):  #takes request 
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now
    return HttpResponse(html)  #must return response


#Django’s Time Zone - settings.TIME_ZONE

#Django includes a TIME_ZONE setting that defaults to America/Chicago. 
#change it in your settings file.




#Returning errors - subclasses of HttpResponse

from django.http import HttpResponse, HttpResponseNotFound

def my_view(request):
    # ...
    if foo:
        return HttpResponseNotFound('<h1>Page not found</h1>')
    else:
        return HttpResponse('<h1>Page was found</h1>')

#or pas status code in ctor of HttpResponse

from django.http import HttpResponse

def my_view(request):
    # ...

    # Return a "created" (201) response code.
    return HttpResponse(status=201)


#Or using - The Http404 exception -class django.http.Http404
#no need to return html code, but simple text message

from django.http import Http404
from django.shortcuts import render_to_response
from polls.models import Poll

def detail(request, poll_id):
    try:
        p = Poll.objects.get(pk=poll_id)
    except Poll.DoesNotExist:
        raise Http404("Poll does not exist")
    return render_to_response('polls/detail.html', {'poll': p})

#To modify, thml page, Modify 404.html and put it in the top level of your template tree.

##OR put you error views  in URLconf

#The page_not_found() view is overridden by handler404:
handler404 = 'mysite.views.my_custom_page_not_found_view'

#The server_error() view is overridden by handler500:
handler500 = 'mysite.views.my_custom_error_view'

#The permission_denied() view is overridden by handler403:
handler403 = 'mysite.views.my_custom_permission_denied_view'

#The bad_request() view is overridden by handler400:
handler400 = 'mysite.views.my_custom_bad_request_view'





###Django - view as decorator 
https://docs.djangoproject.com/en/1.10/topics/http/decorators/


from django.views.decorators.http import require_http_methods

@require_http_methods(["GET", "POST"])  #Http methods in upper case
def my_view(request):
    # I can assume now that only GET or POST requests make it this far
    # ...
    pass

#Other decorators 
require_GET()       Decorator to require that a view only accepts the GET method.
require_POST()      Decorator to require that a view only accepts the POST method.
require_safe()      Decorator to require that a view only accepts the GET and HEAD methods












###Django - Django shortcut functions
https://docs.djangoproject.com/en/1.10/topics/http/shortcuts/

##module -  django.shortcuts 
#render()
render(request, template_name, context=None, content_type=None, status=None, 
    using=None)[source]
Combines a given template with a given context dictionary and returns an HttpResponse object with that rendered text.
context A dictionary of values to add to the template context. 


#renders the template myapp/index.html with the MIME type application/xhtml+xml:


from django.shortcuts import render

def my_view(request):
    # View code here...
    return render(request, 'myapp/index.html', {
        'foo': 'bar',
    }, content_type='application/xhtml+xml')


#is equivalent to:


from django.http import HttpResponse
from django.template import loader

def my_view(request):
    # View code here...
    t = loader.get_template('myapp/index.html')
    c = {'foo': 'bar'}
    return HttpResponse(t.render(c, request), content_type='application/xhtml+xml')


#render_to_response()  - deprecated 
render_to_response(template_name, context=None, content_type=None, status=None, 
    using=None)


#redirect()
redirect(to, permanent=False, *args, **kwargs)
Returns an HttpResponseRedirect to the appropriate URL for the arguments passed.

#to could be 
•A model                    : the model’s get_absolute_url() function will be called.
•A view name,with arguments : reverse() will be used to reverse-resolve the name.
•An absolute or relative URL, which will be used as-is for the redirect location.

#By default issues a temporary redirect; 
#pass permanent=True to issue a permanent redirect.

#Examples 

from django.shortcuts import redirect

def my_view(request):
    ...
    object = MyModel.objects.get(...)
    return redirect(object)    #get_absolute_url() is called 


#or 

def my_view(request):
    ...
    return redirect('some-view-name', foo='bar')  #reverse would be called 


#Or 


def my_view(request):
    ...
    return redirect('/some/url/')


#Or 


def my_view(request):
    ...
    return redirect('https://example.com/')




#get_object_or_404()
get_object_or_404(klass, *args, **kwargs)
Calls get() on a given model manager, klass
but it raises Http404 instead of the model’s DoesNotExist exception.
As with get(), a MultipleObjectsReturned exception will be raised if more than one object is found.

#Example 
from django.shortcuts import get_object_or_404

def my_view(request):
    my_object = get_object_or_404(MyModel, pk=1)


#is equivalent to:


from django.http import Http404

def my_view(request):
    try:
        my_object = MyModel.objects.get(pk=1)
    except MyModel.DoesNotExist:
        raise Http404("No MyModel matches the given query.")


# can also pass a QuerySet instance:
queryset = Book.objects.filter(title__startswith='M')
get_object_or_404(queryset, pk=1)

#is equivalent to 
get_object_or_404(Book, title__startswith='M', pk=1)

#can also use a Manager
get_object_or_404(Book.dahl_objects, title='Matilda')


#can  use related managers:
author = Author.objects.get(name='Roald Dahl')
get_object_or_404(author.book_set, title='Matilda')




#get_list_or_404()
get_list_or_404(klass, *args, **kwargs)
Returns the result of filter() on a given model manager(klass) cast to a list, 
raising Http404 if the resulting list is empty.

#Example - gets all published objects from MyModel:


from django.shortcuts import get_list_or_404

def my_view(request):
    my_objects = get_list_or_404(MyModel, published=True)


#is equivalent to:
from django.http import Http404

def my_view(request):
    my_objects = list(MyModel.objects.filter(published=True))
    if not my_objects:
        raise Http404("No MyModel matches the given query.")






###Django - Request and response objects


class HttpRequest

HttpRequest.body        raw HTTP request body as a byte string
HttpRequest.path        example: /music/bands/the_beatles/
HttpRequest.get_full_path()  example: "/music/bands/the_beatles/?print=true"
HttpRequest.method      'GET' or 'POST' or others 
HttpRequest.content_type
HttpRequest.GET          QueryDict ,  dictionary-like object , access param as ['param']
HttpRequest.POST         QueryDict ,  dictionary-like object 
HttpRequest.FILES       {'name':  UploadedFile_instance}  , Must be : enctype="multipart/form-data" and <form> contains <input type="file" name="" />    


#HttpRequest.META
A standard Python dictionary containing all available HTTP headers. 
a header called X-Bender would be mapped to the META key HTTP_X_BENDER.

•CONTENT_LENGTH         – The length of the request body (as a string).
•CONTENT_TYPE           – The MIME type of the request body.
•HTTP_ACCEPT            – Acceptable content types for the response.
•HTTP_ACCEPT_ENCODING   – Acceptable encodings for the response.
•HTTP_ACCEPT_LANGUAGE   – Acceptable languages for the response.
•HTTP_HOST              – The HTTP Host header sent by the client.
•HTTP_REFERER           – The referring page, if any.
•HTTP_USER_AGENT        – The client’s user-agent string.
•QUERY_STRING           – The query string, as a single (unparsed) string.
•REMOTE_ADDR            – The IP address of the client.
•REMOTE_HOST            – The hostname of the client.
•REMOTE_USER            – The user authenticated by the Web server, if any.
•REQUEST_METHOD         – A string such as "GET" or "POST".
•SERVER_NAME            – The hostname of the server.
•SERVER_PORT            – The port of the server (as a string).



#HttpRequest.user
user is only available with AuthenticationMiddleware activated. 
An object of type AUTH_USER_MODEL representing the currently logged-in user. 
If the user isn’t currently logged in, user will be set to 
an instance of django.contrib.auth.models.AnonymousUser. 

if request.user.is_authenticated():
    # Do something for logged-in users.
else:
    # Do something for anonymous users.


#HttpRequest.session
A readable-and-writable, dictionary-like object that represents the current session. 
This is only available if your Django installation has session support activated.



#HttpRequest.get_host()
Returns the originating host of the request using information 
from the HTTP_X_FORWARDED_HOST (if USE_X_FORWARDED_HOST is enabled) 
and HTTP_HOST headers, in that order. 
or  a combination of SERVER_NAME and SERVER_PORT 


#HttpRequest.get_full_path()
Returns the path, plus an appended query string, if applicable.
Example: "/music/bands/the_beatles/?print=true"


#HttpRequest.build_absolute_uri(location)
Returns the absolute URI form of location. 
Example: "http://example.com/music/bands/the_beatles/?print=true"


#HttpRequest.get_signed_cookie(key, default=RAISE_ERROR, salt='', max_age=None)
Returns a cookie value for a signed cookie, 
or raises a django.core.signing.BadSignature exception if the signature is no longer valid. 



request.get_signed_cookie('name') #'Tony'
request.get_signed_cookie('name', salt='name-salt') #'Tony' # assuming cookie was set using the same salt
request.get_signed_cookie('non-existing-cookie')   #KeyError: 'non-existing-cookie'
request.get_signed_cookie('non-existing-cookie', False) #False
request.get_signed_cookie('cookie-that-was-tampered-with') #BadSignature: ...
request.get_signed_cookie('name', max_age=60) #SignatureExpired: Signature age 1677.3839159 > 60 seconds
request.get_signed_cookie('name', False, max_age=60) #False



#HttpRequest.is_secure()
Returns True if the request is secure; that is, if it was made with HTTPS.


#HttpRequest.is_ajax()
Returns True if the request was made via an XMLHttpRequest, 
by checking the HTTP_X_REQUESTED_WITH header for the string 'XMLHttpRequest'. 

#Reading HttpRequest manually
HttpRequest.read(size=None)
HttpRequest.readline()
HttpRequest.readlines()
HttpRequest.xreadlines()
HttpRequest.__iter__()





##class HttpResponse

#Usage 
from django.http import HttpResponse
response = HttpResponse("Here's the text of the Web page.")
response = HttpResponse("Text only, please.", content_type="text/plain")

#Or incrementally 
response = HttpResponse()
response.write("<p>Here's the text of the Web page.</p>")
response.write("<p>Here's another paragraph.</p>")

#Or can pass iterators



##Setting header fields
#To set or remove a header field in response, treat it like a dictionary:
#HTTP header fields cannot contain newlines
response = HttpResponse()
response['Age'] = 120
del response['Age']



#Telling the browser to treat the response as a file attachment
response = HttpResponse(my_data, content_type='application/vnd.ms-excel')
response['Content-Disposition'] = 'attachment; filename="foo.xls"'


#HttpResponse.content
A bytestring representing the content, encoded from a Unicode object if necessary.

#HttpResponse.status_code
The HTTP status code for the response.

#HttpResponse.reason_phrase
The HTTP reason phrase for the response.

#HttpResponse.has_header(header)
Returns True or False based on a case-insensitive check for a header 
with the given name.

#HttpResponse.set_cookie(key, value='', max_age=None, expires=None, path='/', domain=None, secure=None, httponly=False)
Sets a cookie. The parameters are the same as in the Morsel cookie object in the Python standard library.
Both RFC 2109 and RFC 6265 state that user agents should support cookies of at least 4096 bytes. 

•max_age    should be a number of seconds, or None (default) if the cookie should last only as long as the client’s browser session. If expires is not specified, it will be calculated.
•expires    should either be a string in the format "Wdy, DD-Mon-YY HH:MM:SS GMT" or a datetime.datetime object in UTC. If expires is a datetime object, the max_age will be calculated.
•Use domain if you want to set a cross-domain cookie. For example, domain=".lawrence.com" will set a cookie that is readable by the domains www.lawrence.com, blogs.lawrence.com and calendars.lawrence.com. Otherwise, a cookie will only be readable by the domain that set it.
•Use httponly=True if you want to prevent client-side JavaScript from having access to the cookie.



#HttpResponse.set_signed_cookie(key, value, salt='', max_age=None, expires=None, path='/', domain=None, secure=None, httponly=True)
Like set_cookie(), but cryptographic signing the cookie before setting it. 


#HttpResponse.delete_cookie(key, path='/', domain=None)
Deletes the cookie with the given key. Fails silently if the key doesn’t exist.

#HttpResponse.write(content)
This method makes an HttpResponse instance a file-like object.

#HttpResponse.flush()
This method makes an HttpResponse instance a file-like object.

#HttpResponse.tell()
This method makes an HttpResponse instance a file-like object.


##class HttpResponseRedirect(url)
url be a fully qualified URL (e.g. 'http://www.yahoo.com/search/') 
or an absolute path with no domain (e.g. '/search/'). 


##class HttpResponsePermanentRedirect
Like HttpResponseRedirect, 
but it returns a permanent redirect (HTTP status code 301) instead of a “found” redirect (status code 302).


##class HttpResponseNotModified
The constructor doesn’t take any arguments 
and no content should be added to this response. 
Use this to designate that a page hasn’t been modified since the user’s last request (status code 304).

##class HttpResponseBadRequest
Acts just like HttpResponse but uses a 400 status code.

##class HttpResponseNotFound
Acts just like HttpResponse but uses a 404 status code.

##class HttpResponseForbidden
Acts just like HttpResponse but uses a 403 status code.

##class HttpResponseNotAllowed
Like HttpResponse, but uses a 405 status code. 
The first argument to the constructor is required: a list of permitted methods 
(e.g. ['GET', 'POST']).

##class HttpResponseGone
Acts just like HttpResponse but uses a 410 status code.

##class HttpResponseServerError
Acts just like HttpResponse but uses a 500 status code.


##class JsonResponse
JsonResponse.__init__(data, encoder=DjangoJSONEncoder, safe=True, **kwargs)
An HttpResponse subclass that helps to create a JSON-encoded response. 
data, should be a dict instance. 
If the safe parameter is set to False ,it can be any JSON-serializable object.


from django.http import JsonResponse
response = JsonResponse({'foo': 'bar'})
response.content  #'{"foo": "bar"}'

#Serializing non-dictionary objects

response = JsonResponse([1, 2, 3], safe=False) #Without passing safe=False, a TypeError will be raised.


##class StreamingHttpResponse
The StreamingHttpResponse class is used to stream a response from Django 
to the browser. 
The StreamingHttpResponse is not a subclass of HttpResponse, 
because it features a slightly different but mostly same  API. 

• It should be given an iterator that yields strings as content.
• You cannot access its content, except by iterating the response object itself. 
  This should only occur when the response is returned to the client.
• It has no content attribute. Instead, it has a streaming_content attribute.
• You cannot use the file-like object tell() or write() methods. 


#StreamingHttpResponse.streaming_content
An iterator of strings representing the content.

#StreamingHttpResponse.status_code
The HTTP status code for the response.

#StreamingHttpResponse.reason_phrase
The HTTP reason phrase for the response.

#StreamingHttpResponse.streaming
This is always True.




@@@

###Django - Working with Form 

#Example - simple.html 

<form action="/your-name/" method="post">
    <label for="your_name">Your name: </label>
    <input id="your_name" type="text" name="your_name" value="{{ current_name }}">
    <input type="submit" value="OK">
</form>

#To get above 
#forms.py

from django import forms

class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)


#is equivalent to below , note no <form> tags, or a submit button


<label for="your_name">Your name: </label>
<input id="your_name" type="text" name="your_name" maxlength="100" required />


#To handle the form , instantiate it in the view for the URL 
#views.py

from django.shortcuts import render
from django.http import HttpResponseRedirect

from .forms import NameForm

def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()

    return render(request, 'name.html', {'form': form})

#name.html template csrf - Cross Site Request Forgery protection
#{{form}} - All the form’s fields and their attributes will be unpacked into HTML markup
<form action="/your-name/" method="post">
    {% csrf_token %}
    {{ form }}
    <input type="submit" value="Submit" />
</form>


##Bound and unbound form instances - is_bound attribute
• An unbound form has no data associated with it. 
  When rendered to the user, it will be empty or will contain default values.
• A bound form has submitted data, 
  and hence can be used to tell if that data is valid. 
  If an invalid bound form is rendered, it can include inline error messages 


##More on fields (ref - https://docs.djangoproject.com/en/1.10/ref/forms/fields/)
#forms.py

from django import forms

class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)
    sender = forms.EmailField()
    cc_myself = forms.BooleanField(required=False)


#Each form field has a corresponding Widget class, 
#which in turn corresponds to an HTML form widget such as <input type="text">.

##Attributes 
Field.data
Whatever the data submitted with a form, 
If it has been successfully validated by calling is_valid() 
the validated form data will be in the form.cleaned_data python dictionary. 

#views.py

from django.core.mail import send_mail

if form.is_valid():
    subject = form.cleaned_data['subject']
    message = form.cleaned_data['message']
    sender = form.cleaned_data['sender']
    cc_myself = form.cleaned_data['cc_myself']

    recipients = ['info@example.com']
    if cc_myself:
        recipients.append(sender)

    send_mail(subject, message, sender, recipients)
    return HttpResponseRedirect('/thanks/')

#Form rendering options - other than {{form}}
#provide the surrounding <table> or <ul>
•{{ form.as_table }}    will render them as table cells wrapped in <tr> tags
•{{ form.as_p }}        will render them wrapped in <p> tags
•{{ form.as_ul }}       will render them wrapped in <li> tags


##Rendering fields manually
#Each field is available as an attribute of the form using {{ form.name_of_field }}, 
#label ID is available as {{ form.name_of_field.id_for_label }}
#Any error in field is available as {{ form.name_of_field.errors }}
#{{ form.non_field_errors }} contains all nonfield form error 


{{ form.non_field_errors }}
<div class="fieldWrapper">
    {{ form.subject.errors }}
    <label for="{{ form.subject.id_for_label }}">Email subject:</label>
    {{ form.subject }}
</div>
<div class="fieldWrapper">
    {{ form.message.errors }}
    <label for="{{ form.message.id_for_label }}">Your message:</label>
    {{ form.message }}
</div>
<div class="fieldWrapper">
    {{ form.sender.errors }}
    <label for="{{ form.sender.id_for_label }}">Your email address:</label>
    {{ form.sender }}
</div>
<div class="fieldWrapper">
    {{ form.cc_myself.errors }}
    <label for="{{ form.cc_myself.id_for_label }}">CC yourself?</label>
    {{ form.cc_myself }}
</div>


#Or  <label> elements can also be generated using the label_tag()
<div class="fieldWrapper">
    {{ form.subject.errors }}
    {{ form.subject.label_tag }}
    {{ form.subject }}
</div>



#Rendering form error messages
#{{ form.subject.errors }} would look as below 
<ul class="errorlist">
    <li>Sender is required.</li>
</ul>

#Or iterate manually
{% if form.subject.errors %}
    <ol>
    {% for error in form.subject.errors %}
        <li><strong>{{ error|escape }}</strong></li>
    {% endfor %}
    </ol>
{% endif %}


#{{ form.non_field_errors }} would look like:

<ul class="errorlist nonfield">
    <li>Generic validation error</li>
</ul>



##Looping over the form’s fields - {% for %} loop

{% for field in form %}
    <div class="fieldWrapper">
        {{ field.errors }}
        {{ field.label_tag }} {{ field }}
        {% if field.help_text %}
        <p class="help">{{ field.help_text|safe }}</p>
        {% endif %}
    </div>
{% endfor %}


##Useful attributes on {{ field }} 
{{ field.label }}           The label of the field, e.g. forms.CharField(label='Your name')
{{ field.label_tag }}       The field’s label wrapped in the appropriate HTML <label> tag. 

{{ field.id_for_label }}    The ID that will be used for this field (id_email , <label for="id_email">Email address:</label>). 
{{ field.value }}           The value of the field. e.g someone@example.com.
{{ field.html_name }}       The name of the field that will be used in the input element’s name field. This takes the form prefix into account, if it has been set.
{{ field.help_text }}       Any help text that has been associated with the field.
{{ field.errors }}          Outputs a <ul class="errorlist"> containing any validation errors corresponding to this field. You can customize the presentation of the errors with a {% for error in field.errors %} loop. In this case, each object in the loop is a simple string containing the error message.
{{ field.is_hidden }}       This attribute is True if the form field is a hidden field and False otherwise.
                             {% if field.is_hidden %}
                               {# Do something special #}
                             {% endif %}
{{ field.field }}           The Field instance from the form class that this BoundField wraps. 
                            You can use it to access Field attributes, 
                            e.g. {{ char_field.field.max_length }}.


##Looping over hidden and visible fields- hidden_fields() and visible_fields(). 

{# Include the hidden fields #}
{% for hidden in form.hidden_fields %}
{{ hidden }}
{% endfor %}
{# Include the visible fields #}
{% for field in form.visible_fields %}
    <div class="fieldWrapper">
        {{ field.errors }}
        {{ field.label_tag }} {{ field }}
    </div>
{% endfor %}


#Reusable form templates - use include 

# In your form template:
{% include "form_snippet.html" %}

# In form_snippet.html:
{% for field in form %}
    <div class="fieldWrapper">
        {{ field.errors }}
        {{ field.label_tag }} {{ field }}
    </div>
{% endfor %}


#If the form object passed to a template has a different name within the context,
{% include "form_snippet.html" with form=comment_form %}


## Other Attributes 
Field.clean(value)
For validation 
Raises a django.forms.ValidationError exception or returns the clean value:


>>> from django import forms
>>> f = forms.EmailField()
>>> f.clean('foo@example.com')
'foo@example.com'
>>> f.clean('invalid email address')
Traceback (most recent call last):
...
ValidationError: ['Enter a valid email address.']



##Core field arguments- Each Field class constructor takes at least these arguments.

#Field.required
By default, each Field class assumes the value is required, 


>>> from django import forms
>>> f = forms.CharField()
>>> f.clean('foo')
'foo'
>>> f.clean('')
Traceback (most recent call last):
...
ValidationError: ['This field is required.']
>>> f.clean(None)
Traceback (most recent call last):
...
ValidationError: ['This field is required.']
>>> f.clean(' ')
' '
>>> f.clean(0)
'0'
>>> f.clean(True)
'True'
>>> f.clean(False)
'False'


#To specify that a field is not required, pass required=False 
>>> f = forms.CharField(required=False)
>>> f.clean('foo')
'foo'
>>> f.clean('')
''
>>> f.clean(None)
''
>>> f.clean(0)
'0'
>>> f.clean(True)
'True'
>>> f.clean(False)
'False'



#Field.label_suffix
The label_suffix argument lets you override the form’s label_suffix 

>>> class ContactForm(forms.Form):
        age = forms.IntegerField()
        nationality = forms.CharField()
        captcha_answer = forms.IntegerField(label='2 + 2', label_suffix=' =')
>>> f = ContactForm(label_suffix='?')
>>> print(f.as_p())
<p><label for="id_age">Age?</label> <input id="id_age" name="age" type="number" required /></p>
<p><label for="id_nationality">Nationality?</label> <input id="id_nationality" name="nationality" type="text" required /></p>
<p><label for="id_captcha_answer">2 + 2 =</label> <input id="id_captcha_answer" name="captcha_answer" type="number" required /></p>


#Field.initial
The initial argument lets you specify the initial value to use 
when rendering this Field in an unbound Form.
To specify dynamic initial data, see the Form.initial parameter.
Initial values are only displayed for unbound forms. 
For bound forms, the HTML output will use the bound data.

>>> from django import forms
>>> class CommentForm(forms.Form):
        name = forms.CharField(initial='Your name')
        url = forms.URLField(initial='http://')
        comment = forms.CharField()
>>> f = CommentForm(auto_id=False)
>>> print(f)
<tr><th>Name:</th><td><input type="text" name="name" value="Your name" required /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url" value="http://" required /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" required /></td></tr>

#If used in Form ctor, it triggers validation, 
#and the HTML output will include any validation errors:


>>> class CommentForm(forms.Form):
        name = forms.CharField()
        url = forms.URLField()
        comment = forms.CharField()
>>> default_data = {'name': 'Your name', 'url': 'http://'}
>>> f = CommentForm(default_data, auto_id=False)
>>> print(f)
<tr><th>Name:</th><td><input type="text" name="name" value="Your name" required /></td></tr>
<tr><th>Url:</th><td><ul class="errorlist"><li>Enter a valid URL.</li></ul><input type="url" name="url" value="http://" required /></td></tr>
<tr><th>Comment:</th><td><ul class="errorlist"><li>This field is required.</li></ul><input type="text" name="comment" required /></td></tr>

#Field.help_text
The help_text argument lets you specify descriptive text for this Field. 

#specified auto_id=False to simplify the output:


>>> from django import forms
>>> class HelpTextContactForm(forms.Form):
        subject = forms.CharField(max_length=100, help_text='100 characters max.')
        message = forms.CharField()
        sender = forms.EmailField(help_text='A valid email address, please.')
        cc_myself = forms.BooleanField(required=False)
>>> f = HelpTextContactForm(auto_id=False)
>>> print(f.as_table())
<tr><th>Subject:</th><td><input type="text" name="subject" maxlength="100" required /><br /><span class="helptext">100 characters max.</span></td></tr>
<tr><th>Message:</th><td><input type="text" name="message" required /></td></tr>
<tr><th>Sender:</th><td><input type="email" name="sender" required /><br />A valid email address, please.</td></tr>
<tr><th>Cc myself:</th><td><input type="checkbox" name="cc_myself" /></td></tr>
>>> print(f.as_ul()))
<li>Subject: <input type="text" name="subject" maxlength="100" required /> <span class="helptext">100 characters max.</span></li>
<li>Message: <input type="text" name="message" required /></li>
<li>Sender: <input type="email" name="sender" required /> A valid email address, please.</li>
<li>Cc myself: <input type="checkbox" name="cc_myself" /></li>
>>> print(f.as_p())
<p>Subject: <input type="text" name="subject" maxlength="100" required /> <span class="helptext">100 characters max.</span></p>
<p>Message: <input type="text" name="message" required /></p>
<p>Sender: <input type="email" name="sender" required /> A valid email address, please.</p>
<p>Cc myself: <input type="checkbox" name="cc_myself" /></p>



#Field.error_messages
The error_messages argument lets you override the default messages 

>>> name = forms.CharField(error_messages={'required': 'Please enter your name'})
>>> name.clean('')
Traceback (most recent call last):
  ...
ValidationError: ['Please enter your name']


#Field.validators
The validators argument lets you provide a list of validation functions 


#Field.localize
The localize argument enables the localization of form data input, 
as well as the rendered output.


#Field.disabled
New in Django 1.9. 
when set to True, disables a form field using the disabled HTML attribute 
so that it won’t be editable by users. 

#Field.has_changed()[source]
The has_changed() method is used to determine if the field value has changed 
from the initial value. Returns True or False.

##Form attributies 
@@@
To create an unbound Form instance, simply instantiate the class:


>>> f = ContactForm()


To bind data to a form, pass the data as a dictionary as the first parameter to your Form class constructor:


>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)

Form.is_bound¶
If you need to distinguish between bound and unbound form instances at runtime, check the value of the form’s is_bound attribute:


>>> f = ContactForm()
>>> f.is_bound
False
>>> f = ContactForm({'subject': 'hello'})
>>> f.is_bound
True

Form.clean()¶
Implement a clean() method on your Form when you must add custom validation for fields that are interdependent
Form.is_valid()¶
The primary task of a Form object is to validate data. With a bound Form instance, call the is_valid() method to run validation and return a boolean designating whether the data was valid:


>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> f.is_valid
Form.errors¶
Access the errors attribute to get a dictionary of error messages:


>>> f.errors
{'sender': ['Enter a valid email address.'], 'subject': ['This field is required.']}

Form.errors.as_data()¶
Returns a dict that maps fields to their original ValidationError instances.


>>> f.errors.as_data()
{'sender': [ValidationError(['Enter a valid email address.'])],
'subject': [ValidationError(['This field is required.'])]}

Form.errors.as_json(escape_html=False)¶
Returns the errors serialized as JSON.


>>> f.errors.as_json()
{"sender": [{"message": "Enter a valid email address.", "code": "invalid"}],
"subject": [{"message": "This field is required.", "code": "required"}]}

Form.add_error(field, error)¶
This method allows adding errors to specific fields from within the Form.clean() method, or from outside the form altogether; for instance from a view.
field is None the error will be treated as a non-field error as returned by Form.non_field_errors().

Form.has_error(field, code=None)¶
This method returns a boolean designating whether a field has an error with a specific error code
Form.non_field_errors()¶
This method returns the list of errors from Form.errors that aren’t associated with a particular field

Form.initial¶
Use initial to declare the initial value of form fields at runtime
>>> f = ContactForm(initial={'subject': 'Hi there!'})

Form.has_changed()¶
Use the has_changed() method on your Form when you need to check if the form data has been changed from the initial data.


>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data, initial=data)
>>> f.has_changed()
False

When the form is submitted, we reconstruct it and provide the original data so that the comparison can be done:


>>> f = ContactForm(request.POST, initial=data)
>>> f.has_changed()
True 

Form.changed_data¶
The changed_data attribute returns a list of the names of the fields whose values in the form’s bound data (usually request.POST) differ from what was provided in initial. It returns an empty list if no data differs.
Form.fields¶
You can access the fields of Form instance from its fields attribute:


>>> for row in f.fields.values(): print(row)

Form.cleaned_data¶
Each field in a Form class is responsible not only for validating data, but also for “cleaning” it 
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> f.is_valid()
True
>>> f.cleaned_data
{'cc_myself': True, 'message': 'Hi there', 'sender': 'foo@example.com', 'subject': 'hello'}

Form.error_css_class¶Form.required_css_class¶
It’s pretty common to style form rows and fields that are required or have errors. For example, you might want to present required form rows in bold and highlight errors in red.

The Form class has a couple of hooks you can use to add class attributes to required rows or to rows with errors: simply set the Form.error_css_class and/or Form.required_css_class attributes:


from django.forms import Form

class ContactForm(Form):
    error_css_class = 'error'
    required_css_class = 'required'

    # ... and the rest of your fields here


Once you’ve done that, rows will be given "error" and/or "required" classes, as needed. The HTML will look something like:


>>> f = ContactForm(data)
>>> print(f.as_table())
<tr class="required"><th><label class="required" for="id_subject">Subject:</label>    ...
<tr class="required"><th><label class="required" for="id_message">Message:</label>    ...
<tr class="required error"><th><label class="required" for="id_sender">Sender:</label>      ...
<tr><th><label for="id_cc_myself">Cc myself:<label> ...
>>> f['subject'].label_tag()
<label class="required" for="id_subject">Subject:</label>
>>> f['subject'].label_tag(attrs={'class': 'foo'})
<label for="id_subject" class="foo required">Subject:</label>

Form.auto_id¶
By default, the form rendering methods include:
•HTML id attributes on the form elements.
•The corresponding <label> tags around the labels
If auto_id is False, then the form output will not include <label> tags nor id attributes:


>>> f = ContactForm(auto_id=False)

If auto_id is set to a string containing the format character '%s', then the form output will include <label> tags, and will generate id attributes based on the format string
>>> f = ContactForm(auto_id='id_for_%s')
>>> print(f.as_table())

By default, auto_id is set to the string 'id_%s'.
Form.label_suffix¶
A translatable string (defaults to a colon (:) in English) that will be appended after any label name when a form is rendered.
Form.use_required_attribute¶
New in Django 1.10. 

When set to True (the default), required form fields will have the required HTML attribute
Form.field_order¶
New in Django 1.9. 

By default Form.field_order=None, which retains the order in which you define the fields in your form class
Form.order_fields(field_order)¶
New in Django 1.9. 

You may rearrange the fields any time using order_fields() with a list of field names as in field_order.

##BoundField when field is bounded to data 

To retrieve a single BoundField, use dictionary lookup syntax on your form using the field’s name as the key:


>>> form = ContactForm()
>>> print(form['subject'])
<input id="id_subject" type="text" name="subject" maxlength="100" required />


To retrieve all BoundField objects, iterate the form:


>>> form = ContactForm()
>>> for boundfield in form: print(boundfield)
<input id="id_subject" type="text" name="subject" maxlength="100" required />
<input type="text" name="message" id="id_message" required />
<input type="email" name="sender" id="id_sender" required />
<input type="checkbox" name="cc_myself" id="id_cc_myself" />


The field-specific output honors the form object’s auto_id setting:


>>> f = ContactForm(auto_id=False)
>>> print(f['message'])
<input type="text" name="message" required />
>>> f = ContactForm(auto_id='id_%s')
>>> print(f['message'])
<input type="text" name="message" id="id_message" required />

BoundField.data¶
This property returns the data for this BoundField extracted by the widget’s value_from_datadict() method, or None if it wasn’t given:


>>> unbound_form = ContactForm()
>>> print(unbound_form['subject'].data)
None
>>> bound_form = ContactForm(data={'subject': 'My Subject'})
>>> print(bound_form['subject'].data)
My Subject





Using validators¶

Validation of a form is split into several steps, which can be customized or overridden:

•The to_python() method on a Field is the first step in every validation. It coerces the value to a correct datatype and raises ValidationError if that is not possible. This method accepts the raw value from the widget and returns the converted value. For example, a FloatField will turn the data into a Python float or raise a ValidationError.


•The validate() method on a Field handles field-specific validation that is not suitable for a validator. It takes a value that has been coerced to a correct datatype and raises ValidationError on any error. This method does not return anything and shouldn’t alter the value. You should override it to handle validation logic that you can’t or don’t want to put in a validator.


•The run_validators() method on a Field runs all of the field’s validators and aggregates all the errors into a single ValidationError. You shouldn’t need to override this method.


•The clean() method on a Field subclass is responsible for running to_python(), validate(), and run_validators() in the correct order and propagating their errors. If, at any time, any of the methods raise ValidationError, the validation stops and that error is raised. This method returns the clean data, which is then inserted into the cleaned_data dictionary of the form.


•The clean_<fieldname>() method is called on a form subclass – where <fieldname> is replaced with the name of the form field attribute. This method does any cleaning that is specific to that particular attribute, unrelated to the type of field that it is. This method is not passed any parameters. You will need to look up the value of the field in self.cleaned_data and remember that it will be a Python object at this point, not the original string submitted in the form (it will be in cleaned_data because the general field clean() method, above, has already cleaned the data once).

For example, if you wanted to validate that the contents of a CharField called serialnumber was unique, clean_serialnumber() would be the right place to do this. You don’t need a specific field (it’s just a CharField), but you want a formfield-specific piece of validation and, possibly, cleaning/normalizing the data.

This method should return the cleaned value obtained from cleaned_data, regardless of whether it changed anything or not.


•The form subclass’s clean() method can perform validation that requires access to multiple form fields. This is where you might put in checks such as “if field A is supplied, field B must contain a valid email address”. This method can return a completely different dictionary if it wishes, which will be used as the cleaned_data.

Since the field validation methods have been run by the time clean() is called, you also have access to the form’s errors attribute which contains all the errors raised by cleaning of individual fields.

Note that any errors raised by your Form.clean() override will not be associated with any field in particular. They go into a special “field” (called __all__), which you can access via the non_field_errors() method if you need to. If you want to attach errors to a specific field in the form, you need to call add_error().

Also note that there are special considerations when overriding the clean() method of a ModelForm subclass. (see the ModelForm documentation for more information)


These methods are run in the order given above, one field at a time. That is, for each field in the form (in the order they are declared in the form definition), the Field.clean() method (or its override) is run, then clean_<fieldname>(). Finally, once those two methods are run for every field, the Form.clean() method, or its override, is executed whether or not the previous methods have raised errors.



Django’s form (and model) fields support use of simple utility functions and classes known as validators. A validator is merely a callable object or function that takes a value and simply returns nothing if the value is valid or raises a ValidationError if not. These can be passed to a field’s constructor, via the field’s validators argument, or defined on the Field class itself with the default_validators attribute.

Simple validators can be used to validate values inside the field, let’s have a look at Django’s SlugField:


from django.forms import CharField
from django.core import validators

class SlugField(CharField):
    default_validators = [validators.validate_slug]


As you can see, SlugField is just a CharField with a customized validator that validates that submitted text obeys to some character rules. This can also be done on field definition so:


slug = forms.SlugField()


is equivalent to:


slug = forms.CharField(validators=[validators.validate_slug])

Form field default cleaning¶

Let’s first create a custom form field that validates its input is a string containing comma-separated email addresses. The full class looks like this:


from django import forms
from django.core.validators import validate_email

class MultiEmailField(forms.Field):
    def to_python(self, value):
        """Normalize data to a list of strings."""
        # Return an empty list if no input was given.
        if not value:
            return []
        return value.split(',')

    def validate(self, value):
        """Check if value consists only of valid emails."""
        # Use the parent's handling of required fields, etc.
        super(MultiEmailField, self).validate(value)
        for email in value:
            validate_email(email)


Every form that uses this field will have these methods run before anything else can be done with the field’s data. This is cleaning that is specific to this type of field, regardless of how it is subsequently used.

Let’s create a simple ContactForm to demonstrate how you’d use this field:


class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField()
    sender = forms.EmailField()
    recipients = MultiEmailField()
    cc_myself = forms.BooleanField(required=False)


Simply use MultiEmailField like any other form field. When the is_valid() method is called on the form, the MultiEmailField.clean() method will be run as part of the cleaning process and it will, in turn, call the custom to_python() and validate() methods.
Cleaning a specific field attribute¶

Continuing on from the previous example, suppose that in our ContactForm, we want to make sure that the recipients field always contains the address "fred@example.com". This is validation that is specific to our form, so we don’t want to put it into the general MultiEmailField class. Instead, we write a cleaning method that operates on the recipients field, like so:


from django import forms

class ContactForm(forms.Form):
    # Everything as before.
    ...

    def clean_recipients(self):
        data = self.cleaned_data['recipients']
        if "fred@example.com" not in data:
            raise forms.ValidationError("You have forgotten about Fred!")

        # Always return the cleaned data, whether you have changed it or
        # not.
        return data

Cleaning and validating fields that depend on each other¶

Suppose we add another requirement to our contact form: if the cc_myself field is True, the subject must contain the word "help". We are performing validation on more than one field at a time, so the form’s clean() method is a good spot to do this. Notice that we are talking about the clean() method on the form here, whereas earlier we were writing a clean() method on a field. It’s important to keep the field and form difference clear when working out where to validate things. Fields are single data points, forms are a collection of fields.

By the time the form’s clean() method is called, all the individual field clean methods will have been run (the previous two sections), so self.cleaned_data will be populated with any data that has survived so far. So you also need to remember to allow for the fact that the fields you are wanting to validate might not have survived the initial individual field checks.

There are two ways to report any errors from this step. Probably the most common method is to display the error at the top of the form. To create such an error, you can raise a ValidationError from the clean() method. For example:


from django import forms

class ContactForm(forms.Form):
    # Everything as before.
    ...

    def clean(self):
        cleaned_data = super(ContactForm, self).clean()
        cc_myself = cleaned_data.get("cc_myself")
        subject = cleaned_data.get("subject")

        if cc_myself and subject:
            # Only do something if both fields are valid so far.
            if "help" not in subject:
                raise forms.ValidationError(
                    "Did not send for 'help' in the subject despite "
                    "CC'ing yourself."
                )


In this code, if the validation error is raised, the form will display an error message at the top of the form (normally) describing the problem.

The call to super(ContactForm, self).clean() in the example code ensures that any validation logic in parent classes is maintained. If your form inherits another that doesn’t return a cleaned_data dictionary in its clean() method (doing so is optional), then don’t assign cleaned_data to the result of the super() call and use self.cleaned_data instead:


def clean(self):
    super(ContactForm, self).clean()
    cc_myself = self.cleaned_data.get("cc_myself")
    ...


The second approach for reporting validation errors might involve assigning the error message to one of the fields. In this case, let’s assign an error message to both the “subject” and “cc_myself” rows in the form display. Be careful when doing this in practice, since it can lead to confusing form output. We’re showing what is possible here and leaving it up to you and your designers to work out what works effectively in your particular situation. Our new code (replacing the previous sample) looks like this:


from django import forms

class ContactForm(forms.Form):
    # Everything as before.
    ...

    def clean(self):
        cleaned_data = super(ContactForm, self).clean()
        cc_myself = cleaned_data.get("cc_myself")
        subject = cleaned_data.get("subject")

        if cc_myself and subject and "help" not in subject:
            msg = "Must put 'help' in subject when cc'ing yourself."
            self.add_error('cc_myself', msg)
            self.add_error('subject', msg)


The second argument of add_error() can be a simple string, or preferably an instance of ValidationError. See Raising ValidationError for more details. Note that add_error() automatically removes the field from cleaned_data.



@@@
Creating forms from models
Field types¶

The generated Form class will have a form field for every model field specified, in the order specified in the fields attribute.

Each model field has a corresponding default form field. For example, a CharField on a model is represented as a CharField on a form. A model ManyToManyField is represented as a MultipleChoiceField. Here is the full list of conversions:





Model field

Form field


AutoField Not represented in the form 
BigAutoField Not represented in the form 
BigIntegerField IntegerField with min_value set to -9223372036854775808 and max_value set to 9223372036854775807. 
BooleanField BooleanField 
CharField CharField with max_length set to the model field’s max_length 
CommaSeparatedIntegerField CharField 
DateField DateField 
DateTimeField DateTimeField 
DecimalField DecimalField 
EmailField EmailField 
FileField FileField 
FilePathField FilePathField 
FloatField FloatField 
ForeignKey ModelChoiceField (see below) 
ImageField ImageField 
IntegerField IntegerField 
IPAddressField IPAddressField 
GenericIPAddressField GenericIPAddressField 
ManyToManyField ModelMultipleChoiceField (see below) 
NullBooleanField NullBooleanField 
PositiveIntegerField IntegerField 
PositiveSmallIntegerField IntegerField 
SlugField SlugField 
SmallIntegerField IntegerField 
TextField CharField with widget=forms.Textarea 
TimeField TimeField 
URLField URLField 

As you might expect, the ForeignKey and ManyToManyField model field types are special cases:
•ForeignKey is represented by django.forms.ModelChoiceField, which is a ChoiceField whose choices are a model QuerySet.
•ManyToManyField is represented by django.forms.ModelMultipleChoiceField, which is a MultipleChoiceField whose choices are a model QuerySet.

A full example¶

Consider this set of models:


from django.db import models
from django.forms import ModelForm

TITLE_CHOICES = (
    ('MR', 'Mr.'),
    ('MRS', 'Mrs.'),
    ('MS', 'Ms.'),
)

class Author(models.Model):
    name = models.CharField(max_length=100)
    title = models.CharField(max_length=3, choices=TITLE_CHOICES)
    birth_date = models.DateField(blank=True, null=True)

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Book(models.Model):
    name = models.CharField(max_length=100)
    authors = models.ManyToManyField(Author)

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = ['name', 'title', 'birth_date']

class BookForm(ModelForm):
    class Meta:
        model = Book
        fields = ['name', 'authors']


With these models, the ModelForm subclasses above would be roughly equivalent to this (the only difference being the save() method, which we’ll discuss in a moment.):


from django import forms

class AuthorForm(forms.Form):
    name = forms.CharField(max_length=100)
    title = forms.CharField(
        max_length=3,
        widget=forms.Select(choices=TITLE_CHOICES),
    )
    birth_date = forms.DateField(required=False)

class BookForm(forms.Form):
    name = forms.CharField(max_length=100)
    authors = forms.ModelMultipleChoiceField(queryset=Author.objects.all())



Validation on a ModelForm¶

There are two main steps involved in validating a ModelForm:
1.Validating the form
2.Validating the model instance

Just like normal form validation, model form validation is triggered implicitly when calling is_valid() or accessing the errors attribute and explicitly when calling full_clean(), although you will typically not use the latter method in practice.

Model validation (Model.full_clean()) is triggered from within the form validation step, right after the form’s clean() method is called.
Overriding the clean() method¶

You can override the clean() method on a model form to provide additional validation in the same way you can on a normal form.

A model form instance attached to a model object will contain an instance attribute that gives its methods access to that specific model instance
Interaction with model validation¶

As part of the validation process, ModelForm will call the clean() method of each field on your model that has a corresponding field on your form. If you have excluded any model fields, validation will not be run on those fields


Considerations regarding model’s error_messages¶

Error messages defined at the form field level or at the form Meta level always take precedence over the error messages defined at the model field level.

Error messages defined on model fields are only used when the ValidationError is raised during the model validation step and no corresponding error messages are defined at the form level.

You can override the error messages from NON_FIELD_ERRORS raised by model validation by adding the NON_FIELD_ERRORS key to the error_messages dictionary of the ModelForm’s inner Meta class:


from django.forms import ModelForm
from django.core.exceptions import NON_FIELD_ERRORS

class ArticleForm(ModelForm):
    class Meta:
        error_messages = {
            NON_FIELD_ERRORS: {
                'unique_together': "%(model_name)s's %(field_labels)s are not unique.",
            }
        }



The save() method¶

Every ModelForm also has a save() method. This method creates and saves a database object from the data bound to the form. A subclass of ModelForm can accept an existing model instance as the keyword argument instance; if this is supplied, save() will update that instance. If it’s not supplied, save() will create a new instance of the specified model:


>>> from myapp.models import Article
>>> from myapp.forms import ArticleForm

# Create a form instance from POST data.
>>> f = ArticleForm(request.POST)

# Save a new Article object from the form's data.
>>> new_article = f.save()

# Create a form to edit an existing Article, but use
# POST data to populate the form.
>>> a = Article.objects.get(pk=1)
>>> f = ArticleForm(request.POST, instance=a)
>>> f.save()


Note that if the form hasn’t been validated, calling save() will do so by checking form.errors. A ValueError will be raised if the data in the form doesn’t validate – i.e., if form.errors evaluates to True.

This save() method accepts an optional commit keyword argument, which accepts either True or False. If you call save() with commit=False, then it will return an object that hasn’t yet been saved to the database. In this case, it’s up to you to call save() on the resulting model instance

    
  Selecting the fields to use¶

It is strongly recommended that you explicitly set all fields that should be edited in the form using the fields attribute. Failure to do so can easily lead to security problems when a form unexpectedly allows a user to set certain fields, especially when new fields are added to a model. Depending on how the form is rendered, the problem may not even be visible on the web page.
There are, however, two shortcuts available for cases where you can guarantee these security concerns do not apply to you:

1.Set the fields attribute to the special value '__all__' to indicate that all fields in the model should be used. For example:


from django.forms import ModelForm

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = '__all__'



2.Set the exclude attribute of the ModelForm’s inner Meta class to a list of fields to be excluded from the form.

For example:


class PartialAuthorForm(ModelForm):
    class Meta:
        model = Author
        exclude = ['title']


Since the Author model has the 3 fields name, title and birth_date, this will result in the fields name and birth_date being present on the form.


If either of these are used, the order the fields appear in the form will be the order the fields are defined in the model, with ManyToManyField instances appearing last.

In addition, Django applies the following rule: if you set editable=False on the model field, any form created from the model via ModelForm will not include that field.
Overriding the default fields¶

The default field types, as described in the Field types table above, are sensible defaults. If you have a DateField in your model, chances are you’d want that to be represented as a DateField in your form. But ModelForm gives you the flexibility of changing the form field for a given model.

To specify a custom widget for a field, use the widgets attribute of the inner Meta class. This should be a dictionary mapping field names to widget classes or instances.

For example, if you want the CharField for the name attribute of Author to be represented by a <textarea> instead of its default <input type="text">, you can override the field’s widget:


from django.forms import ModelForm, Textarea
from myapp.models import Author

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = ('name', 'title', 'birth_date')
        widgets = {
            'name': Textarea(attrs={'cols': 80, 'rows': 20}),
        }


The widgets dictionary accepts either widget instances (e.g., Textarea(...)) or classes (e.g., Textarea).

Similarly, you can specify the labels, help_texts and error_messages attributes of the inner Meta class if you want to further customize a field.

For example if you wanted to customize the wording of all user facing strings for the name field:


from django.utils.translation import ugettext_lazy as _

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = ('name', 'title', 'birth_date')
        labels = {
            'name': _('Writer'),
        }
        help_texts = {
            'name': _('Some useful help text.'),
        }
        error_messages = {
            'name': {
                'max_length': _("This writer's name is too long."),
            },
        }


You can also specify field_classes to customize the type of fields instantiated by the form.

For example, if you wanted to use MySlugFormField for the slug field, you could do the following:


from django.forms import ModelForm
from myapp.models import Article

class ArticleForm(ModelForm):
    class Meta:
        model = Article
        fields = ['pub_date', 'headline', 'content', 'reporter', 'slug']
        field_classes = {
            'slug': MySlugFormField,
        }


Finally, if you want complete control over of a field – including its type, validators, required, etc. – you can do this by declaratively specifying fields like you would in a regular Form.

If you want to specify a field’s validators, you can do so by defining the field declaratively and setting its validators parameter:


from django.forms import ModelForm, CharField
from myapp.models import Article

class ArticleForm(ModelForm):
    slug = CharField(validators=[validate_slug])

    class Meta:
        model = Article
        fields = ['pub_date', 'headline', 'content', 'reporter', 'slug']


Enabling localization of fields¶

By default, the fields in a ModelForm will not localize their data. To enable localization for fields, you can use the localized_fields attribute on the Meta class.


>>> from django.forms import ModelForm
>>> from myapp.models import Author
>>> class AuthorForm(ModelForm):
...     class Meta:
...         model = Author
...         localized_fields = ('birth_date',)


If localized_fields is set to the special value '__all__', all fields will be localized.


Form inheritance¶

As with basic forms, you can extend and reuse ModelForms by inheriting them. This is useful if you need to declare extra fields or extra methods on a parent class for use in a number of forms derived from models. For example, using the previous ArticleForm class:


>>> class EnhancedArticleForm(ArticleForm):
...     def clean_pub_date(self):
...         ...


This creates a form that behaves identically to ArticleForm, except there’s some extra validation and cleaning for the pub_date field.

You can also subclass the parent’s Meta inner class if you want to change the Meta.fields or Meta.exclude lists:


>>> class RestrictedArticleForm(EnhancedArticleForm):
...     class Meta(ArticleForm.Meta):
...         exclude = ('body',)


This adds the extra method from the EnhancedArticleForm and modifies the original ArticleForm.Meta to remove one field.

Providing initial values¶

As with regular forms, it’s possible to specify initial data for forms by specifying an initial parameter when instantiating the form. Initial values provided this way will override both initial values from the form field and values from an attached model instance. For example:


>>> article = Article.objects.get(pk=1)
>>> article.headline
'My headline'
>>> form = ArticleForm(initial={'headline': 'Initial headline'}, instance=article)
>>> form['headline'].value()
'Initial headline'



ModelForm factory function¶

You can create forms from a given model using the standalone function modelform_factory(), instead of using a class definition. This may be more convenient if you do not have many customizations to make:


>>> from django.forms import modelform_factory
>>> from myapp.models import Book
>>> BookForm = modelform_factory(Book, fields=("author", "title"))


This can also be used to make simple modifications to existing forms, for example by specifying the widgets to be used for a given field:


>>> from django.forms import Textarea
>>> Form = modelform_factory(Book, form=BookForm,
...                          widgets={"title": Textarea()})


The fields to include can be specified using the fields and exclude keyword arguments, or the corresponding attributes on the ModelForm inner Meta class. Please see the ModelForm Selecting the fields to use documentation.

... or enable localization for specific fields:


>>> Form = modelform_factory(Author, form=AuthorForm, localized_fields=("birth_date",))


@@@
Formsets¶
class BaseFormSet[source]¶
A formset is a layer of abstraction to work with multiple forms on the same page. It can be best compared to a data grid. Let’s say you have the following form:


>>> from django import forms
>>> class ArticleForm(forms.Form):
...     title = forms.CharField()
...     pub_date = forms.DateField()


You might want to allow the user to create several articles at once. To create a formset out of an ArticleForm you would do:


>>> from django.forms import formset_factory
>>> ArticleFormSet = formset_factory(ArticleForm)


You now have created a formset named ArticleFormSet. The formset gives you the ability to iterate over the forms in the formset and display them as you would with a regular form:


>>> formset = ArticleFormSet()
>>> for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" id="id_form-0-pub_date" /></td></tr>


As you can see it only displayed one empty form. The number of empty forms that is displayed is controlled by the extra parameter. By default, formset_factory() defines one extra form; the following example will display two blank forms:


>>> ArticleFormSet = formset_factory(ArticleForm, extra=2)

Using initial data with a formset¶

Initial data is what drives the main usability of a formset. As shown above you can define the number of extra forms. What this means is that you are telling the formset how many additional forms to show in addition to the number of forms it generates from the initial data. Let’s take a look at an example:


>>> import datetime
>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm, extra=2)
>>> formset = ArticleFormSet(initial=[
...     {'title': 'Django is now open source',
...      'pub_date': datetime.date.today(),}
... ])

>>> for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" value="Django is now open source" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" value="2008-05-12" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-1-title">Title:</label></th><td><input type="text" name="form-1-title" id="id_form-1-title" /></td></tr>
<tr><th><label for="id_form-1-pub_date">Pub date:</label></th><td><input type="text" name="form-1-pub_date" id="id_form-1-pub_date" /></td></tr>
<tr><th><label for="id_form-2-title">Title:</label></th><td><input type="text" name="form-2-title" id="id_form-2-title" /></td></tr>
<tr><th><label for="id_form-2-pub_date">Pub date:</label></th><td><input type="text" name="form-2-pub_date" id="id_form-2-pub_date" /></td></tr>

Limiting the maximum number of forms¶

The max_num parameter to formset_factory() gives you the ability to limit the number of forms the formset will display:


>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm, extra=2, max_num=1)
>>> formset = ArticleFormSet()
>>> for form in formset:
...     print(form.as_table())

Formset validation¶

Validation with a formset is almost identical to a regular Form. There is an is_valid method on the formset to provide a convenient way to validate all forms in the formset:


>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm)
>>> data = {
...     'form-TOTAL_FORMS': '1',
...     'form-INITIAL_FORMS': '0',
...     'form-MAX_NUM_FORMS': '',
... }
>>> formset = ArticleFormSet(data)
>>> formset.is_valid()
True


We passed in no data to the formset which is resulting in a valid form. The formset is smart enough to ignore extra forms that were not changed. If we provide an invalid article:


>>> data = {
...     'form-TOTAL_FORMS': '2',
...     'form-INITIAL_FORMS': '0',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': 'Test',
...     'form-0-pub_date': '1904-06-16',
...     'form-1-title': 'Test',
...     'form-1-pub_date': '', # <-- this date is missing but required
... }
>>> formset = ArticleFormSet(data)
>>> formset.is_valid()
False
>>> formset.errors
[{}, {'pub_date': ['This field is required.']}]


As we can see, formset.errors is a list whose entries correspond to the forms in the formset. 
BaseFormSet.total_error_count()[source]¶
To check how many errors there are in the formset, we can use the total_error_count method:


>>> # Using the previous example
>>> formset.errors
[{}, {'pub_date': ['This field is required.']}]
>>> len(formset.errors)
2
>>> formset.total_error_count()
1


We can also check if form data differs from the initial data (i.e. the form was sent without any data):


>>> data = {
...     'form-TOTAL_FORMS': '1',
...     'form-INITIAL_FORMS': '0',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': '',
...     'form-0-pub_date': '',
... }
>>> formset = ArticleFormSet(data)
>>> formset.has_changed()
False



Understanding the ManagementForm¶

You may have noticed the additional data (form-TOTAL_FORMS, form-INITIAL_FORMS and form-MAX_NUM_FORMS) that was required in the formset’s data above. This data is required for the ManagementForm. This form is used by the formset to manage the collection of forms contained in the formset. If you don’t provide this management data, an exception will be raised:


>>> data = {
...     'form-0-title': 'Test',
...     'form-0-pub_date': '',
... }
>>> formset = ArticleFormSet(data)
>>> formset.is_valid()
Traceback (most recent call last):
...
django.forms.utils.ValidationError: ['ManagementForm data is missing or has been tampered with']



total_form_count and initial_form_count¶

BaseFormSet has a couple of methods that are closely related to the ManagementForm, total_form_count and initial_form_count.

total_form_count returns the total number of forms in this formset. initial_form_count returns the number of forms in the formset that were pre-filled, and is also used to determine how many forms are required. You will probably never need to override either of these methods, so please be sure you understand what they do before doing so.


empty_form¶

BaseFormSet provides an additional attribute empty_form which returns a form instance with a prefix of __prefix__ for easier use in dynamic forms with JavaScript.


Custom formset validation¶

A formset has a clean method similar to the one on a Form class. This is where you define your own validation that works at the formset level:


>>> from django.forms import BaseFormSet
>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm

>>> class BaseArticleFormSet(BaseFormSet):
...     def clean(self):
...         """Checks that no two articles have the same title."""
...         if any(self.errors):
...             # Don't bother validating the formset unless each form is valid on its own
...             return
...         titles = []
...         for form in self.forms:
...             title = form.cleaned_data['title']
...             if title in titles:
...                 raise forms.ValidationError("Articles in a set must have distinct titles.")
...             titles.append(title)

>>> ArticleFormSet = formset_factory(ArticleForm, formset=BaseArticleFormSet)
>>> data = {
...     'form-TOTAL_FORMS': '2',
...     'form-INITIAL_FORMS': '0',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': 'Test',
...     'form-0-pub_date': '1904-06-16',
...     'form-1-title': 'Test',
...     'form-1-pub_date': '1912-06-23',
... }
>>> formset = ArticleFormSet(data)
>>> formset.is_valid()
False
>>> formset.errors
[{}, {}]
>>> formset.non_form_errors()
['Articles in a set must have distinct titles.']


The formset clean method is called after all the Form.clean methods have been called. The errors will be found using the non_form_errors() method on the formset.


Validating the number of forms in a formset¶

Django provides a couple ways to validate the minimum or maximum number of submitted forms. Applications which need more customizable validation of the number of forms should use custom formset validation.


validate_max¶

If validate_max=True is passed to formset_factory(), validation will also check that the number of forms in the data set, minus those marked for deletion, is less than or equal to max_num.


>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm, max_num=1, validate_max=True)
>>> data = {
...     'form-TOTAL_FORMS': '2',
...     'form-INITIAL_FORMS': '0',
...     'form-MIN_NUM_FORMS': '',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': 'Test',
...     'form-0-pub_date': '1904-06-16',
...     'form-1-title': 'Test 2',
...     'form-1-pub_date': '1912-06-23',
... }
>>> formset = ArticleFormSet(data)
>>> formset.is_valid()
False
>>> formset.errors
[{}, {}]
>>> formset.non_form_errors()
['Please submit 1 or fewer forms.']


validate_max=True validates against max_num strictly even if max_num was exceeded because the amount of initial data supplied was excessive.
validate_min¶

If validate_min=True is passed to formset_factory(), validation will also check that the number of forms in the data set, minus those marked for deletion, is greater than or equal to min_num.


>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm, min_num=3, validate_min=True)
>>> data = {
...     'form-TOTAL_FORMS': '2',
...     'form-INITIAL_FORMS': '0',
...     'form-MIN_NUM_FORMS': '',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': 'Test',
...     'form-0-pub_date': '1904-06-16',
...     'form-1-title': 'Test 2',
...     'form-1-pub_date': '1912-06-23',
... }
>>> formset = ArticleFormSet(data)
>>> formset.is_valid()
False
>>> formset.errors
[{}, {}]
>>> formset.non_form_errors()
['Please submit 3 or more forms.']


Dealing with ordering and deletion of forms¶

The formset_factory() provides two optional parameters can_order and can_delete to help with ordering of forms in formsets and deletion of forms from a formset.


can_order¶
BaseFormSet.can_order¶
Default: False

Lets you create a formset with the ability to order:


>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm, can_order=True)
>>> formset = ArticleFormSet(initial=[
...     {'title': 'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': 'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
>>> for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" value="Article #1" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" value="2008-05-10" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-0-ORDER">Order:</label></th><td><input type="number" name="form-0-ORDER" value="1" id="id_form-0-ORDER" /></td></tr>
<tr><th><label for="id_form-1-title">Title:</label></th><td><input type="text" name="form-1-title" value="Article #2" id="id_form-1-title" /></td></tr>
<tr><th><label for="id_form-1-pub_date">Pub date:</label></th><td><input type="text" name="form-1-pub_date" value="2008-05-11" id="id_form-1-pub_date" /></td></tr>
<tr><th><label for="id_form-1-ORDER">Order:</label></th><td><input type="number" name="form-1-ORDER" value="2" id="id_form-1-ORDER" /></td></tr>
<tr><th><label for="id_form-2-title">Title:</label></th><td><input type="text" name="form-2-title" id="id_form-2-title" /></td></tr>
<tr><th><label for="id_form-2-pub_date">Pub date:</label></th><td><input type="text" name="form-2-pub_date" id="id_form-2-pub_date" /></td></tr>
<tr><th><label for="id_form-2-ORDER">Order:</label></th><td><input type="number" name="form-2-ORDER" id="id_form-2-ORDER" /></td></tr>


This adds an additional field to each form. This new field is named ORDER and is an forms.IntegerField. For the forms that came from the initial data it automatically assigned them a numeric value. Let’s look at what will happen when the user changes these values:


>>> data = {
...     'form-TOTAL_FORMS': '3',
...     'form-INITIAL_FORMS': '2',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': 'Article #1',
...     'form-0-pub_date': '2008-05-10',
...     'form-0-ORDER': '2',
...     'form-1-title': 'Article #2',
...     'form-1-pub_date': '2008-05-11',
...     'form-1-ORDER': '1',
...     'form-2-title': 'Article #3',
...     'form-2-pub_date': '2008-05-01',
...     'form-2-ORDER': '0',
... }

>>> formset = ArticleFormSet(data, initial=[
...     {'title': 'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': 'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
>>> formset.is_valid()
True
>>> for form in formset.ordered_forms:
...     print(form.cleaned_data)
{'pub_date': datetime.date(2008, 5, 1), 'ORDER': 0, 'title': 'Article #3'}
{'pub_date': datetime.date(2008, 5, 11), 'ORDER': 1, 'title': 'Article #2'}
{'pub_date': datetime.date(2008, 5, 10), 'ORDER': 2, 'title': 'Article #1'}



can_delete¶
BaseFormSet.can_delete¶
Default: False

Lets you create a formset with the ability to select forms for deletion:


>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> ArticleFormSet = formset_factory(ArticleForm, can_delete=True)
>>> formset = ArticleFormSet(initial=[
...     {'title': 'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': 'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
>>> for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" value="Article #1" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" value="2008-05-10" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-0-DELETE">Delete:</label></th><td><input type="checkbox" name="form-0-DELETE" id="id_form-0-DELETE" /></td></tr>
<tr><th><label for="id_form-1-title">Title:</label></th><td><input type="text" name="form-1-title" value="Article #2" id="id_form-1-title" /></td></tr>
<tr><th><label for="id_form-1-pub_date">Pub date:</label></th><td><input type="text" name="form-1-pub_date" value="2008-05-11" id="id_form-1-pub_date" /></td></tr>
<tr><th><label for="id_form-1-DELETE">Delete:</label></th><td><input type="checkbox" name="form-1-DELETE" id="id_form-1-DELETE" /></td></tr>
<tr><th><label for="id_form-2-title">Title:</label></th><td><input type="text" name="form-2-title" id="id_form-2-title" /></td></tr>
<tr><th><label for="id_form-2-pub_date">Pub date:</label></th><td><input type="text" name="form-2-pub_date" id="id_form-2-pub_date" /></td></tr>
<tr><th><label for="id_form-2-DELETE">Delete:</label></th><td><input type="checkbox" name="form-2-DELETE" id="id_form-2-DELETE" /></td></tr>


Similar to can_order this adds a new field to each form named DELETE and is a forms.BooleanField. When data comes through marking any of the delete fields you can access them with deleted_forms:


>>> data = {
...     'form-TOTAL_FORMS': '3',
...     'form-INITIAL_FORMS': '2',
...     'form-MAX_NUM_FORMS': '',
...     'form-0-title': 'Article #1',
...     'form-0-pub_date': '2008-05-10',
...     'form-0-DELETE': 'on',
...     'form-1-title': 'Article #2',
...     'form-1-pub_date': '2008-05-11',
...     'form-1-DELETE': '',
...     'form-2-title': '',
...     'form-2-pub_date': '',
...     'form-2-DELETE': '',
... }

>>> formset = ArticleFormSet(data, initial=[
...     {'title': 'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': 'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
>>> [form.cleaned_data for form in formset.deleted_forms]
[{'DELETE': True, 'pub_date': datetime.date(2008, 5, 10), 'title': 'Article #1'}]


If you are using a ModelFormSet, model instances for deleted forms will be deleted when you call formset.save().

Adding additional fields to a formset¶

If you need to add additional fields to the formset this can be easily accomplished. The formset base class provides an add_fields method. You can simply override this method to add your own fields or even redefine the default fields/attributes of the order and deletion fields:


>>> from django.forms import BaseFormSet
>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm
>>> class BaseArticleFormSet(BaseFormSet):
...     def add_fields(self, form, index):
...         super(BaseArticleFormSet, self).add_fields(form, index)
...         form.fields["my_field"] = forms.CharField()

>>> ArticleFormSet = formset_factory(ArticleForm, formset=BaseArticleFormSet)
>>> formset = ArticleFormSet()
>>> for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-0-my_field">My field:</label></th><td><input type="text" name="form-0-my_field" id="id_form-0-my_field" /></td></tr>



Passing custom parameters to formset forms¶

Sometimes your form class takes custom parameters, like MyArticleForm. You can pass this parameter when instantiating the formset:


>>> from django.forms import BaseFormSet
>>> from django.forms import formset_factory
>>> from myapp.forms import ArticleForm

>>> class MyArticleForm(ArticleForm):
...     def __init__(self, *args, **kwargs):
...         self.user = kwargs.pop('user')
...         super(MyArticleForm, self).__init__(*args, **kwargs)

>>> ArticleFormSet = formset_factory(MyArticleForm)
>>> formset = ArticleFormSet(form_kwargs={'user': request.user})


The form_kwargs may also depend on the specific form instance. The formset base class provides a get_form_kwargs method. The method takes a single argument - the index of the form in the formset. The index is None for the empty_form:


>>> from django.forms import BaseFormSet
>>> from django.forms import formset_factory

>>> class BaseArticleFormSet(BaseFormSet):
...     def get_form_kwargs(self, index):
...         kwargs = super(BaseArticleFormSet, self).get_form_kwargs(index)
...         kwargs['custom_kwarg'] = index
...         return kwargs

Using a formset in views and templates¶

Using a formset inside a view is as easy as using a regular Form class. The only thing you will want to be aware of is making sure to use the management form inside the template. Let’s look at a sample view:


from django.forms import formset_factory
from django.shortcuts import render
from myapp.forms import ArticleForm

def manage_articles(request):
    ArticleFormSet = formset_factory(ArticleForm)
    if request.method == 'POST':
        formset = ArticleFormSet(request.POST, request.FILES)
        if formset.is_valid():
            # do something with the formset.cleaned_data
            pass
    else:
        formset = ArticleFormSet()
    return render(request, 'manage_articles.html', {'formset': formset})


The manage_articles.html template might look like this:


<form method="post" action="">
    {{ formset.management_form }}
    <table>
        {% for form in formset %}
        {{ form }}
        {% endfor %}
    </table>
</form>


However there’s a slight shortcut for the above by letting the formset itself deal with the management form:


<form method="post" action="">
    <table>
        {{ formset }}
    </table>
</form>


The above ends up calling the as_table method on the formset class.


Manually rendered can_delete and can_order¶

If you manually render fields in the template, you can render can_delete parameter with {{ form.DELETE }}:


<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        <ul>
            <li>{{ form.title }}</li>
            <li>{{ form.pub_date }}</li>
            {% if formset.can_delete %}
                <li>{{ form.DELETE }}</li>
            {% endif %}
        </ul>
    {% endfor %}
</form>


Similarly, if the formset has the ability to order (can_order=True), it is possible to render it with {{ form.ORDER }}.


Using more than one formset in a view¶

You are able to use more than one formset in a view if you like. Formsets borrow much of its behavior from forms. With that said you are able to use prefix to prefix formset form field names with a given value to allow more than one formset to be sent to a view without name clashing. Lets take a look at how this might be accomplished:


from django.forms import formset_factory
from django.shortcuts import render
from myapp.forms import ArticleForm, BookForm

def manage_articles(request):
    ArticleFormSet = formset_factory(ArticleForm)
    BookFormSet = formset_factory(BookForm)
    if request.method == 'POST':
        article_formset = ArticleFormSet(request.POST, request.FILES, prefix='articles')
        book_formset = BookFormSet(request.POST, request.FILES, prefix='books')
        if article_formset.is_valid() and book_formset.is_valid():
            # do something with the cleaned_data on the formsets.
            pass
    else:
        article_formset = ArticleFormSet(prefix='articles')
        book_formset = BookFormSet(prefix='books')
    return render(request, 'manage_articles.html', {
        'article_formset': article_formset,
        'book_formset': book_formset,
    })


You would then render the formsets as normal. It is important to point out that you need to pass prefix on both the POST and non-POST cases so that it is rendered and processed correctly.
@@@
Model formsets¶
class models.BaseModelFormSet¶
Like regular formsets, Django provides a couple of enhanced formset classes that make it easy to work with Django models. Let’s reuse the Author model from above:


>>> from django.forms import modelformset_factory
>>> from myapp.models import Author
>>> AuthorFormSet = modelformset_factory(Author, fields=('name', 'title'))


Using fields restricts the formset to use only the given fields. Alternatively, you can take an “opt-out” approach, specifying which fields to exclude:


>>> AuthorFormSet = modelformset_factory(Author, exclude=('birth_date',))


This will create a formset that is capable of working with the data associated with the Author model. It works just like a regular formset:


>>> formset = AuthorFormSet()
>>> print(formset)
<input type="hidden" name="form-TOTAL_FORMS" value="1" id="id_form-TOTAL_FORMS" /><input type="hidden" name="form-INITIAL_FORMS" value="0" id="id_form-INITIAL_FORMS" /><input type="hidden" name="form-MAX_NUM_FORMS" id="id_form-MAX_NUM_FORMS" />
<tr><th><label for="id_form-0-name">Name:</label></th><td><input id="id_form-0-name" type="text" name="form-0-name" maxlength="100" /></td></tr>
<tr><th><label for="id_form-0-title">Title:</label></th><td><select name="form-0-title" id="id_form-0-title">
<option value="" selected="selected">---------</option>
<option value="MR">Mr.</option>
<option value="MRS">Mrs.</option>
<option value="MS">Ms.</option>
</select><input type="hidden" name="form-0-id" id="id_form-0-id" /></td></tr>


  
    
Changing the queryset¶

By default, when you create a formset from a model, the formset will use a queryset that includes all objects in the model (e.g., Author.objects.all()). You can override this behavior by using the queryset argument:


>>> formset = AuthorFormSet(queryset=Author.objects.filter(name__startswith='O'))


Alternatively, you can create a subclass that sets self.queryset in __init__:


from django.forms import BaseModelFormSet
from myapp.models import Author

class BaseAuthorFormSet(BaseModelFormSet):
    def __init__(self, *args, **kwargs):
        super(BaseAuthorFormSet, self).__init__(*args, **kwargs)
        self.queryset = Author.objects.filter(name__startswith='O')


Then, pass your BaseAuthorFormSet class to the factory function:


>>> AuthorFormSet = modelformset_factory(
...     Author, fields=('name', 'title'), formset=BaseAuthorFormSet)


If you want to return a formset that doesn’t include any pre-existing instances of the model, you can specify an empty QuerySet:


>>> AuthorFormSet(queryset=Author.objects.none())



Changing the form¶

By default, when you use modelformset_factory, a model form will be created using modelform_factory(). Often, it can be useful to specify a custom model form. For example, you can create a custom model form that has custom validation:


class AuthorForm(forms.ModelForm):
    class Meta:
        model = Author
        fields = ('name', 'title')

    def clean_name(self):
        # custom validation for the name field
        ...


Then, pass your model form to the factory function:


AuthorFormSet = modelformset_factory(Author, form=AuthorForm)


It is not always necessary to define a custom model form. The modelformset_factory function has several arguments which are passed through to modelform_factory, which are described below.


Specifying widgets to use in the form with widgets¶

Using the widgets parameter, you can specify a dictionary of values to customize the ModelForm’s widget class for a particular field. This works the same way as the widgets dictionary on the inner Meta class of a ModelForm works:


>>> AuthorFormSet = modelformset_factory(
...     Author, fields=('name', 'title'),
...     widgets={'name': Textarea(attrs={'cols': 80, 'rows': 20})})



Enabling localization for fields with localized_fields¶

Using the localized_fields parameter, you can enable localization for fields in the form.


>>> AuthorFormSet = modelformset_factory(
...     Author, fields=('name', 'title', 'birth_date'),
...     localized_fields=('birth_date',))


If localized_fields is set to the special value '__all__', all fields will be localized.


Providing initial values¶

As with regular formsets, it’s possible to specify initial data for forms in the formset by specifying an initial parameter when instantiating the model formset class returned by modelformset_factory(). However, with model formsets, the initial values only apply to extra forms, those that aren’t attached to an existing model instance. If the extra forms with initial data aren’t changed by the user, they won’t be validated or saved.


Saving objects in the formset¶

As with a ModelForm, you can save the data as a model object. This is done with the formset’s save() method:


# Create a formset instance with POST data.
>>> formset = AuthorFormSet(request.POST)

# Assuming all is valid, save the data.
>>> instances = formset.save()


After calling save(), your model formset will have three new attributes containing the formset’s changes:
models.BaseModelFormSet.changed_objects¶models.BaseModelFormSet.deleted_objects¶models.BaseModelFormSet.new_objects¶

Limiting the number of editable objects¶

As with regular formsets, you can use the max_num and extra parameters to modelformset_factory() to limit the number of extra forms displayed.

max_num does not prevent existing objects from being displayed:


>>> Author.objects.order_by('name')
<QuerySet [<Author: Charles Baudelaire>, <Author: Paul Verlaine>, <Author: Walt Whitman>]>

>>> AuthorFormSet = modelformset_factory(Author, fields=('name',), max_num=1)
>>> formset = AuthorFormSet(queryset=Author.objects.order_by('name'))
>>> [x.name for x in formset.get_queryset()]
['Charles Baudelaire', 'Paul Verlaine', 'Walt Whitman']


Also, extra=0 doesn’t prevent creation of new model instances as you can add additional forms with JavaScript or just send additional POST data. Formsets don’t yet provide functionality for an “edit only” view that prevents creation of new instances.

If the value of max_num is greater than the number of existing related objects, up to extra additional blank forms will be added to the formset, so long as the total number of forms does not exceed max_num:


>>> AuthorFormSet = modelformset_factory(Author, fields=('name',), max_num=4, extra=2)
>>> formset = AuthorFormSet(queryset=Author.objects.order_by('name'))
>>> for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-name">Name:</label></th><td><input id="id_form-0-name" type="text" name="form-0-name" value="Charles Baudelaire" maxlength="100" /><input type="hidden" name="form-0-id" value="1" id="id_form-0-id" /></td></tr>
<tr><th><label for="id_form-1-name">Name:</label></th><td><input id="id_form-1-name" type="text" name="form-1-name" value="Paul Verlaine" maxlength="100" /><input type="hidden" name="form-1-id" value="3" id="id_form-1-id" /></td></tr>
<tr><th><label for="id_form-2-name">Name:</label></th><td><input id="id_form-2-name" type="text" name="form-2-name" value="Walt Whitman" maxlength="100" /><input type="hidden" name="form-2-id" value="2" id="id_form-2-id" /></td></tr>
<tr><th><label for="id_form-3-name">Name:</label></th><td><input id="id_form-3-name" type="text" name="form-3-name" maxlength="100" /><input type="hidden" name="form-3-id" id="id_form-3-id" /></td></tr>


A max_num value of None (the default) puts a high limit on the number of forms displayed (1000). In practice this is equivalent to no limit.


Using a model formset in a view¶

Model formsets are very similar to formsets. Let’s say we want to present a formset to edit Author model instances:


from django.forms import modelformset_factory
from django.shortcuts import render
from myapp.models import Author

def manage_authors(request):
    AuthorFormSet = modelformset_factory(Author, fields=('name', 'title'))
    if request.method == 'POST':
        formset = AuthorFormSet(request.POST, request.FILES)
        if formset.is_valid():
            formset.save()
            # do something.
    else:
        formset = AuthorFormSet()
    return render(request, 'manage_authors.html', {'formset': formset})


As you can see, the view logic of a model formset isn’t drastically different than that of a “normal” formset. The only difference is that we call formset.save() to save the data into the database. (This was described above, in Saving objects in the formset.)


Overriding clean() on a ModelFormSet¶

Just like with ModelForms, by default the clean() method of a ModelFormSet will validate that none of the items in the formset violate the unique constraints on your model (either unique, unique_together or unique_for_date|month|year). If you want to override the clean() method on a ModelFormSet and maintain this validation, you must call the parent class’s clean method:


from django.forms import BaseModelFormSet

class MyModelFormSet(BaseModelFormSet):
    def clean(self):
        super(MyModelFormSet, self).clean()
        # example custom validation across forms in the formset
        for form in self.forms:
            # your custom formset validation
            ...


Also note that by the time you reach this step, individual model instances have already been created for each Form. Modifying a value in form.cleaned_data is not sufficient to affect the saved value. If you wish to modify a value in ModelFormSet.clean() you must modify form.instance:


from django.forms import BaseModelFormSet

class MyModelFormSet(BaseModelFormSet):
    def clean(self):
        super(MyModelFormSet, self).clean()

        for form in self.forms:
            name = form.cleaned_data['name'].upper()
            form.cleaned_data['name'] = name
            # update the instance value.
            form.instance.name = name



Using a custom queryset¶

As stated earlier, you can override the default queryset used by the model formset:


from django.forms import modelformset_factory
from django.shortcuts import render
from myapp.models import Author

def manage_authors(request):
    AuthorFormSet = modelformset_factory(Author, fields=('name', 'title'))
    if request.method == "POST":
        formset = AuthorFormSet(
            request.POST, request.FILES,
            queryset=Author.objects.filter(name__startswith='O'),
        )
        if formset.is_valid():
            formset.save()
            # Do something.
    else:
        formset = AuthorFormSet(queryset=Author.objects.filter(name__startswith='O'))
    return render(request, 'manage_authors.html', {'formset': formset})


Note that we pass the queryset argument in both the POST and GET cases in this example.


Using the formset in the template¶

There are three ways to render a formset in a Django template.

First, you can let the formset do most of the work:


<form method="post" action="">
    {{ formset }}
</form>


Second, you can manually render the formset, but let the form deal with itself:


<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        {{ form }}
    {% endfor %}
</form>


When you manually render the forms yourself, be sure to render the management form as shown above. See the management form documentation.

Third, you can manually render each field:


<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        {% for field in form %}
            {{ field.label_tag }} {{ field }}
        {% endfor %}
    {% endfor %}
</form>


If you opt to use this third method and you don’t iterate over the fields with a {% for %} loop, you’ll need to render the primary key field. For example, if you were rendering the name and age fields of a model:


<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        {{ form.id }}
        <ul>
            <li>{{ form.name }}</li>
            <li>{{ form.age }}</li>
        </ul>
    {% endfor %}
</form>


Notice how we need to explicitly render {{ form.id }}. This ensures that the model formset, in the POST case, will work correctly. (This example assumes a primary key named id. If you’ve explicitly defined your own primary key that isn’t called id, make sure it gets rendered.)
Inline formsets¶
class models.BaseInlineFormSet¶
Inline formsets is a small abstraction layer on top of model formsets. These simplify the case of working with related objects via a foreign key. Suppose you have these two models:


from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)


If you want to create a formset that allows you to edit books belonging to a particular author, you could do this:


>>> from django.forms import inlineformset_factory
>>> BookFormSet = inlineformset_factory(Author, Book, fields=('title',))
>>> author = Author.objects.get(name='Mike Royko')
>>> formset = BookFormSet(instance=author)


Overriding methods on an InlineFormSet¶

When overriding methods on InlineFormSet, you should subclass BaseInlineFormSet rather than BaseModelFormSet.

For example, if you want to override clean():


from django.forms import BaseInlineFormSet

class CustomInlineFormSet(BaseInlineFormSet):
    def clean(self):
        super(CustomInlineFormSet, self).clean()
        # example custom validation across forms in the formset
        for form in self.forms:
            # your custom formset validation
            ...


See also Overriding clean() on a ModelFormSet.

Then when you create your inline formset, pass in the optional argument formset:


>>> from django.forms import inlineformset_factory
>>> BookFormSet = inlineformset_factory(Author, Book, fields=('title',),
...     formset=CustomInlineFormSet)
>>> author = Author.objects.get(name='Mike Royko')
>>> formset = BookFormSet(instance=author)



More than one foreign key to the same model¶

If your model contains more than one foreign key to the same model, you’ll need to resolve the ambiguity manually using fk_name. For example, consider the following model:


class Friendship(models.Model):
    from_friend = models.ForeignKey(
        Friend,
        on_delete=models.CASCADE,
        related_name='from_friends',
    )
    to_friend = models.ForeignKey(
        Friend,
        on_delete=models.CASCADE,
        related_name='friends',
    )
    length_in_months = models.IntegerField()


To resolve this, you can use fk_name to inlineformset_factory():


>>> FriendshipFormSet = inlineformset_factory(Friend, Friendship, fk_name='from_friend',
...     fields=('to_friend', 'length_in_months'))



Using an inline formset in a view¶

You may want to provide a view that allows a user to edit the related objects of a model. Here’s how you can do that:


def manage_books(request, author_id):
    author = Author.objects.get(pk=author_id)
    BookInlineFormSet = inlineformset_factory(Author, Book, fields=('title',))
    if request.method == "POST":
        formset = BookInlineFormSet(request.POST, request.FILES, instance=author)
        if formset.is_valid():
            formset.save()
            # Do something. Should generally end with a redirect. For example:
            return HttpResponseRedirect(author.get_absolute_url())
    else:
        formset = BookInlineFormSet(instance=author)
    return render(request, 'manage_books.html', {'formset': formset})


    
    
    
    


@@@


https://docs.djangoproject.com/en/1.10/topics/class-based-views/
Class-based views¶

A view is a callable which takes a request and returns a response


Simple usage in your URLconf¶

The simplest way to use generic views is to create them directly in your URLconf. If you’re only changing a few simple attributes on a class-based view, you can simply pass them into the as_view() method call itself:


from django.conf.urls import url
from django.views.generic import TemplateView

urlpatterns = [
    url(r'^about/$', TemplateView.as_view(template_name="about.html")),
]


Any arguments passed to as_view() will override attributes set on the class. In this example, we set template_name on the TemplateView. A similar overriding pattern can be used for the url attribute on RedirectView.


Subclassing generic views¶

The second, more powerful way to use generic views is to inherit from an existing view and override attributes (such as the template_name) or methods (such as get_context_data) in your subclass to provide new values or methods. Consider, for example, a view that just displays one template, about.html. Django has a generic view to do this - TemplateView - so we can just subclass it, and override the template name:


# some_app/views.py
from django.views.generic import TemplateView

class AboutView(TemplateView):
    template_name = "about.html"


Then we just need to add this new view into our URLconf. TemplateView is a class, not a function, so we point the URL to the as_view() class method instead, which provides a function-like entry to class-based views:


# urls.py
from django.conf.urls import url
from some_app.views import AboutView

urlpatterns = [
    url(r'^about/$', AboutView.as_view()),
]


For more information on how to use the built in generic views, consult the next topic on generic class-based views.


Supporting other HTTP methods¶

Suppose somebody wants to access our book library over HTTP using the views as an API. The API client would connect every now and then and download book data for the books published since last visit. But if no new books appeared since then, it is a waste of CPU time and bandwidth to fetch the books from the database, render a full response and send it to the client. It might be preferable to ask the API when the most recent book was published.

We map the URL to book list view in the URLconf:


from django.conf.urls import url
from books.views import BookListView

urlpatterns = [
    url(r'^books/$', BookListView.as_view()),
]



https://docs.djangoproject.com/en/1.10/topics/class-based-views/intro/

Using class-based views¶

At its core, a class-based view allows you to respond to different HTTP request methods with different class instance methods, instead of with conditionally branching code inside a single view function.

So where the code to handle HTTP GET in a view function would look something like:


from django.http import HttpResponse

def my_view(request):
    if request.method == 'GET':
        # <view logic>
        return HttpResponse('result')


In a class-based view, this would become:


from django.http import HttpResponse
from django.views import View

class MyView(View):
    def get(self, request):
        # <view logic>
        return HttpResponse('result')


Because Django’s URL resolver expects to send the request and associated arguments to a callable function, not a class, class-based views have an as_view() class method which serves as the callable entry point to your class. The as_view entry point creates an instance of your class and calls its dispatch() method. dispatch looks at the request to determine whether it is a GET, POST, etc, and relays the request to a matching method if one is defined, or raises HttpResponseNotAllowed if not:


# urls.py
from django.conf.urls import url
from myapp.views import MyView

urlpatterns = [
    url(r'^about/$', MyView.as_view()),
]

 there are two ways to configure or set class attributes.

The first is the standard Python way of subclassing and overriding attributes and methods in the subclass. So that if your parent class had an attribute greeting like this:


from django.http import HttpResponse
from django.views import View

class GreetingView(View):
    greeting = "Good Day"

    def get(self, request):
        return HttpResponse(self.greeting)


You can override that in a subclass:


class MorningGreetingView(GreetingView):
    greeting = "Morning to ya"


Another option is to configure class attributes as keyword arguments to the as_view() call in the URLconf:


urlpatterns = [
    url(r'^about/$', GreetingView.as_view(greeting="G'day")),
]


Handling forms with class-based views¶

A basic function-based view that handles forms may look something like this:


from django.http import HttpResponseRedirect
from django.shortcuts import render

from .forms import MyForm

def myview(request):
    if request.method == "POST":
        form = MyForm(request.POST)
        if form.is_valid():
            # <process form cleaned data>
            return HttpResponseRedirect('/success/')
    else:
        form = MyForm(initial={'key': 'value'})

    return render(request, 'form_template.html', {'form': form})


A similar class-based view might look like:


from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views import View

from .forms import MyForm

class MyFormView(View):
    form_class = MyForm
    initial = {'key': 'value'}
    template_name = 'form_template.html'

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            # <process form cleaned data>
            return HttpResponseRedirect('/success/')

        return render(request, self.template_name, {'form': form})


This is a very simple case, but you can see that you would then have the option of customizing this view by overriding any of the class attributes, e.g. form_class, via URLconf configuration, or subclassing and overriding one or more of the methods (or both!).


Decorating class-based views¶

The extension of class-based views isn’t limited to using mixins. You can also use decorators. Since class-based views aren’t functions, decorating them works differently depending on if you’re using as_view() or creating a subclass.


Decorating in URLconf¶

The simplest way of decorating class-based views is to decorate the result of the as_view() method. The easiest place to do this is in the URLconf where you deploy your view:


from django.contrib.auth.decorators import login_required, permission_required
from django.views.generic import TemplateView

from .views import VoteView

urlpatterns = [
    url(r'^about/$', login_required(TemplateView.as_view(template_name="secret.html"))),
    url(r'^vote/$', permission_required('polls.can_vote')(VoteView.as_view())),
]


This approach applies the decorator on a per-instance basis. If you want every instance of a view to be decorated, you need to take a different approach.


Decorating the class¶

To decorate every instance of a class-based view, you need to decorate the class definition itself. To do this you apply the decorator to the dispatch() method of the class.

A method on a class isn’t quite the same as a standalone function, so you can’t just apply a function decorator to the method – you need to transform it into a method decorator first. The method_decorator decorator transforms a function decorator into a method decorator so that it can be used on an instance method. For example:


from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView

class ProtectedView(TemplateView):
    template_name = 'secret.html'

    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super(ProtectedView, self).dispatch(*args, **kwargs)


Or, more succinctly, you can decorate the class instead and pass the name of the method to be decorated as the keyword argument name:


@method_decorator(login_required, name='dispatch')
class ProtectedView(TemplateView):
    template_name = 'secret.html'


If you have a set of common decorators used in several places, you can define a list or tuple of decorators and use this instead of invoking method_decorator() multiple times. These two classes are equivalent:


decorators = [never_cache, login_required]

@method_decorator(decorators, name='dispatch')
class ProtectedView(TemplateView):
    template_name = 'secret.html'

@method_decorator(never_cache, name='dispatch')
@method_decorator(login_required, name='dispatch')
class ProtectedView(TemplateView):
    template_name = 'secret.html'


The decorators will process a request in the order they are passed to the decorator. In the example, never_cache() will process the request before login_required().

Changed in Django 1.9: 
The ability to use method_decorator() on a class and the ability for it to accept a list or tuple of decorators were added.

In this example, every instance of ProtectedView will have login protection.

Built-in class-based views API¶

Class-based views API reference. For introductory material, see the Class-based views topic guide.

•Base views ?View
?TemplateView
?RedirectView

•Generic display views ?DetailView
?ListView

•Generic editing views ?FormView
?CreateView
?UpdateView
?DeleteView

•Generic date views ?ArchiveIndexView
?YearArchiveView
?MonthArchiveView
?WeekArchiveView
?DayArchiveView
?TodayArchiveView
?DateDetailView

•Class-based views mixins ?Simple mixins ?ContextMixin
?TemplateResponseMixin

?Single object mixins ?SingleObjectMixin
?SingleObjectTemplateResponseMixin

?Multiple object mixins ?MultipleObjectMixin
?MultipleObjectTemplateResponseMixin

?Editing mixins ?FormMixin
?ModelFormMixin
?ProcessFormView
?DeletionMixin

?Date-based mixins ?YearMixin
?MonthMixin
?DayMixin
?WeekMixin
?DateMixin
?BaseDateListView


•Class-based generic views - flattened index ?Simple generic views ?View
?TemplateView
?RedirectView

?Detail Views ?DetailView

?List Views ?ListView

?Editing views ?FormView
?CreateView
?UpdateView
?DeleteView

?Date-based views ?ArchiveIndexView
?YearArchiveView
?MonthArchiveView
?WeekArchiveView
?DayArchiveView
?TodayArchiveView
?DateDetailView




Specification¶

Each request served by a class-based view has an independent state; therefore, it is safe to store state variables on the instance (i.e., self.foo = 3 is a thread-safe operation).

A class-based view is deployed into a URL pattern using the as_view() classmethod:


urlpatterns = [
    url(r'^view/$', MyView.as_view(size=42)),
]


https://docs.djangoproject.com/en/1.10/ref/class-based-views/base/

Base views
View¶
class django.views.generic.base.View¶
The master class-based base view. All other class-based views inherit from this base class. It isn’t strictly a generic view and thus can also be imported from django.views.

Changed in Django 1.10: 
The ability to import from django.views was added.

Method Flowchart
1.dispatch()
2.http_method_not_allowed()
3.options()

Example views.py:


from django.http import HttpResponse
from django.views import View

class MyView(View):

    def get(self, request, *args, **kwargs):
        return HttpResponse('Hello, World!')


Example urls.py:


from django.conf.urls import url

from myapp.views import MyView

urlpatterns = [
    url(r'^mine/$', MyView.as_view(), name='my-view'),
]

TemplateView¶
class django.views.generic.base.TemplateView¶
Renders a given template, with the context containing parameters captured in the URL.

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.base.ContextMixin
•django.views.generic.base.View

Method Flowchart
1.dispatch()
2.http_method_not_allowed()
3.get_context_data()

Example views.py:


from django.views.generic.base import TemplateView

from articles.models import Article

class HomePageView(TemplateView):

    template_name = "home.html"

    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        context['latest_articles'] = Article.objects.all()[:5]
        return context


Example urls.py:


from django.conf.urls import url

from myapp.views import HomePageView

urlpatterns = [
    url(r'^$', HomePageView.as_view(), name='home'),
]

RedirectView¶
class django.views.generic.base.RedirectView¶
Redirects to a given URL.

The given URL may contain dictionary-style string formatting, which will be interpolated against the parameters captured in the URL. Because keyword interpolation is always done (even if no arguments are passed in), any "%" characters in the URL must be written as "%%" so that Python will convert them to a single percent sign on output.

If the given URL is None, Django will return an HttpResponseGone (410).

Ancestors (MRO)

This view inherits methods and attributes from the following view:
•django.views.generic.base.View

Method Flowchart
1.dispatch()
2.http_method_not_allowed()
3.get_redirect_url()

Example views.py:


from django.shortcuts import get_object_or_404
from django.views.generic.base import RedirectView

from articles.models import Article

class ArticleCounterRedirectView(RedirectView):

    permanent = False
    query_string = True
    pattern_name = 'article-detail'

    def get_redirect_url(self, *args, **kwargs):
        article = get_object_or_404(Article, pk=kwargs['pk'])
        article.update_counter()
        return super(ArticleCounterRedirectView, self).get_redirect_url(*args, **kwargs)


Example urls.py:


from django.conf.urls import url
from django.views.generic.base import RedirectView

from article.views import ArticleCounterRedirectView, ArticleDetail

urlpatterns = [
    url(r'^counter/(?P<pk>[0-9]+)/$', ArticleCounterRedirectView.as_view(), name='article-counter'),
    url(r'^details/(?P<pk>[0-9]+)/$', ArticleDetail.as_view(), name='article-detail'),
    url(r'^go-to-django/$', RedirectView.as_view(url='https://djangoproject.com'), name='go-to-django'),
]




https://docs.djangoproject.com/en/1.10/topics/class-based-views/generic-display/
Generic views of objects¶

TemplateView certainly is useful, but Django’s generic views really shine when it comes to presenting views of your database content. Because it’s such a common task, Django comes with a handful of built-in generic views that make generating list and detail views of objects incredibly easy.

Let’s start by looking at some examples of showing a list of objects or an individual object.

We’ll be using these models:


# models.py
from django.db import models

class Publisher(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    state_province = models.CharField(max_length=30)
    country = models.CharField(max_length=50)
    website = models.URLField()

    class Meta:
        ordering = ["-name"]

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Author(models.Model):
    salutation = models.CharField(max_length=10)
    name = models.CharField(max_length=200)
    email = models.EmailField()
    headshot = models.ImageField(upload_to='author_headshots')

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Book(models.Model):
    title = models.CharField(max_length=100)
    authors = models.ManyToManyField('Author')
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)
    publication_date = models.DateField()


Now we need to define a view:


# views.py
from django.views.generic import ListView
from books.models import Publisher

class PublisherList(ListView):
    model = Publisher


Finally hook that view into your urls:


# urls.py
from django.conf.urls import url
from books.views import PublisherList

urlpatterns = [
    url(r'^publishers/$', PublisherList.as_view()),
]


That’s all the Python code we need to write. We still need to write a template, however. We could explicitly tell the view which template to use by adding a template_name attribute to the view, but in the absence of an explicit template Django will infer one from the object’s name. In this case, the inferred template will be "books/publisher_list.html" – the “books” part comes from the name of the app that defines the model, while the “publisher” bit is just the lowercased version of the model’s name.


?Note

Thus, when (for example) the APP_DIRS option of a DjangoTemplates backend is set to True in TEMPLATES, a template location could be: /path/to/project/books/templates/books/publisher_list.html

This template will be rendered against a context containing a variable called object_list that contains all the publisher objects. A very simple template might look like the following:


{% extends "base.html" %}

{% block content %}
    <h2>Publishers</h2>
    <ul>
        {% for publisher in object_list %}
            <li>{{ publisher.name }}</li>
        {% endfor %}
    </ul>
{% endblock %}


Making “friendly” template contexts¶

You might have noticed that our sample publisher list template stores all the publishers in a variable named object_list. While this works just fine, it isn’t all that “friendly” to template authors: they have to “just know” that they’re dealing with publishers here.

Well, if you’re dealing with a model object, this is already done for you. When you are dealing with an object or queryset, Django is able to populate the context using the lower cased version of the model class’ name. This is provided in addition to the default object_list entry, but contains exactly the same data, i.e. publisher_list.

If this still isn’t a good match, you can manually set the name of the context variable. The context_object_name attribute on a generic view specifies the context variable to use:


# views.py
from django.views.generic import ListView
from books.models import Publisher

class PublisherList(ListView):
    model = Publisher
    context_object_name = 'my_favorite_publishers'


Providing a useful context_object_name is always a good idea. Your coworkers who design templates will thank you.


Adding extra context¶

Often you simply need to present some extra information beyond that provided by the generic view. For example, think of showing a list of all the books on each publisher detail page. The DetailView generic view provides the publisher to the context, but how do we get additional information in that template?

The answer is to subclass DetailView and provide your own implementation of the get_context_data method. The default implementation simply adds the object being displayed to the template, but you can override it to send more:


from django.views.generic import DetailView
from books.models import Publisher, Book

class PublisherDetail(DetailView):

    model = Publisher

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(PublisherDetail, self).get_context_data(**kwargs)
        # Add in a QuerySet of all the books
        context['book_list'] = Book.objects.all()
        return context

Viewing subsets of objects¶

Now let’s take a closer look at the model argument we’ve been using all along. The model argument, which specifies the database model that the view will operate upon, is available on all the generic views that operate on a single object or a collection of objects. However, the model argument is not the only way to specify the objects that the view will operate upon – you can also specify the list of objects using the queryset argument:


from django.views.generic import DetailView
from books.models import Publisher

class PublisherDetail(DetailView):

    context_object_name = 'publisher'
    queryset = Publisher.objects.all()


Specifying model = Publisher is really just shorthand for saying queryset = Publisher.objects.all(). However, by using queryset to define a filtered list of objects you can be more specific about the objects that will be visible in the view (see Making queries for more information about QuerySet objects, and see the class-based views reference for the complete details).

To pick a simple example, we might want to order a list of books by publication date, with the most recent first:


from django.views.generic import ListView
from books.models import Book

class BookList(ListView):
    queryset = Book.objects.order_by('-publication_date')
    context_object_name = 'book_list'


That’s a pretty simple example, but it illustrates the idea nicely. Of course, you’ll usually want to do more than just reorder objects. If you want to present a list of books by a particular publisher, you can use the same technique:


from django.views.generic import ListView
from books.models import Book

class AcmeBookList(ListView):

    context_object_name = 'book_list'
    queryset = Book.objects.filter(publisher__name='ACME Publishing')
    template_name = 'books/acme_list.html'


Notice that along with a filtered queryset, we’re also using a custom template name. If we didn’t, the generic view would use the same template as the “vanilla” object list, which might not be what we want.

Dynamic filtering¶

Another common need is to filter down the objects given in a list page by some key in the URL. Earlier we hard-coded the publisher’s name in the URLconf, but what if we wanted to write a view that displayed all the books by some arbitrary publisher?

Handily, the ListView has a get_queryset() method we can override. Previously, it has just been returning the value of the queryset attribute, but now we can add more logic.

The key part to making this work is that when class-based views are called, various useful things are stored on self; as well as the request (self.request) this includes the positional (self.args) and name-based (self.kwargs) arguments captured according to the URLconf.

Here, we have a URLconf with a single captured group:


# urls.py
from django.conf.urls import url
from books.views import PublisherBookList

urlpatterns = [
    url(r'^books/([\w-]+)/$', PublisherBookList.as_view()),
]


Next, we’ll write the PublisherBookList view itself:


# views.py
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from books.models import Book, Publisher

class PublisherBookList(ListView):

    template_name = 'books/books_by_publisher.html'

    def get_queryset(self):
        self.publisher = get_object_or_404(Publisher, name=self.args[0])
        return Book.objects.filter(publisher=self.publisher)


As you can see, it’s quite easy to add more logic to the queryset selection; if we wanted, we could use self.request.user to filter using the current user, or other more complex logic.

We can also add the publisher into the context at the same time, so we can use it in the template:


# ...

def get_context_data(self, **kwargs):
    # Call the base implementation first to get a context
    context = super(PublisherBookList, self).get_context_data(**kwargs)
    # Add in the publisher
    context['publisher'] = self.publisher
    return context



Performing extra work¶

The last common pattern we’ll look at involves doing some extra work before or after calling the generic view.

Imagine we had a last_accessed field on our Author model that we were using to keep track of the last time anybody looked at that author:


# models.py
from django.db import models

class Author(models.Model):
    salutation = models.CharField(max_length=10)
    name = models.CharField(max_length=200)
    email = models.EmailField()
    headshot = models.ImageField(upload_to='author_headshots')
    last_accessed = models.DateTimeField()


The generic DetailView class, of course, wouldn’t know anything about this field, but once again we could easily write a custom view to keep that field updated.

First, we’d need to add an author detail bit in the URLconf to point to a custom view:


from django.conf.urls import url
from books.views import AuthorDetailView

urlpatterns = [
    #...
    url(r'^authors/(?P<pk>[0-9]+)/$', AuthorDetailView.as_view(), name='author-detail'),
]


Then we’d write our new view – get_object is the method that retrieves the object – so we simply override it and wrap the call:


from django.views.generic import DetailView
from django.utils import timezone
from books.models import Author

class AuthorDetailView(DetailView):

    queryset = Author.objects.all()

    def get_object(self):
        # Call the superclass
        object = super(AuthorDetailView, self).get_object()
        # Record the last accessed date
        object.last_accessed = timezone.now()
        object.save()
        # Return the object
        return object


https://docs.djangoproject.com/en/1.10/ref/class-based-views/generic-display/

DetailView¶
class django.views.generic.detail.DetailView¶
While this view is executing, self.object will contain the object that the view is operating upon.

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.detail.SingleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.detail.BaseDetailView
•django.views.generic.detail.SingleObjectMixin
•django.views.generic.base.View

Method Flowchart
1.dispatch()
2.http_method_not_allowed()
3.get_template_names()
4.get_slug_field()
5.get_queryset()
6.get_object()
7.get_context_object_name()
8.get_context_data()
9.get()
10.render_to_response()

Example myapp/views.py:


from django.views.generic.detail import DetailView
from django.utils import timezone

from articles.models import Article

class ArticleDetailView(DetailView):

    model = Article

    def get_context_data(self, **kwargs):
        context = super(ArticleDetailView, self).get_context_data(**kwargs)
        context['now'] = timezone.now()
        return context


Example myapp/urls.py:


from django.conf.urls import url

from article.views import ArticleDetailView

urlpatterns = [
    url(r'^(?P<slug>[-\w]+)/$', ArticleDetailView.as_view(), name='article-detail'),
]


Example myapp/article_detail.html:


<h1>{{ object.headline }}</h1>
<p>{{ object.content }}</p>
<p>Reporter: {{ object.reporter }}</p>
<p>Published: {{ object.pub_date|date }}</p>
<p>Date: {{ now|date }}</p>



ListView¶
class django.views.generic.list.ListView¶
A page representing a list of objects.

While this view is executing, self.object_list will contain the list of objects (usually, but not necessarily a queryset) that the view is operating upon.

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.list.MultipleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.list.BaseListView
•django.views.generic.list.MultipleObjectMixin
•django.views.generic.base.View

Method Flowchart
1.dispatch()
2.http_method_not_allowed()
3.get_template_names()
4.get_queryset()
5.get_context_object_name()
6.get_context_data()
7.get()
8.render_to_response()

Example views.py:


from django.views.generic.list import ListView
from django.utils import timezone

from articles.models import Article

class ArticleListView(ListView):

    model = Article

    def get_context_data(self, **kwargs):
        context = super(ArticleListView, self).get_context_data(**kwargs)
        context['now'] = timezone.now()
        return context


Example myapp/urls.py:


from django.conf.urls import url

from article.views import ArticleListView

urlpatterns = [
    url(r'^$', ArticleListView.as_view(), name='article-list'),
]


Example myapp/article_list.html:


<h1>Articles</h1>
<ul>
{% for article in object_list %}
    <li>{{ article.pub_date|date }} - {{ article.headline }}</li>
{% empty %}
    <li>No articles yet.</li>
{% endfor %}
</ul>



https://docs.djangoproject.com/en/1.10/topics/class-based-views/generic-editing/
Form handling with class-based views¶

Form processing generally has 3 paths:
•Initial GET (blank or prepopulated form)
•POST with invalid data (typically redisplay form with errors)
•POST with valid data (process the data and typically redirect)

Implementing this yourself often results in a lot of repeated boilerplate code (see Using a form in a view). To help avoid this, Django provides a collection of generic class-based views for form processing.


Basic forms¶

Given a simple contact form:


forms.py

from django import forms

class ContactForm(forms.Form):
    name = forms.CharField()
    message = forms.CharField(widget=forms.Textarea)

    def send_email(self):
        # send email using the self.cleaned_data dictionary
        pass


The view can be constructed using a FormView:


views.py

from myapp.forms import ContactForm
from django.views.generic.edit import FormView

class ContactView(FormView):
    template_name = 'contact.html'
    form_class = ContactForm
    success_url = '/thanks/'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        form.send_email()
        return super(ContactView, self).form_valid(form)


Notes:
•FormView inherits TemplateResponseMixin so template_name can be used here.
•The default implementation for form_valid() simply redirects to the success_url.


Model forms¶

Generic views really shine when working with models. These generic views will automatically create a ModelForm, so long as they can work out which model class to use:
•If the model attribute is given, that model class will be used.
•If get_object() returns an object, the class of that object will be used.
•If a queryset is given, the model for that queryset will be used.

Model form views provide a form_valid() implementation that saves the model automatically. You can override this if you have any special requirements; see below for examples.

You don’t even need to provide a success_url for CreateView or UpdateView - they will use get_absolute_url() on the model object if available.

If you want to use a custom ModelForm (for instance to add extra validation) simply set form_class on your view.


?Note

When specifying a custom form class, you must still specify the model, even though the form_class may be a ModelForm.

First we need to add get_absolute_url() to our Author class:


models.py

from django.urls import reverse
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=200)

    def get_absolute_url(self):
        return reverse('author-detail', kwargs={'pk': self.pk})


Then we can use CreateView and friends to do the actual work. Notice how we’re just configuring the generic class-based views here; we don’t have to write any logic ourselves:


views.py

from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from myapp.models import Author

class AuthorCreate(CreateView):
    model = Author
    fields = ['name']

class AuthorUpdate(UpdateView):
    model = Author
    fields = ['name']

class AuthorDelete(DeleteView):
    model = Author
    success_url = reverse_lazy('author-list')



?Note

We have to use reverse_lazy() here, not just reverse() as the urls are not loaded when the file is imported.

The fields attribute works the same way as the fields attribute on the inner Meta class on ModelForm. Unless you define the form class in another way, the attribute is required and the view will raise an ImproperlyConfigured exception if it’s not.

If you specify both the fields and form_class attributes, an ImproperlyConfigured exception will be raised.

Finally, we hook these new views into the URLconf:


urls.py

from django.conf.urls import url
from myapp.views import AuthorCreate, AuthorUpdate, AuthorDelete

urlpatterns = [
    # ...
    url(r'author/add/$', AuthorCreate.as_view(), name='author-add'),
    url(r'author/(?P<pk>[0-9]+)/$', AuthorUpdate.as_view(), name='author-update'),
    url(r'author/(?P<pk>[0-9]+)/delete/$', AuthorDelete.as_view(), name='author-delete'),
]



Models and request.user¶

To track the user that created an object using a CreateView, you can use a custom ModelForm to do this. First, add the foreign key relation to the model:


models.py

from django.contrib.auth.models import User
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    # ...


In the view, ensure that you don’t include created_by in the list of fields to edit, and override form_valid() to add the user:


views.py

from django.views.generic.edit import CreateView
from myapp.models import Author

class AuthorCreate(CreateView):
    model = Author
    fields = ['name']

    def form_valid(self, form):
        form.instance.created_by = self.request.user
        return super(AuthorCreate, self).form_valid(form)


Note that you’ll need to decorate this view using login_required(), or alternatively handle unauthorized users in the form_valid().


AJAX example¶

Here is a simple example showing how you might go about implementing a form that works for AJAX requests as well as ‘normal’ form POSTs:


from django.http import JsonResponse
from django.views.generic.edit import CreateView
from myapp.models import Author

class AjaxableResponseMixin(object):
    """
    Mixin to add AJAX support to a form.
    Must be used with an object-based FormView (e.g. CreateView)
    """
    def form_invalid(self, form):
        response = super(AjaxableResponseMixin, self).form_invalid(form)
        if self.request.is_ajax():
            return JsonResponse(form.errors, status=400)
        else:
            return response

    def form_valid(self, form):
        # We make sure to call the parent's form_valid() method because
        # it might do some processing (in the case of CreateView, it will
        # call form.save() for example).
        response = super(AjaxableResponseMixin, self).form_valid(form)
        if self.request.is_ajax():
            data = {
                'pk': self.object.pk,
            }
            return JsonResponse(data)
        else:
            return response

class AuthorCreate(AjaxableResponseMixin, CreateView):
    model = Author
    fields = ['name']



https://docs.djangoproject.com/en/1.10/ref/class-based-views/generic-editing/
Generic editing views¶

The following views are described on this page and provide a foundation for editing content:
•django.views.generic.edit.FormView
•django.views.generic.edit.CreateView
•django.views.generic.edit.UpdateView
•django.views.generic.edit.DeleteView


?Note

Some of the examples on this page assume that an Author model has been defined as follows in myapp/models.py:


from django.urls import reverse
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=200)

    def get_absolute_url(self):
        return reverse('author-detail', kwargs={'pk': self.pk})



FormView¶
class django.views.generic.edit.FormView¶
A view that displays a form. On error, redisplays the form with validation errors; on success, redirects to a new URL.

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.edit.BaseFormView
•django.views.generic.edit.FormMixin
•django.views.generic.edit.ProcessFormView
•django.views.generic.base.View

Example myapp/forms.py:


from django import forms

class ContactForm(forms.Form):
    name = forms.CharField()
    message = forms.CharField(widget=forms.Textarea)

    def send_email(self):
        # send email using the self.cleaned_data dictionary
        pass


Example myapp/views.py:


from myapp.forms import ContactForm
from django.views.generic.edit import FormView

class ContactView(FormView):
    template_name = 'contact.html'
    form_class = ContactForm
    success_url = '/thanks/'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        form.send_email()
        return super(ContactView, self).form_valid(form)


Example myapp/contact.html:


<form action="" method="post">{% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Send message" />
</form>



CreateView¶
class django.views.generic.edit.CreateView¶
A view that displays a form for creating an object, redisplaying the form with validation errors (if there are any) and saving the object.

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.detail.SingleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.edit.BaseCreateView
•django.views.generic.edit.ModelFormMixin
•django.views.generic.edit.FormMixin
•django.views.generic.detail.SingleObjectMixin
•django.views.generic.edit.ProcessFormView
•django.views.generic.base.View

Attributes
template_name_suffix¶
The CreateView page displayed to a GET request uses a template_name_suffix of '_form'. For example, changing this attribute to '_create_form' for a view creating objects for the example Author model would cause the default template_name to be 'myapp/author_create_form.html'.
object¶
When using CreateView you have access to self.object, which is the object being created. If the object hasn’t been created yet, the value will be None.

Example myapp/views.py:


from django.views.generic.edit import CreateView
from myapp.models import Author

class AuthorCreate(CreateView):
    model = Author
    fields = ['name']


Example myapp/author_form.html:


<form action="" method="post">{% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Save" />
</form>



UpdateView¶
class django.views.generic.edit.UpdateView¶
A view that displays a form for editing an existing object, redisplaying the form with validation errors (if there are any) and saving changes to the object. This uses a form automatically generated from the object’s model class (unless a form class is manually specified).

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.detail.SingleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.edit.BaseUpdateView
•django.views.generic.edit.ModelFormMixin
•django.views.generic.edit.FormMixin
•django.views.generic.detail.SingleObjectMixin
•django.views.generic.edit.ProcessFormView
•django.views.generic.base.View

Attributes
template_name_suffix¶
The UpdateView page displayed to a GET request uses a template_name_suffix of '_form'. For example, changing this attribute to '_update_form' for a view updating objects for the example Author model would cause the default template_name to be 'myapp/author_update_form.html'.
object¶
When using UpdateView you have access to self.object, which is the object being updated.

Example myapp/views.py:


from django.views.generic.edit import UpdateView
from myapp.models import Author

class AuthorUpdate(UpdateView):
    model = Author
    fields = ['name']
    template_name_suffix = '_update_form'


Example myapp/author_update_form.html:


<form action="" method="post">{% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Update" />
</form>



DeleteView¶
class django.views.generic.edit.DeleteView¶
A view that displays a confirmation page and deletes an existing object. The given object will only be deleted if the request method is POST. If this view is fetched via GET, it will display a confirmation page that should contain a form that POSTs to the same URL.

Ancestors (MRO)

This view inherits methods and attributes from the following views:
•django.views.generic.detail.SingleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.edit.BaseDeleteView
•django.views.generic.edit.DeletionMixin
•django.views.generic.detail.BaseDetailView
•django.views.generic.detail.SingleObjectMixin
•django.views.generic.base.View

Attributes
template_name_suffix¶
The DeleteView page displayed to a GET request uses a template_name_suffix of '_confirm_delete'. For example, changing this attribute to '_check_delete' for a view deleting objects for the example Author model would cause the default template_name to be 'myapp/author_check_delete.html'.

Example myapp/views.py:


from django.views.generic.edit import DeleteView
from django.urls import reverse_lazy
from myapp.models import Author

class AuthorDelete(DeleteView):
    model = Author
    success_url = reverse_lazy('author-list')


Example myapp/author_confirm_delete.html:


<form action="" method="post">{% csrf_token %}
    <p>Are you sure you want to delete "{{ object }}"?</p>
    <input type="submit" value="Confirm" />
</form>


https://docs.djangoproject.com/en/1.10/ref/class-based-views/generic-date-based/

Generic date views¶

Date-based generic views, provided in django.views.generic.dates, are views for displaying drilldown pages for date-based data.


?Note

Some of the examples on this page assume that an Article model has been defined as follows in myapp/models.py:


from django.db import models
from django.urls import reverse

class Article(models.Model):
    title = models.CharField(max_length=200)
    pub_date = models.DateField()

    def get_absolute_url(self):
        return reverse('article-detail', kwargs={'pk': self.pk})



ArchiveIndexView¶
class ArchiveIndexView[source]¶
A top-level index page showing the “latest” objects, by date. Objects with a date in the future are not included unless you set allow_future to True.

Ancestors (MRO)
•django.views.generic.list.MultipleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.dates.BaseArchiveIndexView
•django.views.generic.dates.BaseDateListView
•django.views.generic.list.MultipleObjectMixin
•django.views.generic.dates.DateMixin
•django.views.generic.base.View

Context

In addition to the context provided by django.views.generic.list.MultipleObjectMixin (via django.views.generic.dates.BaseDateListView), the template’s context will be:
•date_list: A QuerySet object containing all years that have objects available according to queryset, represented as datetime.datetime objects, in descending order.

Notes
•Uses a default context_object_name of latest.
•Uses a default template_name_suffix of _archive.
•Defaults to providing date_list by year, but this can be altered to month or day using the attribute date_list_period. This also applies to all subclass views.

Example myapp/urls.py:


from django.conf.urls import url
from django.views.generic.dates import ArchiveIndexView

from myapp.models import Article

urlpatterns = [
    url(r'^archive/$',
        ArchiveIndexView.as_view(model=Article, date_field="pub_date"),
        name="article_archive"),
]


Example myapp/article_archive.html:


<ul>
    {% for article in latest %}
        <li>{{ article.pub_date }}: {{ article.title }}</li>
    {% endfor %}
</ul>


This will output all articles.


YearArchiveView¶
class YearArchiveView[source]¶
A yearly archive page showing all available months in a given year. Objects with a date in the future are not displayed unless you set allow_future to True.

Ancestors (MRO)
•django.views.generic.list.MultipleObjectTemplateResponseMixin
•django.views.generic.base.TemplateResponseMixin
•django.views.generic.dates.BaseYearArchiveView
•django.views.generic.dates.YearMixin
•django.views.generic.dates.BaseDateListView
•django.views.generic.list.MultipleObjectMixin
•django.views.generic.dates.DateMixin
•django.views.generic.base.View
make_object_list¶
A boolean specifying whether to retrieve the full list of objects for this year and pass those to the template. If True, the list of objects will be made available to the context. If False, the None queryset will be used as the object list. By default, this is False.
get_make_object_list()¶
Determine if an object list will be returned as part of the context. Returns make_object_list by default.

Context

In addition to the context provided by django.views.generic.list.MultipleObjectMixin (via django.views.generic.dates.BaseDateListView), the template’s context will be:
•date_list: A QuerySet object containing all months that have objects available according to queryset, represented as datetime.datetime objects, in ascending order.
•year: A date object representing the given year.
•next_year: A date object representing the first day of the next year, according to allow_empty and allow_future.
•previous_year: A date object representing the first day of the previous year, according to allow_empty and allow_future.

Notes
•Uses a default template_name_suffix of _archive_year.

Example myapp/views.py:


from django.views.generic.dates import YearArchiveView

from myapp.models import Article

class ArticleYearArchiveView(YearArchiveView):
    queryset = Article.objects.all()
    date_field = "pub_date"
    make_object_list = True
    allow_future = True


Example myapp/urls.py:


from django.conf.urls import url

from myapp.views import ArticleYearArchiveView

urlpatterns = [
    url(r'^(?P<year>[0-9]{4})/$',
        ArticleYearArchiveView.as_view(),
        name="article_year_archive"),
]


Example myapp/article_archive_year.html:


<ul>
    {% for date in date_list %}
        <li>{{ date|date }}</li>
    {% endfor %}
</ul>

<div>
    <h1>All Articles for {{ year|date:"Y" }}</h1>
    {% for obj in object_list %}
        <p>
            {{ obj.title }} - {{ obj.pub_date|date:"F j, Y" }}
        </p>
    {% endfor %}
</div>

Similar view are 
MonthArchiveView¶
class MonthArchiveView[source]¶
A monthly archive page showing all objects in a given month. Objects with a date in the future are not displayed unless you set allow_future to True.
Example myapp/views.py:


from django.views.generic.dates import MonthArchiveView

from myapp.models import Article

class ArticleMonthArchiveView(MonthArchiveView):
    queryset = Article.objects.all()
    date_field = "pub_date"
    allow_future = True


Example myapp/urls.py:


from django.conf.urls import url

from myapp.views import ArticleMonthArchiveView

urlpatterns = [
    # Example: /2012/aug/
    url(r'^(?P<year>[0-9]{4})/(?P<month>[-\w]+)/$',
        ArticleMonthArchiveView.as_view(),
        name="archive_month"),
    # Example: /2012/08/
    url(r'^(?P<year>[0-9]{4})/(?P<month>[0-9]+)/$',
        ArticleMonthArchiveView.as_view(month_format='%m'),
        name="archive_month_numeric"),
]


Example myapp/article_archive_month.html:


<ul>
    {% for article in object_list %}
        <li>{{ article.pub_date|date:"F j, Y" }}: {{ article.title }}</li>
    {% endfor %}
</ul>

<p>
    {% if previous_month %}
        Previous Month: {{ previous_month|date:"F Y" }}
    {% endif %}
    {% if next_month %}
        Next Month: {{ next_month|date:"F Y" }}
    {% endif %}
</p>


class WeekArchiveView[source]¶
A weekly archive page showing all objects in a given week. Objects with a date in the future are not displayed unless you set allow_future to True.
same as the ISO 8601 week number.


Example myapp/views.py:


from django.views.generic.dates import WeekArchiveView

from myapp.models import Article

class ArticleWeekArchiveView(WeekArchiveView):
    queryset = Article.objects.all()
    date_field = "pub_date"
    week_format = "%W"
    allow_future = True


Example myapp/urls.py:


from django.conf.urls import url

from myapp.views import ArticleWeekArchiveView

urlpatterns = [
    # Example: /2012/week/23/
    url(r'^(?P<year>[0-9]{4})/week/(?P<week>[0-9]+)/$',
        ArticleWeekArchiveView.as_view(),
        name="archive_week"),
]


Example myapp/article_archive_week.html:


<h1>Week {{ week|date:'W' }}</h1>

<ul>
    {% for article in object_list %}
        <li>{{ article.pub_date|date:"F j, Y" }}: {{ article.title }}</li>
    {% endfor %}
</ul>

<p>
    {% if previous_week %}
        Previous Week: {{ previous_week|date:"W" }} of year {{ previous_week|date:"Y" }}
    {% endif %}
    {% if previous_week and next_week %}--{% endif %}
    {% if next_week %}
        Next week: {{ next_week|date:"W" }} of year {{ next_week|date:"Y" }}
    {% endif %}
</p>


class DayArchiveView[source]¶
A day archive page showing all objects in a given day. Days in the future throw a 404 error, regardless of whether any objects exist for future days, unless you set allow_future to True.
Example myapp/views.py:


from django.views.generic.dates import DayArchiveView

from myapp.models import Article

class ArticleDayArchiveView(DayArchiveView):
    queryset = Article.objects.all()
    date_field = "pub_date"
    allow_future = True


Example myapp/urls.py:


from django.conf.urls import url

from myapp.views import ArticleDayArchiveView

urlpatterns = [
    # Example: /2012/nov/10/
    url(r'^(?P<year>[0-9]{4})/(?P<month>[-\w]+)/(?P<day>[0-9]+)/$',
        ArticleDayArchiveView.as_view(),
        name="archive_day"),
]


Example myapp/article_archive_day.html:


<h1>{{ day }}</h1>

<ul>
    {% for article in object_list %}
        <li>{{ article.pub_date|date:"F j, Y" }}: {{ article.title }}</li>
    {% endfor %}
</ul>

<p>
    {% if previous_day %}
        Previous Day: {{ previous_day }}
    {% endif %}
    {% if previous_day and next_day %}--{% endif %}
    {% if next_day %}
        Next Day: {{ next_day }}
    {% endif %}
</p>


TodayArchiveView¶
class TodayArchiveView[source]¶
A day archive page showing all objects for today. This is exactly the same as django.views.generic.dates.DayArchiveView, except today’s date is used instead of the year/month/day arguments.
DateDetailView¶
Example myapp/views.py:


from django.views.generic.dates import TodayArchiveView

from myapp.models import Article

class ArticleTodayArchiveView(TodayArchiveView):
    queryset = Article.objects.all()
    date_field = "pub_date"
    allow_future = True


Example myapp/urls.py:


from django.conf.urls import url

from myapp.views import ArticleTodayArchiveView

urlpatterns = [
    url(r'^today/$',
        ArticleTodayArchiveView.as_view(),
        name="archive_today"),
]



class DateDetailView[source]¶
A page representing an individual object. If the object has a date value in the future, the view will throw a 404 error by default, unless you set allow_future to True.
Example myapp/urls.py:


from django.conf.urls import url
from django.views.generic.dates import DateDetailView

urlpatterns = [
    url(r'^(?P<year>[0-9]{4})/(?P<month>[-\w]+)/(?P<day>[0-9]+)/(?P<pk>[0-9]+)/$',
        DateDetailView.as_view(model=Article, date_field="pub_date"),
        name="archive_date_detail"),
]


Example myapp/article_detail.html:


<h1>{{ object.title }}</h1>




https://docs.djangoproject.com/en/1.10/topics/class-based-views/mixins/

Context and template responses¶

Two central mixins are provided that help in providing a consistent interface to working with templates in class-based views.
TemplateResponseMixin
Every built in view which returns a TemplateResponse will call the render_to_response() method that TemplateResponseMixin provides. Most of the time this will be called for you (for instance, it is called by the get() method implemented by both TemplateView and DetailView); 

render_to_response() itself calls get_template_names(), which by default will just look up template_name on the class-based view; two other mixins (SingleObjectTemplateResponseMixin and MultipleObjectTemplateResponseMixin) override this to provide more flexible defaults when dealing with actual objects.
ContextMixinEvery built in view which needs context data, such as for rendering a template (including TemplateResponseMixin above), should call get_context_data() passing any data they want to ensure is in there as keyword arguments. get_context_data() returns a dictionary; in ContextMixin it simply returns its keyword arguments, but it is common to override this to add more members to the dictionary.


Using SingleObjectMixin with View¶

If we want to write a simple class-based view that responds only to POST, we’ll subclass View and write a post() method in the subclass. However if we want our processing to work on a particular object, identified from the URL, we’ll want the functionality provided by SingleObjectMixin.

We’ll demonstrate this with the Author model we used in the generic class-based views introduction.


views.py

from django.http import HttpResponseForbidden, HttpResponseRedirect
from django.urls import reverse
from django.views import View
from django.views.generic.detail import SingleObjectMixin
from books.models import Author

class RecordInterest(SingleObjectMixin, View):
    """Records the current user's interest in an author."""
    model = Author

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseForbidden()

        # Look up the author we're interested in.
        self.object = self.get_object()
        # Actually record interest somehow here!

        return HttpResponseRedirect(reverse('author-detail', kwargs={'pk': self.object.pk}))


In practice you’d probably want to record the interest in a key-value store rather than in a relational database, so we’ve left that bit out. The only bit of the view that needs to worry about using SingleObjectMixin is where we want to look up the author we’re interested in, which it just does with a simple call to self.get_object(). Everything else is taken care of for us by the mixin.

We can hook this into our URLs easily enough:


urls.py

from django.conf.urls import url
from books.views import RecordInterest

urlpatterns = [
    #...
    url(r'^author/(?P<pk>[0-9]+)/interest/$', RecordInterest.as_view(), name='author-interest'),
]


Note the pk named group, which get_object() uses to look up the Author instance. You could also use a slug, or any of the other features of SingleObjectMixin.


Using SingleObjectMixin with ListView¶

ListView provides built-in pagination, but you might want to paginate a list of objects that are all linked (by a foreign key) to another object. In our publishing example, you might want to paginate through all the books by a particular publisher.

One way to do this is to combine ListView with SingleObjectMixin, so that the queryset for the paginated list of books can hang off the publisher found as the single object. In order to do this, we need to have two different querysets:
Book queryset for use by ListViewSince we have access to the Publisher whose books we want to list, we simply override get_queryset() and use the Publisher’s reverse foreign key manager.Publisher queryset for use in get_object()We’ll rely on the default implementation of get_object() to fetch the correct Publisher object. However, we need to explicitly pass a queryset argument because otherwise the default implementation of get_object() would call get_queryset() which we have overridden to return Book objects instead of Publisher ones.

Using FormMixin with DetailView¶

Think back to our earlier example of using View and SingleObjectMixin together. We were recording a user’s interest in a particular author; say now that we want to let them leave a message saying why they like them. Again, let’s assume we’re not going to store this in a relational database but instead in something more esoteric that we won’t worry about here.

At this point it’s natural to reach for a Form to encapsulate the information sent from the user’s browser to Django. Say also that we’re heavily invested in REST, so we want to use the same URL for displaying the author as for capturing the message from the user



An alternative better solution¶

What we’re really trying to do here is to use two different class based views from the same URL. So why not do just that? We have a very clear division here: GET requests should get the DetailView (with the Form added to the context data), and POST requests should get the FormView. Let’s set up those views first.

The AuthorDisplay view is almost the same as when we first introduced AuthorDetail; we have to write our own get_context_data() to make the AuthorInterestForm available to the template. We’ll skip the get_object() override from before for clarity:


from django.views.generic import DetailView
from django import forms
from books.models import Author

class AuthorInterestForm(forms.Form):
    message = forms.CharField()

class AuthorDisplay(DetailView):
    model = Author

    def get_context_data(self, **kwargs):
        context = super(AuthorDisplay, self).get_context_data(**kwargs)
        context['form'] = AuthorInterestForm()
        return context


Then the AuthorInterest is a simple FormView, but we have to bring in SingleObjectMixin so we can find the author we’re talking about, and we have to remember to set template_name to ensure that form errors will render the same template as AuthorDisplay is using on GET:


from django.urls import reverse
from django.http import HttpResponseForbidden
from django.views.generic import FormView
from django.views.generic.detail import SingleObjectMixin

class AuthorInterest(SingleObjectMixin, FormView):
    template_name = 'books/author_detail.html'
    form_class = AuthorInterestForm
    model = Author

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseForbidden()
        self.object = self.get_object()
        return super(AuthorInterest, self).post(request, *args, **kwargs)

    def get_success_url(self):
        return reverse('author-detail', kwargs={'pk': self.object.pk})


Finally we bring this together in a new AuthorDetail view. We already know that calling as_view() on a class-based view gives us something that behaves exactly like a function based view, so we can do that at the point we choose between the two subviews.

You can of course pass through keyword arguments to as_view() in the same way you would in your URLconf, such as if you wanted the AuthorInterest behavior to also appear at another URL but using a different template:


from django.views import View

class AuthorDetail(View):

    def get(self, request, *args, **kwargs):
        view = AuthorDisplay.as_view()
        return view(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        view = AuthorInterest.as_view()
        return view(request, *args, **kwargs)


This approach can also be used with any other generic class-based views or your own class-based views inheriting directly from View or TemplateView, as it keeps the different views as separate as possible.


More than just HTML¶

Where class-based views shine is when you want to do the same thing many times. Suppose you’re writing an API, and every view should return JSON instead of rendered HTML.

We can create a mixin class to use in all of our views, handling the conversion to JSON once.

For example, a simple JSON mixin might look something like this:


from django.http import JsonResponse

class JSONResponseMixin(object):
    """
    A mixin that can be used to render a JSON response.
    """
    def render_to_json_response(self, context, **response_kwargs):
        """
        Returns a JSON response, transforming 'context' to make the payload.
        """
        return JsonResponse(
            self.get_data(context),
            **response_kwargs
        )

    def get_data(self, context):
        """
        Returns an object that will be serialized as JSON by json.dumps().
        """
        # Note: This is *EXTREMELY* naive; in reality, you'll need
        # to do much more complex handling to ensure that arbitrary
        # objects -- such as Django model instances or querysets
        # -- can be serialized as JSON.
        return context



?Note

Check out the Serializing Django objects documentation for more information on how to correctly transform Django models and querysets into JSON.

This mixin provides a render_to_json_response() method with the same signature as render_to_response(). To use it, we simply need to mix it into a TemplateView for example, and override render_to_response() to call render_to_json_response() instead:


from django.views.generic import TemplateView

class JSONView(JSONResponseMixin, TemplateView):
    def render_to_response(self, context, **response_kwargs):
        return self.render_to_json_response(context, **response_kwargs)


Equally we could use our mixin with one of the generic views. We can make our own version of DetailView by mixing JSONResponseMixin with the django.views.generic.detail.BaseDetailView – (the DetailView before template rendering behavior has been mixed in):


from django.views.generic.detail import BaseDetailView

class JSONDetailView(JSONResponseMixin, BaseDetailView):
    def render_to_response(self, context, **response_kwargs):
        return self.render_to_json_response(context, **response_kwargs)


This view can then be deployed in the same way as any other DetailView, with exactly the same behavior – except for the format of the response.

If you want to be really adventurous, you could even mix a DetailView subclass that is able to return both HTML and JSON content, depending on some property of the HTTP request, such as a query argument or a HTTP header. Just mix in both the JSONResponseMixin and a SingleObjectTemplateResponseMixin, and override the implementation of render_to_response() to defer to the appropriate rendering method depending on the type of response that the user requested:


from django.views.generic.detail import SingleObjectTemplateResponseMixin

class HybridDetailView(JSONResponseMixin, SingleObjectTemplateResponseMixin, BaseDetailView):
    def render_to_response(self, context):
        # Look for a 'format=json' GET argument
        if self.request.GET.get('format') == 'json':
            return self.render_to_json_response(context)
        else:
            return super(HybridDetailView, self).render_to_response(context)


Because of the way that Python resolves method overloading, the call to super(HybridDetailView, self).render_to_response(context) ends up calling the render_to_response() implementation of TemplateResponseMixin.










###Django - file uploads 
https://docs.djangoproject.com/en/1.10/topics/http/file-uploads/

##Basic file uploads
#simple form containing a FileField:

#forms.py

from django import forms

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()   #or ImageField

#request.FILES will only contain data if the request method was POST 
#and the <form> that posted the request has the attribute enctype="multipart/form-data". 
#Otherwise, request.FILES will be empty.

#views.py   - request.FILES is dict with key as FileField attribute name , can be multiple 

from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm

# Imaginary function to handle an uploaded file.
from somewhere import handle_uploaded_file

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)   #must to bound to request 
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})


#To handle an uploaded file:


def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():  #f is UploadedFile
            destination.write(chunk)

            
#Methods of UploadedFile
UploadedFile.read()
Read the entire uploaded data from the file

UploadedFile.multiple_chunks(chunk_size=None)
Returns True if the uploaded file is big enough to require reading in multiple chunks

UploadedFile.chunks(chunk_size=None)
A generator returning chunks of the file. 
If multiple_chunks() is True, Use this 

UploadedFile.name
The name of the uploaded file (e.g. my_file.txt).

UploadedFile.size
The size, in bytes, of the uploaded file.

UploadedFile.content_type
The content-type header uploaded with the file (e.g. text/plain or application/pdf)

#Subclasses of UploadedFile include:
class TemporaryUploadedFile[source]
A file uploaded to a temporary location (i.e. stream-to-disk). 
This class is used by the TemporaryFileUploadHandler. 
In addition to the methods from UploadedFile, it has one additional method:
TemporaryUploadedFile.temporary_file_path()
Returns the full path to the temporary uploaded file.

class InMemoryUploadedFile
A file uploaded into memory (i.e. stream-to-memory). 
This class is used by the MemoryFileUploadHandler.


##Handling uploaded files with a model

If you’re saving a file on a Model with a FileField, using a ModelForm makes this process much easier. The file object will be saved to the location specified by the upload_to argument of the corresponding FileField when calling form.save():


from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import ModelFormWithFileField

def upload_file(request):
    if request.method == 'POST':
        form = ModelFormWithFileField(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            return HttpResponseRedirect('/success/url/')
    else:
        form = ModelFormWithFileField()
    return render(request, 'upload.html', {'form': form})


If you are constructing an object manually, you can simply assign the file object from request.FILES to the file field in the model:


from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from .models import ModelWithFileField

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            instance = ModelWithFileField(file_field=request.FILES['file'])
            instance.save()
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})



Uploading multiple files¶

If you want to upload multiple files using one form field, set the multiple HTML attribute of field’s widget:


forms.py

from django import forms

class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))


Then override the post method of your FormView subclass to handle multiple file uploads:


views.py

from django.views.generic.edit import FormView
from .forms import FileFieldForm

class FileFieldView(FormView):
    form_class = FileFieldForm
    template_name = 'upload.html'  # Replace with your template.
    success_url = '...'  # Replace with your URL or reverse().

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            for f in files:
                ...  # Do something with each file.
            return self.form_valid(form)
        else:
            return self.form_invalid(form)



Upload Handlers¶

When a user uploads a file, Django passes off the file data to an upload handler – a small class that handles file data as it gets uploaded. Upload handlers are initially defined in the FILE_UPLOAD_HANDLERS setting, which defaults to:


["django.core.files.uploadhandler.MemoryFileUploadHandler",
 "django.core.files.uploadhandler.TemporaryFileUploadHandler"]


Together MemoryFileUploadHandler and TemporaryFileUploadHandler provide Django’s default file upload behavior of reading small files into memory and large ones onto disk
Where uploaded data is stored¶

Before you save uploaded files, the data needs to be stored somewhere.

By default, if an uploaded file is smaller than 2.5 megabytes, Django will hold the entire contents of the upload in memory. This means that saving the file involves only a read from memory and a write to disk and thus is very fast.

However, if an uploaded file is too large, Django will write the uploaded file to a temporary file stored in your system’s temporary directory. On a Unix-like platform this means you can expect Django to generate a file called something like /tmp/tmpzfp6I6.upload

#Seetings.py settings for Upload file 
•DEFAULT_FILE_STORAGE
•FILE_CHARSET
•FILE_UPLOAD_HANDLERS
•FILE_UPLOAD_MAX_MEMORY_SIZE
•FILE_UPLOAD_PERMISSIONS
•FILE_UPLOAD_TEMP_DIR
•MEDIA_ROOT
•MEDIA_URL





@@@

Template


A template is simply a text file. It can generate any text-based format (HTML, XML, CSV, etc.).

A template contains variables, which get replaced with values when the template is evaluated, and tags, which control the logic of the template.

Example:

{% extends "base_generic.html" %}

{% block title %}{{ section.title }}{% endblock %}

{% block content %}
<h1>{{ section.title }}</h1>

{% for story in story_list %}
<h2>
  <a href="{{ story.get_absolute_url }}">
    {{ story.headline|upper }}
  </a>
</h2>
<p>{{ story.tease|truncatewords:"100" }}</p>
{% endfor %}
{% endblock %}



Variables   {{ variable }}. 

When the template engine encounters a variable, it evaluates that variable and replaces it with the result. 
Variable names consist of any combination of alphanumeric characters and the underscore ("_"). 
Use a dot (.) to access attributes of a variable.

when the template system encounters a dot, it tries the following lookups, in this order:

•Dictionary lookup
•Attribute or method lookup
•Numeric index lookup


If the resulting value is callable, it is called with no arguments. The result of the call becomes the template value.


If you use a variable that doesn’t exist, the template system will insert the value of the TEMPLATE_STRING_IF_INVALID setting, which is set to '' (the empty string) by default.

Note that “bar” in a template expression like {{ foo.bar }} will be interpreted as a literal string and not using the value of the variable “bar”, if one exists in the template context.


Filters -Use a pipe (|) to apply a filter.

Filters look like this: {{ name|lower }}. This displays the value of the {{ name }} variable after being filtered through the lower filter, which converts text to lowercase. 

Filters can be “chained.” The output of one filter is applied to the next. 
{{ text|escape|linebreaks }} is a common idiom for escaping text contents, then converting line breaks to <p> tags.

Some filters take arguments. A filter argument looks like this: {{ bio|truncatewords:30 }}. This will display the first 30 words of the bio variable.

Filter arguments that contain spaces must be quoted; for example, to join a list with commas and spaced you’d use {{ list|join:", " }}.



List of filters

add
Adds the argument to the value.

For example:

{{ value|add:"2" }}
If value is 4, then the output will be 6.

This filter will first try to coerce both values to integers. If this fails, it’ll attempt to add the values together anyway. This will work on some data types (strings, list, etc.) and fail on others. If it fails, the result will be an empty string.

For example, if we have:

{{ first|add:second }}
and first is [1, 2, 3] and second is [4, 5, 6], then the output will be [1, 2, 3, 4, 5, 6].



addslashes
Adds slashes before quotes. Useful for escaping strings in CSV, for example.

For example:

{{ value|addslashes }}
If value is "I'm using Django", the output will be "I\'m using Django".

capfirst
Capitalizes the first character of the value. If the first character is not a letter, this filter has no effect.

For example:

{{ value|capfirst }}
If value is "django", the output will be "Django".

center
Centers the value in a field of a given width.

For example:

"{{ value|center:"15" }}"
If value is "Django", the output will be "     Django    ".

cut
Removes all values of arg from the given string.

For example:

{{ value|cut:" " }}
If value is "String with spaces", the output will be "Stringwithspaces".

date
Formats a date according to the given format.

Uses a similar format as PHP’s date() function (http://php.net/date) with some differences.

Note

These format characters are not used in Django outside of templates. They were designed to be compatible with PHP to ease transitioning for designers.

Available format strings:

Format character Description Example output 
a 'a.m.' or 'p.m.' (Note that this is slightly different than PHP’s output, because this includes periods to match Associated Press style.) 'a.m.' 
A 'AM' or 'PM'. 'AM' 
b Month, textual, 3 letters, lowercase. 'jan' 
B Not implemented.   
c ISO 8601 format. (Note: unlike others formatters, such as “Z”, “O” or “r”, the “c” formatter will not add timezone offset if value is a naive datetime (see datetime.tzinfo). 2008-01-02T10:30:00.000123+02:00, or 2008-01-02T10:30:00.000123 if the datetime is naive 
d Day of the month, 2 digits with leading zeros. '01' to '31' 
D Day of the week, textual, 3 letters. 'Fri' 
e Timezone name. Could be in any format, or might return an empty string, depending on the datetime. '', 'GMT', '-500', 'US/Eastern', etc. 
E Month, locale specific alternative representation usually used for long date representation. 'listopada' (for Polish locale, as opposed to 'Listopad') 
f Time, in 12-hour hours and minutes, with minutes left off if they’re zero. Proprietary extension. '1', '1:30' 
F Month, textual, long. 'January' 
g Hour, 12-hour format without leading zeros. '1' to '12' 
G Hour, 24-hour format without leading zeros. '0' to '23' 
h Hour, 12-hour format. '01' to '12' 
H Hour, 24-hour format. '00' to '23' 
i Minutes. '00' to '59' 
I Daylight Savings Time, whether it’s in effect or not. '1' or '0' 
j Day of the month without leading zeros. '1' to '31' 
l Day of the week, textual, long. 'Friday' 
L Boolean for whether it’s a leap year. True or False 
m Month, 2 digits with leading zeros. '01' to '12' 
M Month, textual, 3 letters. 'Jan' 
n Month without leading zeros. '1' to '12' 
N Month abbreviation in Associated Press style. Proprietary extension. 'Jan.', 'Feb.', 'March', 'May' 
o ISO-8601 week-numbering year, corresponding to the ISO-8601 week number (W) '1999' 
O Difference to Greenwich time in hours. '+0200' 
P Time, in 12-hour hours, minutes and ‘a.m.’/’p.m.’, with minutes left off if they’re zero and the special-case strings ‘midnight’ and ‘noon’ if appropriate. Proprietary extension. '1 a.m.', '1:30 p.m.', 'midnight', 'noon', '12:30 p.m.' 
r RFC 2822 formatted date. 'Thu, 21 Dec 2000 16:01:07 +0200' 
s Seconds, 2 digits with leading zeros. '00' to '59' 
S English ordinal suffix for day of the month, 2 characters. 'st', 'nd', 'rd' or 'th' 
t Number of days in the given month. 28 to 31 
T Time zone of this machine. 'EST', 'MDT' 
u Microseconds. 000000 to 999999 
U Seconds since the Unix Epoch (January 1 1970 00:00:00 UTC).   
w Day of the week, digits without leading zeros. '0' (Sunday) to '6' (Saturday) 
W ISO-8601 week number of year, with weeks starting on Monday. 1, 53 
y Year, 2 digits. '99' 
Y Year, 4 digits. '1999' 
z Day of the year. 0 to 365 
Z Time zone offset in seconds. The offset for timezones west of UTC is always negative, and for those east of UTC is always positive. -43200 to 43200 

For example:

{{ value|date:"D d M Y" }}
If value is a datetime object (e.g., the result of datetime.datetime.now()), the output will be the string 'Wed 09 Jan 2008'.

The format passed can be one of the predefined ones DATE_FORMAT, DATETIME_FORMAT, SHORT_DATE_FORMAT or SHORT_DATETIME_FORMAT, or a custom format that uses the format specifiers shown in the table above. Note that predefined formats may vary depending on the current locale.

Assuming that USE_L10N is True and LANGUAGE_CODE is, for example, "es", then for:

{{ value|date:"SHORT_DATE_FORMAT" }}
the output would be the string "09/01/2008" (the "SHORT_DATE_FORMAT" format specifier for the es locale as shipped with Django is "d/m/Y").

When used without a format string:

{{ value|date }}
...the formatting string defined in the DATE_FORMAT setting will be used, without applying any localization.

You can combine date with the time filter to render a full representation of a datetime value. E.g.:

{{ value|date:"D d M Y" }} {{ value|time:"H:i" }}


default
If value evaluates to False, uses the given default. Otherwise, uses the value.

For example:

{{ value|default:"nothing" }}
If value is "" (the empty string), the output will be nothing.

default_if_none
If (and only if) value is None, uses the given default. Otherwise, uses the value.

Note that if an empty string is given, the default value will not be used. Use the default filter if you want to fallback for empty strings.

For example:

{{ value|default_if_none:"nothing" }}
If value is None, the output will be the string "nothing".

dictsort
Takes a list of dictionaries and returns that list sorted by the key given in the argument.

For example:

{{ value|dictsort:"name" }}
If value is:

[
    {'name': 'zed', 'age': 19},
    {'name': 'amy', 'age': 22},
    {'name': 'joe', 'age': 31},
]
then the output would be:

[
    {'name': 'amy', 'age': 22},
    {'name': 'joe', 'age': 31},
    {'name': 'zed', 'age': 19},
]
You can also do more complicated things like:

{% for book in books|dictsort:"author.age" %}
    * {{ book.title }} ({{ book.author.name }})
{% endfor %}
If books is:

[
    {'title': '1984', 'author': {'name': 'George', 'age': 45}},
    {'title': 'Timequake', 'author': {'name': 'Kurt', 'age': 75}},
    {'title': 'Alice', 'author': {'name': 'Lewis', 'age': 33}},
]
then the output would be:

* Alice (Lewis)
* 1984 (George)
* Timequake (Kurt)
dictsortreversed
Takes a list of dictionaries and returns that list sorted in reverse order by the key given in the argument. This works exactly the same as the above filter, but the returned value will be in reverse order.

divisibleby
Returns True if the value is divisible by the argument.

For example:

{{ value|divisibleby:"3" }}
If value is 21, the output would be True.

escape
Escapes a string’s HTML. Specifically, it makes these replacements:

•< is converted to &lt;
•> is converted to &gt;
•' (single quote) is converted to &#39;
•" (double quote) is converted to &quot;
•& is converted to &amp;
The escaping is only applied when the string is output, so it does not matter where in a chained sequence of filters you put escape: it will always be applied as though it were the last filter. If you want escaping to be applied immediately, use the force_escape filter.

Applying escape to a variable that would normally have auto-escaping applied to the result will only result in one round of escaping being done. So it is safe to use this function even in auto-escaping environments. If you want multiple escaping passes to be applied, use the force_escape filter.

For example, you can apply escape to fields when autoescape is off:

{% autoescape off %}
    {{ title|escape }}
{% endautoescape %}


escapejs
Escapes characters for use in JavaScript strings. This does not make the string safe for use in HTML, but does protect you from syntax errors when using templates to generate JavaScript/JSON.

For example:

{{ value|escapejs }}
If value is "testing\r\njavascript \'string" <b>escaping</b>", the output will be "testing\\u000D\\u000Ajavascript \\u0027string\\u0022 \\u003Cb\\u003Eescaping\\u003C/b\\u003E".

filesizeformat
Formats the value like a ‘human-readable’ file size (i.e. '13 KB', '4.1 MB', '102 bytes', etc).

For example:

{{ value|filesizeformat }}
If value is 123456789, the output would be 117.7 MB.

File sizes and SI units

Strictly speaking, filesizeformat does not conform to the International System of Units which recommends using KiB, MiB, GiB, etc. when byte sizes are calculated in powers of 1024 (which is the case here). Instead, Django uses traditional unit names (KB, MB, GB, etc.) corresponding to names that are more commonly used.

first
Returns the first item in a list.

For example:

{{ value|first }}
If value is the list ['a', 'b', 'c'], the output will be 'a'.

fix_ampersands
Note

This is rarely useful as ampersands are automatically escaped. See escape for more information.

Deprecated since version 1.7: 
This filter has been deprecated and will be removed in Django 1.8.

Replaces ampersands with &amp; entities.

For example:

{{ value|fix_ampersands }}
If value is Tom & Jerry, the output will be Tom &amp; Jerry.

However, ampersands used in named entities and numeric character references will not be replaced. For example, if value is Caf&eacute;, the output will not be Caf&amp;eacute; but remain Caf&eacute;. This means that in some edge cases, such as acronyms followed by semicolons, this filter will not replace ampersands that need replacing. For example, if value is Contact the R&D;, the output will remain unchanged because &D; resembles a named entity.

floatformat
When used without an argument, rounds a floating-point number to one decimal place – but only if there’s a decimal part to be displayed. For example:

value Template Output 
34.23234 {{ value|floatformat }} 34.2 
34.00000 {{ value|floatformat }} 34 
34.26000 {{ value|floatformat }} 34.3 

If used with a numeric integer argument, floatformat rounds a number to that many decimal places. For example:

value Template Output 
34.23234 {{ value|floatformat:3 }} 34.232 
34.00000 {{ value|floatformat:3 }} 34.000 
34.26000 {{ value|floatformat:3 }} 34.260 

Particularly useful is passing 0 (zero) as the argument which will round the float to the nearest integer.

value Template Output 
34.23234 {{ value|floatformat:"0" }} 34 
34.00000 {{ value|floatformat:"0" }} 34 
39.56000 {{ value|floatformat:"0" }} 40 

If the argument passed to floatformat is negative, it will round a number to that many decimal places – but only if there’s a decimal part to be displayed. For example:

value Template Output 
34.23234 {{ value|floatformat:"-3" }} 34.232 
34.00000 {{ value|floatformat:"-3" }} 34 
34.26000 {{ value|floatformat:"-3" }} 34.260 

Using floatformat with no argument is equivalent to using floatformat with an argument of -1.

force_escape
Applies HTML escaping to a string (see the escape filter for details). This filter is applied immediately and returns a new, escaped string. This is useful in the rare cases where you need multiple escaping or want to apply other filters to the escaped results. Normally, you want to use the escape filter.

For example, if you want to catch the <p> HTML elements created by the linebreaks filter:

{% autoescape off %}
    {{ body|linebreaks|force_escape }}
{% endautoescape %}
get_digit
Given a whole number, returns the requested digit, where 1 is the right-most digit, 2 is the second-right-most digit, etc. Returns the original value for invalid input (if input or argument is not an integer, or if argument is less than 1). Otherwise, output is always an integer.

For example:

{{ value|get_digit:"2" }}
If value is 123456789, the output will be 8.

iriencode
Converts an IRI (Internationalized Resource Identifier) to a string that is suitable for including in a URL. This is necessary if you’re trying to use strings containing non-ASCII characters in a URL.

It’s safe to use this filter on a string that has already gone through the urlencode filter.

For example:

{{ value|iriencode }}
If value is "?test=1&me=2", the output will be "?test=1&amp;me=2".

join
Joins a list with a string, like Python’s str.join(list)

For example:

{{ value|join:" // " }}
If value is the list ['a', 'b', 'c'], the output will be the string "a // b // c".

last
Returns the last item in a list.

For example:

{{ value|last }}
If value is the list ['a', 'b', 'c', 'd'], the output will be the string "d".

length
Returns the length of the value. This works for both strings and lists.

For example:

{{ value|length }}
If value is ['a', 'b', 'c', 'd'], the output will be 4.

length_is
Returns True if the value’s length is the argument, or False otherwise.

For example:

{{ value|length_is:"4" }}
If value is ['a', 'b', 'c', 'd'], the output will be True.

linebreaks
Replaces line breaks in plain text with appropriate HTML; a single newline becomes an HTML line break (<br />) and a new line followed by a blank line becomes a paragraph break (</p>).

For example:

{{ value|linebreaks }}
If value is Joel\nis a slug, the output will be <p>Joel<br />is a slug</p>.

linebreaksbr
Converts all newlines in a piece of plain text to HTML line breaks (<br />).

For example:

{{ value|linebreaksbr }}
If value is Joel\nis a slug, the output will be Joel<br />is a slug.

linenumbers
Displays text with line numbers.

For example:

{{ value|linenumbers }}
If value is:

one
two
three
the output will be:

1. one
2. two
3. three
ljust
Left-aligns the value in a field of a given width.

Argument: field size

For example:

"{{ value|ljust:"10" }}"
If value is Django, the output will be "Django    ".

lower
Converts a string into all lowercase.

For example:

{{ value|lower }}
If value is Still MAD At Yoko, the output will be still mad at yoko.

make_list
Returns the value turned into a list. For a string, it’s a list of characters. For an integer, the argument is cast into an unicode string before creating a list.

For example:

{{ value|make_list }}
If value is the string "Joel", the output would be the list [u'J', u'o', u'e', u'l']. If value is 123, the output will be the list [u'1', u'2', u'3'].

phone2numeric
Converts a phone number (possibly containing letters) to its numerical equivalent.

The input doesn’t have to be a valid phone number. This will happily convert any string.

For example:

{{ value|phone2numeric }}
If value is 800-COLLECT, the output will be 800-2655328.

pluralize
Returns a plural suffix if the value is not 1. By default, this suffix is 's'.

Example:

You have {{ num_messages }} message{{ num_messages|pluralize }}.
If num_messages is 1, the output will be You have 1 message. If num_messages is 2 the output will be You have 2 messages.

For words that require a suffix other than 's', you can provide an alternate suffix as a parameter to the filter.

Example:

You have {{ num_walruses }} walrus{{ num_walruses|pluralize:"es" }}.
For words that don’t pluralize by simple suffix, you can specify both a singular and plural suffix, separated by a comma.

Example:

You have {{ num_cherries }} cherr{{ num_cherries|pluralize:"y,ies" }}.
Note

Use blocktrans to pluralize translated strings.

pprint
A wrapper around pprint.pprint() – for debugging, really.

random
Returns a random item from the given list.

For example:

{{ value|random }}
If value is the list ['a', 'b', 'c', 'd'], the output could be "b".

removetags
Removes a space-separated list of [X]HTML tags from the output.

For example:

{{ value|removetags:"b span" }}
If value is "<b>Joel</b> <button>is</button> a <span>slug</span>" the unescaped output will be "Joel <button>is</button> a slug".

Note that this filter is case-sensitive.

If value is "<B>Joel</B> <button>is</button> a <span>slug</span>" the unescaped output will be "<B>Joel</B> <button>is</button> a slug".

No safety guarantee

Note that removetags doesn’t give any guarantee about its output being HTML safe. In particular, it doesn’t work recursively, so an input like "<sc<script>ript>alert('XSS')</sc</script>ript>" won’t be safe even if you apply |removetags:"script". So if the input is user provided, NEVER apply the safe filter to a removetags output. If you are looking for something more robust, you can use the bleach Python library, notably its clean method.

rjust
Right-aligns the value in a field of a given width.

Argument: field size

For example:

"{{ value|rjust:"10" }}"
If value is Django, the output will be "    Django".

safe
Marks a string as not requiring further HTML escaping prior to output. When autoescaping is off, this filter has no effect.

Note

If you are chaining filters, a filter applied after safe can make the contents unsafe again. For example, the following code prints the variable as is, unescaped:

{{ var|safe|escape }}
safeseq
Applies the safe filter to each element of a sequence. Useful in conjunction with other filters that operate on sequences, such as join. For example:

{{ some_list|safeseq|join:", " }}
You couldn’t use the safe filter directly in this case, as it would first convert the variable into a string, rather than working with the individual elements of the sequence.

slice
Returns a slice of the list.

Uses the same syntax as Python’s list slicing. See http://www.diveintopython3.net/native-datatypes.html#slicinglists for an introduction.

Example:

{{ some_list|slice:":2" }}
If some_list is ['a', 'b', 'c'], the output will be ['a', 'b'].

slugify
Converts to ASCII. Converts spaces to hyphens. Removes characters that aren’t alphanumerics, underscores, or hyphens. Converts to lowercase. Also strips leading and trailing whitespace.

For example:

{{ value|slugify }}
If value is "Joel is a slug", the output will be "joel-is-a-slug".

stringformat
Formats the variable according to the argument, a string formatting specifier. This specifier uses Python string formatting syntax, with the exception that the leading “%” is dropped.

See http://docs.python.org/library/stdtypes.html#string-formatting-operations for documentation of Python string formatting

For example:

{{ value|stringformat:"E" }}
If value is 10, the output will be 1.000000E+01.

striptags
Makes all possible efforts to strip all [X]HTML tags.

For example:

{{ value|striptags }}
If value is "<b>Joel</b> <button>is</button> a <span>slug</span>", the output will be "Joel is a slug".

No safety guarantee

Note that striptags doesn’t give any guarantee about its output being HTML safe, particularly with non valid HTML input. So NEVER apply the safe filter to a striptags output. If you are looking for something more robust, you can use the bleach Python library, notably its clean method.


time
Formats a time according to the given format.

Given format can be the predefined one TIME_FORMAT, or a custom format, same as the date filter. Note that the predefined format is locale-dependent.

For example:

{{ value|time:"H:i" }}
If value is equivalent to datetime.datetime.now(), the output will be the string "01:23".

Another example:

Assuming that USE_L10N is True and LANGUAGE_CODE is, for example, "de", then for:

{{ value|time:"TIME_FORMAT" }}
the output will be the string "01:23:00" (The "TIME_FORMAT" format specifier for the de locale as shipped with Django is "H:i:s").

The time filter will only accept parameters in the format string that relate to the time of day, not the date (for obvious reasons). If you need to format a date value, use the date filter instead (or along time if you need to render a full datetime value).

There is one exception the above rule: When passed a datetime value with attached timezone information (a time-zone-aware datetime instance) the time filter will accept the timezone-related format specifiers 'e', 'O' , 'T' and 'Z'.

When used without a format string:

{{ value|time }}
...the formatting string defined in the TIME_FORMAT setting will be used, without applying any localization.

Changed in Django 1.7: 
The ability to receive and act on values with attached timezone information was added in Django 1.7.

timesince
Formats a date as the time since that date (e.g., “4 days, 6 hours”).

Takes an optional argument that is a variable containing the date to use as the comparison point (without the argument, the comparison point is now). For example, if blog_date is a date instance representing midnight on 1 June 2006, and comment_date is a date instance for 08:00 on 1 June 2006, then the following would return “8 hours”:

{{ blog_date|timesince:comment_date }}
Comparing offset-naive and offset-aware datetimes will return an empty string.

Minutes is the smallest unit used, and “0 minutes” will be returned for any date that is in the future relative to the comparison point.

timeuntil
Similar to timesince, except that it measures the time from now until the given date or datetime. For example, if today is 1 June 2006 and conference_date is a date instance holding 29 June 2006, then {{ conference_date|timeuntil }} will return “4 weeks”.

Takes an optional argument that is a variable containing the date to use as the comparison point (instead of now). If from_date contains 22 June 2006, then the following will return “1 week”:

{{ conference_date|timeuntil:from_date }}
Comparing offset-naive and offset-aware datetimes will return an empty string.

Minutes is the smallest unit used, and “0 minutes” will be returned for any date that is in the past relative to the comparison point.

title
Converts a string into titlecase by making words start with an uppercase character and the remaining characters lowercase. This tag makes no effort to keep “trivial words” in lowercase.

For example:

{{ value|title }}
If value is "my FIRST post", the output will be "My First Post".

truncatechars
Truncates a string if it is longer than the specified number of characters. Truncated strings will end with a translatable ellipsis sequence (”...”).

Argument: Number of characters to truncate to

For example:

{{ value|truncatechars:9 }}
If value is "Joel is a slug", the output will be "Joel i...".

truncatechars_html
New in Django 1.7. 
Similar to truncatechars, except that it is aware of HTML tags. Any tags that are opened in the string and not closed before the truncation point are closed immediately after the truncation.

For example:

{{ value|truncatechars_html:9 }}
If value is "<p>Joel is a slug</p>", the output will be "<p>Joel i...</p>".

Newlines in the HTML content will be preserved.

truncatewords
Truncates a string after a certain number of words.

Argument: Number of words to truncate after

For example:

{{ value|truncatewords:2 }}
If value is "Joel is a slug", the output will be "Joel is ...".

Newlines within the string will be removed.

truncatewords_html
Similar to truncatewords, except that it is aware of HTML tags. Any tags that are opened in the string and not closed before the truncation point, are closed immediately after the truncation.

This is less efficient than truncatewords, so should only be used when it is being passed HTML text.

For example:

{{ value|truncatewords_html:2 }}
If value is "<p>Joel is a slug</p>", the output will be "<p>Joel is ...</p>".

Newlines in the HTML content will be preserved.

unordered_list
Recursively takes a self-nested list and returns an HTML unordered list – WITHOUT opening and closing <ul> tags.

The list is assumed to be in the proper format. For example, if var contains ['States', ['Kansas', ['Lawrence', 'Topeka'], 'Illinois']], then {{ var|unordered_list }} would return:

<li>States
<ul>
        <li>Kansas
        <ul>
                <li>Lawrence</li>
                <li>Topeka</li>
        </ul>
        </li>
        <li>Illinois</li>
</ul>
</li>
Note: An older, more restrictive and verbose input format is also supported: ['States', [['Kansas', [['Lawrence', []], ['Topeka', []]]], ['Illinois', []]]],

upper
Converts a string into all uppercase.

For example:

{{ value|upper }}
If value is "Joel is a slug", the output will be "JOEL IS A SLUG".

urlencode
Escapes a value for use in a URL.

For example:

{{ value|urlencode }}
If value is "http://www.example.org/foo?a=b&c=d", the output will be "http%3A//www.example.org/foo%3Fa%3Db%26c%3Dd".

An optional argument containing the characters which should not be escaped can be provided.

If not provided, the ‘/’ character is assumed safe. An empty string can be provided when all characters should be escaped. For example:

{{ value|urlencode:"" }}
If value is "http://www.example.org/", the output will be "http%3A%2F%2Fwww.example.org%2F".

urlize
Converts URLs and email addresses in text into clickable links.

This template tag works on links prefixed with http://, https://, or www.. For example, http://goo.gl/aia1t will get converted but goo.gl/aia1t won’t.

It also supports domain-only links ending in one of the original top level domains (.com, .edu, .gov, .int, .mil, .net, and .org). For example, djangoproject.com gets converted.

Links can have trailing punctuation (periods, commas, close-parens) and leading punctuation (opening parens), and urlize will still do the right thing.

Links generated by urlize have a rel="nofollow" attribute added to them.

For example:

{{ value|urlize }}
If value is "Check out www.djangoproject.com", the output will be "Check out <a href="http://www.djangoproject.com" rel="nofollow">www.djangoproject.com</a>".

In addition to web links, urlize also converts email addresses into mailto: links. If value is "Send questions to foo@example.com", the output will be "Send questions to <a href="mailto:foo@example.com">foo@example.com</a>".

The urlize filter also takes an optional parameter autoescape. If autoescape is True, the link text and URLs will be escaped using Django’s built-in escape filter. The default value for autoescape is True.

Note

If urlize is applied to text that already contains HTML markup, things won’t work as expected. Apply this filter only to plain text.

urlizetrunc
Converts URLs and email addresses into clickable links just like urlize, but truncates URLs longer than the given character limit.

Argument: Number of characters that link text should be truncated to, including the ellipsis that’s added if truncation is necessary.

For example:

{{ value|urlizetrunc:15 }}
If value is "Check out www.djangoproject.com", the output would be 'Check out <a href="http://www.djangoproject.com" rel="nofollow">www.djangopr...</a>'.

As with urlize, this filter should only be applied to plain text.

wordcount
Returns the number of words.

For example:

{{ value|wordcount }}
If value is "Joel is a slug", the output will be 4.

wordwrap
Wraps words at specified line length.

Argument: number of characters at which to wrap the text

For example:

{{ value|wordwrap:5 }}
If value is Joel is a slug, the output would be:

Joel
is a
slug
yesno
Maps values for True, False, and (optionally) None, to the strings “yes”, “no”, “maybe”, or a custom mapping passed as a comma-separated list, and returns one of those strings according to the value:

For example:

{{ value|yesno:"yeah,no,maybe" }}
Value Argument Outputs 
True   yes 
True "yeah,no,maybe" yeah 
False "yeah,no,maybe" no 
None "yeah,no,maybe" maybe 
None "yeah,no" no (converts None to False if no mapping for None is given) 



Internationalization tags and filters
Django provides template tags and filters to control each aspect of internationalization in templates. They allow for granular control of translations, formatting, and time zone conversions.

i18n
This library allows specifying translatable text in templates. To enable it, set USE_I18N to True, then load it with {% load i18n %}.


l10n
This library provides control over the localization of values in templates. You only need to load the library using {% load l10n %}, but you’ll often set USE_L10N to True so that localization is active by default.

See Controlling localization in templates.

tz
This library provides control over time zone conversions in templates. Like l10n, you only need to load the library using {% load tz %}, but you’ll usually also set USE_TZ to True so that conversion to local time happens by default.

See Time zone aware output in templates.

Other tags and filters libraries
Django comes with a couple of other template-tag libraries that you have to enable explicitly in your INSTALLED_APPS setting and enable in your template with the {% load %} tag.

django.contrib.humanize
A set of Django template filters useful for adding a “human touch” to data.

django.contrib.webdesign
A collection of template tags that can be useful while designing a Web site, such as a generator of Lorem Ipsum text. 





static
To link to static files that are saved in STATIC_ROOT Django ships with a static template tag. You can use this regardless if you’re using RequestContext or not.

{% load static %}
<img src="{% static "images/hi.jpg" %}" alt="Hi!" />

It is also able to consume standard context variables, e.g. assuming a user_stylesheet variable is passed to the template:

{% load static %}
<link rel="stylesheet" href="{% static user_stylesheet %}" type="text/css" media="screen" />
If you’d like to retrieve a static URL without displaying it, you can use a slightly different call:

{% load static %}
{% static "images/hi.jpg" as myphoto %}
<img src="{{ myphoto }}"></img>
Note

The staticfiles contrib app also ships with a static template tag which uses staticfiles' STATICFILES_STORAGE to build the URL of the given path (rather than simply using urllib.parse.urljoin() with the STATIC_URL setting and the given path). Use that instead if you have an advanced use case such as using a cloud service to serve static files:

{% load static from staticfiles %}
<img src="{% static "images/hi.jpg" %}" alt="Hi!" />
get_static_prefix
You should prefer the static template tag, but if you need more control over exactly where and how STATIC_URL is injected into the template, you can use the get_static_prefix template tag:

{% load static %}
<img src="{% get_static_prefix %}images/hi.jpg" alt="Hi!" />
There’s also a second form you can use to avoid extra processing if you need the value multiple times:

{% load static %}
{% get_static_prefix as STATIC_PREFIX %}

<img src="{{ STATIC_PREFIX }}images/hi.jpg" alt="Hi!" />
<img src="{{ STATIC_PREFIX }}images/hi2.jpg" alt="Hello!" />
get_media_prefix
Similar to the get_static_prefix, get_media_prefix populates a template variable with the media prefix MEDIA_URL, e.g.:

{% load static %}
<body data-media-url="{% get_media_prefix %}">
By storing the value in a data attribute, we ensure it’s escaped appropriately if we want to use it in a JavaScript context








Tags {% tag %}



for
Loop over each item in an array. For example, to display a list of athletes provided in athlete_list:

<ul>
{% for athlete in athlete_list %}
    <li>{{ athlete.name }}</li>
{% endfor %}
</ul>


if, elif, and else
Evaluates a variable, and if that variable is “true” the contents of the block are displayed:

{% if athlete_list %}
    Number of athletes: {{ athlete_list|length }}
{% elif athlete_in_locker_room_list %}
    Athletes should be out of the locker room soon!
{% else %}
    No athletes.
{% endif %}
In the above, if athlete_list is not empty, the number of athletes will be displayed by the {{ athlete_list|length }} variable. Otherwise, if athlete_in_locker_room_list is not empty, the message “Athletes should be out...” will be displayed. If both lists are empty, “No athletes.” will be displayed.



You can also use filters and various operators in the if tag:

{% if athlete_list|length > 1 %}
   Team: {% for athlete in athlete_list %} ... {% endfor %}
{% else %}
   Athlete: {{ athlete_list.0.name }}
{% endif %}

While the above example works, be aware that most template filters return strings, so mathematical comparisons using filters will generally not work as you expect.
length is an exception.



Built-in tag reference
autoescape
Controls the current auto-escaping behavior. This tag takes either on or off as an argument and that determines whether auto-escaping is in effect inside the block. The block is closed with an endautoescape ending tag.

When auto-escaping is in effect, all variable content has HTML escaping applied to it before placing the result into the output (but after any filters have been applied). This is equivalent to manually applying the escape filter to each variable.

The only exceptions are variables that are already marked as “safe” from escaping, either by the code that populated the variable, or because it has had the safe or escape filters applied.

Sample usage:

{% autoescape on %}
    {{ body }}
{% endautoescape %}
block
Defines a block that can be overridden by child templates. See Template inheritance for more information.

comment
Ignores everything between {% comment %} and {% endcomment %}. An optional note may be inserted in the first tag. For example, this is useful when commenting out code for documenting why the code was disabled.

Sample usage:

<p>Rendered text with {{ pub_date|date:"c" }}</p>
{% comment "Optional note" %}
    <p>Commented out text with {{ create_date|date:"c" }}</p>
{% endcomment %}
comment tags cannot be nested.

csrf_token
This tag is used for CSRF protection, as described in the documentation for Cross Site Request Forgeries.

cycle
Produces one of its arguments each time this tag is encountered. The first argument is produced on the first encounter, the second argument on the second encounter, and so forth. Once all arguments are exhausted, the tag cycles to the first argument and produces it again.

This tag is particularly useful in a loop:

{% for o in some_list %}
    <tr class="{% cycle 'row1' 'row2' %}">
        ...
    </tr>
{% endfor %}
The first iteration produces HTML that refers to class row1, the second to row2, the third to row1 again, and so on for each iteration of the loop.

You can use variables, too. For example, if you have two template variables, rowvalue1 and rowvalue2, you can alternate between their values like this:

{% for o in some_list %}
    <tr class="{% cycle rowvalue1 rowvalue2 %}">
        ...
    </tr>
{% endfor %}
Note that the variables included in the cycle will not be escaped. Any HTML or Javascript code contained in the printed variable will be rendered as-is, which could potentially lead to security issues. So either make sure that you trust their values or use explicit escaping like this:

{% for o in some_list %}
    <tr class="{% filter force_escape %}{% cycle rowvalue1 rowvalue2 %}{% endfilter %}">
        ...
    </tr>
{% endfor %}
You can mix variables and strings:

{% for o in some_list %}
    <tr class="{% cycle 'row1' rowvalue2 'row3' %}">
        ...
    </tr>
{% endfor %}
In some cases you might want to refer to the current value of a cycle without advancing to the next value. To do this, just give the {% cycle %} tag a name, using “as”, like this:

{% cycle 'row1' 'row2' as rowcolors %}
From then on, you can insert the current value of the cycle wherever you’d like in your template by referencing the cycle name as a context variable. If you want to move the cycle to the next value independently of the original cycle tag, you can use another cycle tag and specify the name of the variable. So, the following template:

<tr>
    <td class="{% cycle 'row1' 'row2' as rowcolors %}">...</td>
    <td class="{{ rowcolors }}">...</td>
</tr>
<tr>
    <td class="{% cycle rowcolors %}">...</td>
    <td class="{{ rowcolors }}">...</td>
</tr>
would output:

<tr>
    <td class="row1">...</td>
    <td class="row1">...</td>
</tr>
<tr>
    <td class="row2">...</td>
    <td class="row2">...</td>
</tr>
You can use any number of values in a cycle tag, separated by spaces. Values enclosed in single quotes (') or double quotes (") are treated as string literals, while values without quotes are treated as template variables.

By default, when you use the as keyword with the cycle tag, the usage of {% cycle %} that initiates the cycle will itself produce the first value in the cycle. This could be a problem if you want to use the value in a nested loop or an included template. If you only want to declare the cycle but not produce the first value, you can add a silent keyword as the last keyword in the tag. For example:

{% for obj in some_list %}
    {% cycle 'row1' 'row2' as rowcolors silent %}
    <tr class="{{ rowcolors }}">{% include "subtemplate.html" %}</tr>
{% endfor %}
This will output a list of <tr> elements with class alternating between row1 and row2. The subtemplate will have access to rowcolors in its context and the value will match the class of the <tr> that encloses it. If the silent keyword were to be omitted, row1 and row2 would be emitted as normal text, outside the <tr> element.

When the silent keyword is used on a cycle definition, the silence automatically applies to all subsequent uses of that specific cycle tag. The following template would output nothing, even though the second call to {% cycle %} doesn’t specify silent:

{% cycle 'row1' 'row2' as rowcolors silent %}
{% cycle rowcolors %}
For backward compatibility, the {% cycle %} tag supports the much inferior old syntax from previous Django versions. You shouldn’t use this in any new projects, but for the sake of the people who are still using it, here’s what it looks like:

{% cycle row1,row2,row3 %}
In this syntax, each value gets interpreted as a literal string, and there’s no way to specify variable values. Or literal commas. Or spaces. Did we mention you shouldn’t use this syntax in any new projects?

Changed in Django 1.6. 
To improve safety, future versions of cycle will automatically escape their output. You’re encouraged to activate this behavior by loading cycle from the future template library:

{% load cycle from future %}
When using the future version, you can disable auto-escaping with:

{% for o in some_list %}
    <tr class="{% autoescape off %}{% cycle rowvalue1 rowvalue2 %}{% endautoescape %}">
        ...
    </tr>
{% endfor %}
debug
Outputs a whole load of debugging information, including the current context and imported modules.

extends
Signals that this template extends a parent template.

This tag can be used in two ways:

•{% extends "base.html" %} (with quotes) uses the literal value "base.html" as the name of the parent template to extend.
•{% extends variable %} uses the value of variable. If the variable evaluates to a string, Django will use that string as the name of the parent template. If the variable evaluates to a Template object, Django will use that object as the parent template.
See Template inheritance for more information.

filter
Filters the contents of the block through one or more filters. Multiple filters can be specified with pipes and filters can have arguments, just as in variable syntax.

Note that the block includes all the text between the filter and endfilter tags.

Sample usage:

{% filter force_escape|lower %}
    This text will be HTML-escaped, and will appear in all lowercase.
{% endfilter %}
Note

The escape and safe filters are not acceptable arguments. Instead, use the autoescape tag to manage autoescaping for blocks of template code.

firstof
Outputs the first argument variable that is not False. This tag does not auto-escape variable values.

Outputs nothing if all the passed variables are False.

Sample usage:

{% firstof var1 var2 var3 %}
This is equivalent to:

{% if var1 %}
    {{ var1|safe }}
{% elif var2 %}
    {{ var2|safe }}
{% elif var3 %}
    {{ var3|safe }}
{% endif %}
You can also use a literal string as a fallback value in case all passed variables are False:

{% firstof var1 var2 var3 "fallback value" %}
Note that currently the variables included in the firstof tag will not be escaped. Any HTML or Javascript code contained in the printed variable will be rendered as-is, which could potentially lead to security issues. If you need to escape the variables in the firstof tag, you must do so explicitly:

{% filter force_escape %}
    {% firstof var1 var2 var3 "fallback value" %}
{% endfilter %}
Changed in Django 1.6: 
To improve safety, future versions of firstof will automatically escape their output. You’re encouraged to activate this behavior by loading firstof from the future template library:

{% load firstof from future %}
When using the future version, you can disable auto-escaping with:

{% autoescape off %}
    {% firstof var1 var2 var3 "<strong>fallback value</strong>" %}
{% endautoescape %}
Or if only some variables should be escaped, you can use:

{% firstof var1 var2|safe var3 "<strong>fallback value</strong>"|safe %}
for
Loops over each item in an array, making the item available in a context variable. For example, to display a list of athletes provided in athlete_list:

<ul>
{% for athlete in athlete_list %}
    <li>{{ athlete.name }}</li>
{% endfor %}
</ul>
You can loop over a list in reverse by using {% for obj in list reversed %}.

If you need to loop over a list of lists, you can unpack the values in each sublist into individual variables. For example, if your context contains a list of (x,y) coordinates called points, you could use the following to output the list of points:

{% for x, y in points %}
    There is a point at {{ x }},{{ y }}
{% endfor %}
This can also be useful if you need to access the items in a dictionary. For example, if your context contained a dictionary data, the following would display the keys and values of the dictionary:

{% for key, value in data.items %}
    {{ key }}: {{ value }}
{% endfor %}
The for loop sets a number of variables available within the loop:

Variable Description 
forloop.counter The current iteration of the loop (1-indexed) 
forloop.counter0 The current iteration of the loop (0-indexed) 
forloop.revcounter The number of iterations from the end of the loop (1-indexed) 
forloop.revcounter0 The number of iterations from the end of the loop (0-indexed) 
forloop.first True if this is the first time through the loop 
forloop.last True if this is the last time through the loop 
forloop.parentloop For nested loops, this is the loop surrounding the current one 

for ... empty
The for tag can take an optional {% empty %} clause whose text is displayed if the given array is empty or could not be found:

<ul>
{% for athlete in athlete_list %}
    <li>{{ athlete.name }}</li>
{% empty %}
    <li>Sorry, no athletes in this list.</li>
{% endfor %}
</ul>
The above is equivalent to – but shorter, cleaner, and possibly faster than – the following:

<ul>
  {% if athlete_list %}
    {% for athlete in athlete_list %}
      <li>{{ athlete.name }}</li>
    {% endfor %}
  {% else %}
    <li>Sorry, no athletes in this list.</li>
  {% endif %}
</ul>
if
The {% if %} tag evaluates a variable, and if that variable is “true” (i.e. exists, is not empty, and is not a false boolean value) the contents of the block are output:

{% if athlete_list %}
    Number of athletes: {{ athlete_list|length }}
{% elif athlete_in_locker_room_list %}
    Athletes should be out of the locker room soon!
{% else %}
    No athletes.
{% endif %}
In the above, if athlete_list is not empty, the number of athletes will be displayed by the {{ athlete_list|length }} variable.

As you can see, the if tag may take one or several {% elif %} clauses, as well as an {% else %} clause that will be displayed if all previous conditions fail. These clauses are optional.

Boolean operators
if tags may use and, or or not to test a number of variables or to negate a given variable:

{% if athlete_list and coach_list %}
    Both athletes and coaches are available.
{% endif %}

{% if not athlete_list %}
    There are no athletes.
{% endif %}

{% if athlete_list or coach_list %}
    There are some athletes or some coaches.
{% endif %}

{% if not athlete_list or coach_list %}
    There are no athletes or there are some coaches (OK, so
    writing English translations of boolean logic sounds
    stupid; it's not our fault).
{% endif %}

{% if athlete_list and not coach_list %}
    There are some athletes and absolutely no coaches.
{% endif %}
Use of both and and or clauses within the same tag is allowed, with and having higher precedence than or e.g.:

{% if athlete_list and coach_list or cheerleader_list %}
will be interpreted like:

if (athlete_list and coach_list) or cheerleader_list
Use of actual parentheses in the if tag is invalid syntax. If you need them to indicate precedence, you should use nested if tags.

if tags may also use the operators ==, !=, <, >, <=, >= and in which work as follows:

== operator
Equality. Example:

{% if somevar == "x" %}
  This appears if variable somevar equals the string "x"
{% endif %}
!= operator
Inequality. Example:

{% if somevar != "x" %}
  This appears if variable somevar does not equal the string "x",
  or if somevar is not found in the context
{% endif %}
< operator
Less than. Example:

{% if somevar < 100 %}
  This appears if variable somevar is less than 100.
{% endif %}
> operator
Greater than. Example:

{% if somevar > 0 %}
  This appears if variable somevar is greater than 0.
{% endif %}
<= operator
Less than or equal to. Example:

{% if somevar <= 100 %}
  This appears if variable somevar is less than 100 or equal to 100.
{% endif %}
>= operator
Greater than or equal to. Example:

{% if somevar >= 1 %}
  This appears if variable somevar is greater than 1 or equal to 1.
{% endif %}
in operator
Contained within. This operator is supported by many Python containers to test whether the given value is in the container. The following are some examples of how x in y will be interpreted:

{% if "bc" in "abcdef" %}
  This appears since "bc" is a substring of "abcdef"
{% endif %}

{% if "hello" in greetings %}
  If greetings is a list or set, one element of which is the string
  "hello", this will appear.
{% endif %}

{% if user in users %}
  If users is a QuerySet, this will appear if user is an
  instance that belongs to the QuerySet.
{% endif %}
not in operator
Not contained within. This is the negation of the in operator.

The comparison operators cannot be ‘chained’ like in Python or in mathematical notation. For example, instead of using:

{% if a > b > c %}  (WRONG)
you should use:

{% if a > b and b > c %}
Filters
You can also use filters in the if expression. For example:

{% if messages|length >= 100 %}
   You have lots of messages today!
{% endif %}
Complex expressions
All of the above can be combined to form complex expressions. For such expressions, it can be important to know how the operators are grouped when the expression is evaluated - that is, the precedence rules. The precedence of the operators, from lowest to highest, is as follows:

•or
•and
•not
•in
•==, !=, <, >, <=, >=
(This follows Python exactly). So, for example, the following complex if tag:

{% if a == b or c == d and e %}
...will be interpreted as:

(a == b) or ((c == d) and e)
If you need different precedence, you will need to use nested if tags. Sometimes that is better for clarity anyway, for the sake of those who do not know the precedence rules.

ifchanged
Check if a value has changed from the last iteration of a loop.

The {% ifchanged %} block tag is used within a loop. It has two possible uses.

1.Checks its own rendered contents against its previous state and only displays the content if it has changed. For example, this displays a list of days, only displaying the month if it changes:

<h1>Archive for {{ year }}</h1>

{% for date in days %}
    {% ifchanged %}<h3>{{ date|date:"F" }}</h3>{% endifchanged %}
    <a href="{{ date|date:"M/d"|lower }}/">{{ date|date:"j" }}</a>
{% endfor %}
2.If given one or more variables, check whether any variable has changed. For example, the following shows the date every time it changes, while showing the hour if either the hour or the date has changed:

{% for date in days %}
    {% ifchanged date.date %} {{ date.date }} {% endifchanged %}
    {% ifchanged date.hour date.date %}
        {{ date.hour }}
    {% endifchanged %}
{% endfor %}
The ifchanged tag can also take an optional {% else %} clause that will be displayed if the value has not changed:

{% for match in matches %}
    <div style="background-color:
        {% ifchanged match.ballot_id %}
            {% cycle "red" "blue" %}
        {% else %}
            gray
        {% endifchanged %}
    ">{{ match }}</div>
{% endfor %}
ifequal
Output the contents of the block if the two arguments equal each other.

Example:

{% ifequal user.pk comment.user_id %}
    ...
{% endifequal %}
As in the if tag, an {% else %} clause is optional.

The arguments can be hard-coded strings, so the following is valid:

{% ifequal user.username "adrian" %}
    ...
{% endifequal %}
An alternative to the ifequal tag is to use the if tag and the == operator.

ifnotequal
Just like ifequal, except it tests that the two arguments are not equal.

An alternative to the ifnotequal tag is to use the if tag and the != operator.

include
Loads a template and renders it with the current context. This is a way of “including” other templates within a template.

The template name can either be a variable or a hard-coded (quoted) string, in either single or double quotes.

This example includes the contents of the template "foo/bar.html":

{% include "foo/bar.html" %}
This example includes the contents of the template whose name is contained in the variable template_name:

{% include template_name %}
Changed in Django 1.7: 
The variable may also be any object with a render() method that accepts a context. This allows you to reference a compiled Template in your context.

An included template is rendered within the context of the template that includes it. This example produces the output "Hello, John!":

•Context: variable person is set to "John" and variable greeting is set to "Hello".

•Template:

{% include "name_snippet.html" %}
•The name_snippet.html template:

{{ greeting }}, {{ person|default:"friend" }}!
You can pass additional context to the template using keyword arguments:

{% include "name_snippet.html" with person="Jane" greeting="Hello" %}
If you want to render the context only with the variables provided (or even no variables at all), use the only option. No other variables are available to the included template:

{% include "name_snippet.html" with greeting="Hi" only %}
Note

The include tag should be considered as an implementation of “render this subtemplate and include the HTML”, not as “parse this subtemplate and include its contents as if it were part of the parent”. This means that there is no shared state between included templates – each include is a completely independent rendering process.

Blocks are evaluated before they are included. This means that a template that includes blocks from another will contain blocks that have already been evaluated and rendered - not blocks that can be overridden by, for example, an extending template.

See also: {% ssi %}.

load
Loads a custom template tag set.

For example, the following template would load all the tags and filters registered in somelibrary and otherlibrary located in package package:

{% load somelibrary package.otherlibrary %}
You can also selectively load individual filters or tags from a library, using the from argument. In this example, the template tags/filters named foo and bar will be loaded from somelibrary:

{% load foo bar from somelibrary %}
See Custom tag and filter libraries for more information.

now
Displays the current date and/or time, using a format according to the given string. Such string can contain format specifiers characters as described in the date filter section.

Example:

It is {% now "jS F Y H:i" %}
Note that you can backslash-escape a format string if you want to use the “raw” value. In this example, both “o” and “f” are backslash-escaped, because otherwise each is a format string that displays the year and the time, respectively:

It is the {% now "jS \o\f F" %}
This would display as “It is the 4th of September”.

Note

The format passed can also be one of the predefined ones DATE_FORMAT, DATETIME_FORMAT, SHORT_DATE_FORMAT or SHORT_DATETIME_FORMAT. The predefined formats may vary depending on the current locale and if Format localization is enabled, e.g.:

It is {% now "SHORT_DATETIME_FORMAT" %}
regroup
Regroups a list of alike objects by a common attribute.

This complex tag is best illustrated by way of an example: say that “places” is a list of cities represented by dictionaries containing "name", "population", and "country" keys:

cities = [
    {'name': 'Mumbai', 'population': '19,000,000', 'country': 'India'},
    {'name': 'Calcutta', 'population': '15,000,000', 'country': 'India'},
    {'name': 'New York', 'population': '20,000,000', 'country': 'USA'},
    {'name': 'Chicago', 'population': '7,000,000', 'country': 'USA'},
    {'name': 'Tokyo', 'population': '33,000,000', 'country': 'Japan'},
]
...and you’d like to display a hierarchical list that is ordered by country, like this:

•India
?Mumbai: 19,000,000
?Calcutta: 15,000,000
•USA
?New York: 20,000,000
?Chicago: 7,000,000
•Japan
?Tokyo: 33,000,000
You can use the {% regroup %} tag to group the list of cities by country. The following snippet of template code would accomplish this:

{% regroup cities by country as country_list %}

<ul>
{% for country in country_list %}
    <li>{{ country.grouper }}
    <ul>
        {% for item in country.list %}
          <li>{{ item.name }}: {{ item.population }}</li>
        {% endfor %}
    </ul>
    </li>
{% endfor %}
</ul>
Let’s walk through this example. {% regroup %} takes three arguments: the list you want to regroup, the attribute to group by, and the name of the resulting list. Here, we’re regrouping the cities list by the country attribute and calling the result country_list.

{% regroup %} produces a list (in this case, country_list) of group objects. Each group object has two attributes:

•grouper – the item that was grouped by (e.g., the string “India” or “Japan”).
•list – a list of all items in this group (e.g., a list of all cities with country=’India’).
Note that {% regroup %} does not order its input! Our example relies on the fact that the cities list was ordered by country in the first place. If the cities list did not order its members by country, the regrouping would naively display more than one group for a single country. For example, say the cities list was set to this (note that the countries are not grouped together):

cities = [
    {'name': 'Mumbai', 'population': '19,000,000', 'country': 'India'},
    {'name': 'New York', 'population': '20,000,000', 'country': 'USA'},
    {'name': 'Calcutta', 'population': '15,000,000', 'country': 'India'},
    {'name': 'Chicago', 'population': '7,000,000', 'country': 'USA'},
    {'name': 'Tokyo', 'population': '33,000,000', 'country': 'Japan'},
]
With this input for cities, the example {% regroup %} template code above would result in the following output:

•India
?Mumbai: 19,000,000
•USA
?New York: 20,000,000
•India
?Calcutta: 15,000,000
•USA
?Chicago: 7,000,000
•Japan
?Tokyo: 33,000,000
The easiest solution to this gotcha is to make sure in your view code that the data is ordered according to how you want to display it.

Another solution is to sort the data in the template using the dictsort filter, if your data is in a list of dictionaries:

{% regroup cities|dictsort:"country" by country as country_list %}
Grouping on other properties
Any valid template lookup is a legal grouping attribute for the regroup tag, including methods, attributes, dictionary keys and list items. For example, if the “country” field is a foreign key to a class with an attribute “description,” you could use:

{% regroup cities by country.description as country_list %}
Or, if country is a field with choices, it will have a get_FOO_display() method available as an attribute, allowing you to group on the display string rather than the choices key:

{% regroup cities by get_country_display as country_list %}
{{ country.grouper }} will now display the value fields from the choices set rather than the keys.

spaceless
Removes whitespace between HTML tags. This includes tab characters and newlines.

Example usage:

{% spaceless %}
    <p>
        <a href="foo/">Foo</a>
    </p>
{% endspaceless %}
This example would return this HTML:

<p><a href="foo/">Foo</a></p>
Only space between tags is removed – not space between tags and text. In this example, the space around Hello won’t be stripped:

{% spaceless %}
    <strong>
        Hello
    </strong>
{% endspaceless %}
ssi
Outputs the contents of a given file into the page.

Like a simple include tag, {% ssi %} includes the contents of another file – which must be specified using an absolute path – in the current page:

{% ssi '/home/html/ljworld.com/includes/right_generic.html' %}
The first parameter of ssi can be a quoted literal or any other context variable.

If the optional parsed parameter is given, the contents of the included file are evaluated as template code, within the current context:

{% ssi '/home/html/ljworld.com/includes/right_generic.html' parsed %}
Note that if you use {% ssi %}, you’ll need to define ALLOWED_INCLUDE_ROOTS in your Django settings, as a security measure.

Note

With the ssi tag and the parsed parameter there is no shared state between files – each include is a completely independent rendering process. This means it’s not possible for example to define blocks or alter the context in the current page using the included file.

See also: {% include %}.

templatetag
Outputs one of the syntax characters used to compose template tags.

Since the template system has no concept of “escaping”, to display one of the bits used in template tags, you must use the {% templatetag %} tag.

The argument tells which template bit to output:

Argument Outputs 
openblock {% 
closeblock %} 
openvariable {{ 
closevariable }} 
openbrace { 
closebrace } 
opencomment {# 
closecomment #} 

Sample usage:

{% templatetag openblock %} url 'entry_list' {% templatetag closeblock %}
url
Returns an absolute path reference (a URL without the domain name) matching a given view function and optional parameters.

Changed in Django 1.6: 
Any special characters in the resulting path will be encoded using iri_to_uri().

This is a way to output links without violating the DRY principle by having to hard-code URLs in your templates:

{% url 'path.to.some_view' v1 v2 %}
The first argument is a path to a view function in the format package.package.module.function. It can be a quoted literal or any other context variable. Additional arguments are optional and should be space-separated values that will be used as arguments in the URL. The example above shows passing positional arguments. Alternatively you may use keyword syntax:

{% url 'path.to.some_view' arg1=v1 arg2=v2 %}
Do not mix both positional and keyword syntax in a single call. All arguments required by the URLconf should be present.

For example, suppose you have a view, app_views.client, whose URLconf takes a client ID (here, client() is a method inside the views file app_views.py). The URLconf line might look like this:

('^client/(\d+)/$', 'app_views.client')
If this app’s URLconf is included into the project’s URLconf under a path such as this:

('^clients/', include('project_name.app_name.urls'))
...then, in a template, you can create a link to this view like this:

{% url 'app_views.client' client.id %}
The template tag will output the string /clients/client/123/.

If you’re using named URL patterns, you can refer to the name of the pattern in the url tag instead of using the path to the view.

Note that if the URL you’re reversing doesn’t exist, you’ll get an NoReverseMatch exception raised, which will cause your site to display an error page.

If you’d like to retrieve a URL without displaying it, you can use a slightly different call:

{% url 'path.to.view' arg arg2 as the_url %}

<a href="{{ the_url }}">I'm linking to {{ the_url }}</a>
The scope of the variable created by the as var syntax is the {% block %} in which the {% url %} tag appears.

This {% url ... as var %} syntax will not cause an error if the view is missing. In practice you’ll use this to link to views that are optional:

{% url 'path.to.view' as the_url %}
{% if the_url %}
  <a href="{{ the_url }}">Link to optional stuff</a>
{% endif %}
If you’d like to retrieve a namespaced URL, specify the fully qualified name:

{% url 'myapp:view-name' %}
This will follow the normal namespaced URL resolution strategy, including using any hints provided by the context as to the current application.

Warning

Don’t forget to put quotes around the function path or pattern name, otherwise the value will be interpreted as a context variable!

verbatim
Stops the template engine from rendering the contents of this block tag.

A common use is to allow a Javascript template layer that collides with Django’s syntax. For example:

{% verbatim %}
    {{if dying}}Still alive.{{/if}}
{% endverbatim %}
You can also designate a specific closing tag, allowing the use of {% endverbatim %} as part of the unrendered contents:

{% verbatim myblock %}
    Avoid template rendering via the {% verbatim %}{% endverbatim %} block.
{% endverbatim myblock %}
widthratio
For creating bar charts and such, this tag calculates the ratio of a given value to a maximum value, and then applies that ratio to a constant.

For example:

<img src="bar.png" alt="Bar"
     height="10" width="{% widthratio this_value max_value max_width %}" />
If this_value is 175, max_value is 200, and max_width is 100, the image in the above example will be 88 pixels wide (because 175/200 = .875; .875 * 100 = 87.5 which is rounded up to 88).

In some cases you might want to capture the result of widthratio in a variable. It can be useful, for instance, in a blocktrans like this:

{% widthratio this_value max_value max_width as width %}
{% blocktrans %}The width is: {{ width }}{% endblocktrans %}
Changed in Django 1.7: 
The ability to use “as” with this tag like in the example above was added.

with
Caches a complex variable under a simpler name. This is useful when accessing an “expensive” method (e.g., one that hits the database) multiple times.

For example:

{% with total=business.employees.count %}
    {{ total }} employee{{ total|pluralize }}
{% endwith %}
The populated variable (in the example above, total) is only available between the {% with %} and {% endwith %} tags.

You can assign more than one context variable:

{% with alpha=1 beta=2 %}
    ...
{% endwith %}
Note

The previous more verbose format is still supported: {% with business.employees.count as total %}














Comments
To comment-out part of a line in a template, use the comment syntax: {# #}.

For example, this template would render as 'hello':

{# greeting #}hello
A comment can contain any template code, invalid or not. For example:

{# {% if foo %}bar{% else %} #}
This syntax can only be used for single-line comments (no newlines are permitted between the {# and #} delimiters). If you need to comment out a multiline portion of the template, see the comment tag.




Template inheritance

Example:base.html:

<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="style.css" />
    <title>{% block title %}My amazing site{% endblock %}</title>
</head>

<body>
    <div id="sidebar">
        {% block sidebar %}
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/blog/">Blog</a></li>
        </ul>
        {% endblock %}
    </div>

    <div id="content">
        {% block content %}{% endblock %}
    </div>
</body>
</html>

child.html:

{% extends "base.html" %}

{% block title %}My amazing blog{% endblock %}

{% block content %}
{% for entry in blog_entries %}
    <h2>{{ entry.title }}</h2>
    <p>{{ entry.body }}</p>
{% endfor %}
{% endblock %}

Then output=>

<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="style.css" />
    <title>My amazing blog</title>
</head>

<body>
    <div id="sidebar">
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/blog/">Blog</a></li>
        </ul>
    </div>

    <div id="content">
        <h2>Entry one</h2>
        <p>This is my first entry.</p>

        <h2>Entry two</h2>
        <p>This is my second entry.</p>
    </div>
</body>
</html>


Recomended Practice:
You can use as many levels of inheritance as needed. One common way of using inheritance is the following three-level approach:

•Create a base.html template that holds the main look-and-feel of your site.
•Create a base_SECTIONNAME.html template for each “section” of your site. For example, base_news.html, base_sports.html. These templates all extend base.html and include section-specific styles/design.
•Create individual templates for each type of page, such as a news article or blog entry. These templates extend the appropriate section template.



Tips

•If you use {% extends %} in a template, it must be the first template tag in that template. Template inheritance won’t work, otherwise.

•More {% block %} tags in your base templates are better. Remember, child templates don’t have to define all parent blocks, so you can fill in reasonable defaults in a number of blocks, then only define the ones you need later. It’s better to have more hooks than fewer hooks.

•If you find yourself duplicating content in a number of templates, it probably means you should move that content to a {% block %} in a parent template.

•If you need to get the content of the block from the parent template, the {{ block.super }} variable will do work. This is useful if you want to add to the contents of a parent block instead of completely overriding it. Data inserted using {{ block.super }} will not be automatically escaped (see the next section), since it was already escaped, if necessary, in the parent template.

•For extra readability, you can optionally give a name to your {% endblock %} tag. For example:

{% block content %}
...
{% endblock content %}
In larger templates, this technique helps you see which {% block %} tags are being closed.

 note that you can’t define multiple block tags with the same name in the same template. 



Automatic HTML escaping
When generating HTML from templates, there’s always a risk that a variable will include characters that affect the resulting HTML. For example, consider this template fragment:

Hello, {{ name }}.
At first, this seems like a harmless way to display a user’s name, but consider what would happen if the user entered their name as this:

<script>alert('hello')</script>
With this name value, the template would be rendered as:

Hello, <script>alert('hello')</script>
...which means the browser would pop-up a JavaScript alert box!

Similarly, what if the name contained a '<' symbol, like this?

<b>username
That would result in a rendered template like this:

Hello, <b>username
...which, in turn, would result in the remainder of the Web page being bolded!


To avoid this problem, you have two options:

•One, you can make sure to run each untrusted variable through the escape filter , which converts potentially harmful HTML characters to unharmful ones. This was the default solution in Django for its first few years, but the problem is that it puts the onus on you, the developer / template author, to ensure you’re escaping everything. It’s easy to forget to escape data.
•Two, you can take advantage of Django’s automatic HTML escaping. 


By default in Django, every template automatically escapes the output of every variable tag. Specifically, these five characters are escaped:

•< is converted to &lt;
•> is converted to &gt;
•' (single quote) is converted to &#39;
•" (double quote) is converted to &quot;
•& is converted to &amp;



How to turn it off

For individual variables
To disable auto-escaping for an individual variable, use the safe filter:

This will be escaped: {{ data }}
This will not be escaped: {{ data|safe }}

In this example, if data contains '<b>', the output will be:

This will be escaped: &lt;b&gt;
This will not be escaped: <b>


For template blocks
To control auto-escaping for a template, wrap the template (or just a particular section of the template) in the autoescape tag, like so:

{% autoescape off %}
    Hello {{ name }}
{% endautoescape %}

Auto-escaping is on by default. Hello {{ name }}

{% autoescape off %}
    This will not be auto-escaped: {{ data }}.

    Nor this: {{ other_data }}
    {% autoescape on %}
        Auto-escaping applies again: {{ name }}
    {% endautoescape %}
{% endautoescape %}


The auto-escaping tag passes its effect onto templates that extend the current one as well as templates included via the include tag, just like all block tags. For example:

base.html
{% autoescape off %}
<h1>{% block title %}{% endblock %}</h1>
{% block content %}
{% endblock %}
{% endautoescape %}

child.html
{% extends "base.html" %}
{% block title %}This &amp; that{% endblock %}
{% block content %}{{ greeting }}{% endblock %}
Because auto-escaping is turned off in the base template, it will also be turned off in the child template, resulting in the following rendered HTML when the greeting variable contains the string <b>Hello!</b>:

<h1>This &amp; that</h1>
<b>Hello!</b>


If you’re creating a template that might be used in situations where you’re not sure whether auto-escaping is enabled, then add an escape filter to any variable that needs escaping. When auto-escaping is on, there’s no danger of the escape filter double-escaping data – the escape filter does not affect auto-escaped variables.



String literals and automatic escaping
As we mentioned earlier, filter arguments can be strings:

{{ data|default:"This is a string literal." }}
All string literals are inserted without any automatic escaping into the template – they act as if they were all passed through the safe filter. 

This means you would write

{{ data|default:"3 &lt; 2" }}
...rather than:

{{ data|default:"3 < 2" }}  {# Bad! Don't do this. #}
Note: The variable’s contents are still automatically escaped, if necessary, because they’re beyond the control of the template author.



Accessing method calls
Most method calls attached to objects are also available from within templates. This means that templates have access to much more than just class attributes (like field names) and variables passed in from views. 
For example, the Django ORM provides the “entry_set” syntax for finding a collection of objects related on a foreign key. Therefore, given a model called “comment” with a foreign key relationship to a model called “task” you can loop through all comments attached to a given task like this:

{% for comment in task.comment_set.all %}
    {{ comment }}
{% endfor %}

Similarly, QuerySets provide a count() method to count the number of objects they contain. Therefore, you can obtain a count of all comments related to the current task with:

{{ task.comment_set.all.count }}

And of course you can easily access methods you’ve explicitly defined on your own models:

models.py
class Task(models.Model):
    def foo(self):
        return "bar"
template.html
{{ task.foo }}

it is not possible to pass arguments to method calls accessed from within templates. Data should be calculated in views, then passed to templates for display.



Custom tag and filter libraries
Certain applications provide custom tag and filter libraries. To access them in a template, ensure the application is in INSTALLED_APPS (we’d add 'django.contrib.humanize' for this example), and then use the load tag in a template:

{% load humanize %}

{{ 45000|intcomma }}

In the above, the load tag loads the humanize tag library, which then makes the intcomma filter available for use. 
If you’ve enabled django.contrib.admindocs, you can consult the documentation area in your admin to find the list of custom libraries in your installation.

The load tag can take multiple library names, separated by spaces. Example:

{% load humanize i18n %}


Custom libraries and template inheritance
When you load a custom tag or filter library, the tags/filters are only made available to the current template – not any parent or child templates along the template-inheritance path.

For example, if a template foo.html has {% load humanize %}, a child template (e.g., one that has {% extends "foo.html" %}) will not have access to the humanize template tags and filters. The child template is responsible for its own {% load humanize %}.






django.contrib.humanize
A set of Django template filters useful for adding a “human touch” to data.

To activate these filters, add 'django.contrib.humanize' to your INSTALLED_APPS setting. Once you’ve done that, use {% load humanize %} in a template, and you’ll have access to the following filters.

apnumber
For numbers 1-9, returns the number spelled out. Otherwise, returns the number. This follows Associated Press style.

Examples:

•1 becomes one.
•2 becomes two.
•10 becomes 10.
You can pass in either an integer or a string representation of an integer.

intcomma
Converts an integer to a string containing commas every three digits.

Examples:

•4500 becomes 4,500.
•45000 becomes 45,000.
•450000 becomes 450,000.
•4500000 becomes 4,500,000.
Format localization will be respected if enabled, e.g. with the 'de' language:

•45000 becomes '45.000'.
•450000 becomes '450.000'.
You can pass in either an integer or a string representation of an integer.

intword
Converts a large integer to a friendly text representation. Works best for numbers over 1 million.

Examples:

•1000000 becomes 1.0 million.
•1200000 becomes 1.2 million.
•1200000000 becomes 1.2 billion.
Values up to 10^100 (Googol) are supported.

Format localization will be respected if enabled, e.g. with the 'de' language:

•1000000 becomes '1,0 Million'.
•1200000 becomes '1,2 Million'.
•1200000000 becomes '1,2 Milliarden'.
You can pass in either an integer or a string representation of an integer.

naturalday
For dates that are the current day or within one day, return “today”, “tomorrow” or “yesterday”, as appropriate. Otherwise, format the date using the passed in format string.

Argument: Date formatting string as described in the date tag.

Examples (when ‘today’ is 17 Feb 2007):

•16 Feb 2007 becomes yesterday.
•17 Feb 2007 becomes today.
•18 Feb 2007 becomes tomorrow.
•Any other day is formatted according to given argument or the DATE_FORMAT setting if no argument is given.
naturaltime
For datetime values, returns a string representing how many seconds, minutes or hours ago it was – falling back to the timesince format if the value is more than a day old. In case the datetime value is in the future the return value will automatically use an appropriate phrase.

Examples (when ‘now’ is 17 Feb 2007 16:30:00):

•17 Feb 2007 16:30:00 becomes now.
•17 Feb 2007 16:29:31 becomes 29 seconds ago.
•17 Feb 2007 16:29:00 becomes a minute ago.
•17 Feb 2007 16:25:35 becomes 4 minutes ago.
•17 Feb 2007 15:30:29 becomes 59 minutes ago.
•17 Feb 2007 15:30:01 becomes 59 minutes ago.
•17 Feb 2007 15:30:00 becomes an hour ago.
•17 Feb 2007 13:31:29 becomes 2 hours ago.
•16 Feb 2007 13:31:29 becomes 1 day, 2 hours ago.
•16 Feb 2007 13:30:01 becomes 1 day, 2 hours ago.
•16 Feb 2007 13:30:00 becomes 1 day, 3 hours ago.
•17 Feb 2007 16:30:30 becomes 30 seconds from now.
•17 Feb 2007 16:30:29 becomes 29 seconds from now.
•17 Feb 2007 16:31:00 becomes a minute from now.
•17 Feb 2007 16:34:35 becomes 4 minutes from now.
•17 Feb 2007 17:30:29 becomes an hour from now.
•17 Feb 2007 18:31:29 becomes 2 hours from now.
•18 Feb 2007 16:31:29 becomes 1 day from now.
•26 Feb 2007 18:31:29 becomes 1 week, 2 days from now.
ordinal
Converts an integer to its ordinal as a string.

Examples:

•1 becomes 1st.
•2 becomes 2nd.
•3 becomes 3rd.
You can pass in either an integer or a string representation of an integer.









class Template
Using the template system in Python is a two-step process:

•First, you compile the raw template code into a Template object.
•Then, you call the render() method of the Template object with a given context.


Compiling a string
The easiest way to create a Template object is by instantiating it directly. The class lives at django.template.Template. The constructor takes one argument – the raw template code:

from django.template import Template
t = Template("My name is {{ my_name }}.")
print(t)
<django.template.Template instance>



Rendering a context render(context)
The Context class lives at django.template.Context, and the constructor takes two (optional) arguments:

•A dictionary mapping variable names to variable values.
•The name of the current application. This application name is used to help resolve namespaced URLs. If you’re not using namespaced URLs, you can ignore this argument.
Call the Template object’s render() method with the context to “fill” the template:

from django.template import Context, Template
t = Template("My name is {{ my_name }}.")

c = Context({"my_name": "Adrian"})
t.render(c)
"My name is Adrian."

c = Context({"my_name": "Dolores"})
t.render(c)
"My name is Dolores."


Variables and lookups
Variable names must consist of any letter (A-Z), any digit (0-9), an underscore (but they must not start with an underscore) or a dot.

A dot in a variable name signifies a lookup. Specifically, when the template system encounters a dot in a variable name, it tries the following lookups, in this order:

•Dictionary lookup. Example: foo["bar"]
•Attribute lookup. Example: foo.bar
•List-index lookup. Example: foo[bar]
Note that “bar” in a template expression like {{ foo.bar }} will be interpreted as a literal string and not using the value of the variable “bar”, if one exists in the template context.

The template system uses the first lookup type that works. It’s short-circuit logic. Here are a few examples:

from django.template import Context, Template
t = Template("My name is {{ person.first_name }}.")
d = {"person": {"first_name": "Joe", "last_name": "Johnson"}}
t.render(Context(d))
"My name is Joe."

class PersonClass: pass
p = PersonClass()
p.first_name = "Ron"
p.last_name = "Nasty"
t.render(Context({"person": p}))
"My name is Ron."

t = Template("The first stooge in the list is {{ stooges.0 }}.")
c = Context({"stooges": ["Larry", "Curly", "Moe"]})
t.render(c)
"The first stooge in the list is Larry."

If any part of the variable is callable, the template system will try calling it. Example:

class PersonClass2:
...     def name(self):
...         return "Samantha"
t = Template("My name is {{ person.name }}.")
t.render(Context({"person": PersonClass2}))
"My name is Samantha."



•If the variable raises an exception when called, the exception will be propagated, unless the exception has an attribute silent_variable_failure whose value is True. If the exception does have a silent_variable_failure attribute whose value is True, the variable will render as the value of the TEMPLATE_STRING_IF_INVALID setting (an empty string, by default). Example:

t = Template("My name is {{ person.first_name }}.")
class PersonClass3:
...     def first_name(self):
...         raise AssertionError("foo")
p = PersonClass3()
t.render(Context({"person": p}))
Traceback (most recent call last):
...
AssertionError: foo

class SilentAssertionError(Exception):
...     silent_variable_failure = True
class PersonClass4:
...     def first_name(self):
...         raise SilentAssertionError
p = PersonClass4()
t.render(Context({"person": p}))
"My name is ."

Note that django.core.exceptions.ObjectDoesNotExist, which is the base class for all Django database API DoesNotExist exceptions, has silent_variable_failure = True. 
So if you’re using Django templates with Django model objects, any DoesNotExist exception will fail silently.

•A variable can only be called if it has no required arguments. Otherwise, the system will return the value of TEMPLATE_STRING_IF_INVALID.


How to prevent side effects when variable is called

A good example is the delete() method on each Django model object. The template system shouldn’t be allowed to do something like this:

{{ data.delete }}

To prevent this, set an alters_data attribute on the callable variable. The template system won’t call a variable if it has alters_data=True set, and will instead replace the variable with TEMPLATE_STRING_IF_INVALID, unconditionally. 
The dynamically-generated delete() and save() methods on Django model objects get alters_data=True automatically. Example:

def sensitive_function(self):
    self.database_record.delete()
sensitive_function.alters_data = True


How to  tell the template system to leave a variable uncalled no matter what. To do so, set a do_not_call_in_templates attribute on the callable with the value True. The template system then will act as if your variable is not callable (allowing you to access attributes of the callable, for example).



How invalid variables are handled
Generally, if a variable doesn’t exist, the template system inserts the value of the TEMPLATE_STRING_IF_INVALID setting, which is set to '' (the empty string) by default.

Filters that are applied to an invalid variable will only be applied if TEMPLATE_STRING_IF_INVALID is set to '' (the empty string). If TEMPLATE_STRING_IF_INVALID is set to any other value, variable filters will be ignored.

This behavior is slightly different for the if, for and regroup template tags. If an invalid variable is provided to one of these template tags, the variable will be interpreted as None. Filters are always applied to invalid variables within these template tags.

If TEMPLATE_STRING_IF_INVALID contains a '%s', the format marker will be replaced with the name of the invalid variable.



Builtin variables
Every context contains True, False and None. As you would expect, these variables resolve to the corresponding Python objects.



Limitations with string literals
Django’s template language has no way to escape the characters used for its own syntax. For example, the templatetag tag is required if you need to output character sequences like {% and %}.

A similar issue exists if you want to include these sequences in template filter or tag arguments. For example, when parsing a block tag, Django’s template parser looks for the first occurrence of %} after a {%. This prevents the use of "%}" as a string literal. For example, a TemplateSyntaxError will be raised for the following expressions:

{% include "template.html" tvar="Some string literal with %} in it." %}

{% with tvar="Some string literal with %} in it." %}{% endwith %}

The same issue can be triggered by using a reserved sequence in filter arguments:

{{ some.variable|default:"}}" }}
If you need to use strings with these sequences, store them in template variables or use a custom template tag or filter to workaround the limitation.





class Context

from django.template import Context
c = Context({"foo": "bar"})
c['foo']
'bar'
del c['foo']
c['foo']
Traceback (most recent call last):
...
KeyError: 'foo'
c['newvariable'] = 'hello'
c['newvariable']
'hello'


Context.get(key, otherwise=None)
Returns the value for key if key is in the context, else returns otherwise.

Context.pop()
Context.push()
exception ContextPopException
A Context object is a stack. That is, you can push() and pop() it. If you pop() too much, it’ll raise django.template.ContextPopException:

c = Context()
c['foo'] = 'first level'
c.push()
{}
c['foo'] = 'second level'
c['foo']
'second level'
c.pop()
{'foo': 'second level'}
c['foo']
'first level'
c['foo'] = 'overwritten'
c['foo']
'overwritten'
c.pop()
Traceback (most recent call last):
...
ContextPopException
You can also use push() as a context manager to ensure a matching pop() is called.

c = Context()
c['foo'] = 'first level'
with c.push():
...     c['foo'] = 'second level'
...     c['foo']
'second level'
c['foo']
'first level'
All arguments passed to push() will be passed to the dict constructor used to build the new context level.

c = Context()
c['foo'] = 'first level'
with c.push(foo='second level'):
...     c['foo']
'second level'
c['foo']
'first level'


Context.update(other_dict)
This works like push() but takes a dictionary as an argument and pushes that dictionary onto the stack instead of an empty one.

c = Context()
c['foo'] = 'first level'
c.update({'foo': 'updated'})
{'foo': 'updated'}
c['foo']
'updated'
c.pop()
{'foo': 'updated'}
c['foo']
'first level'
Using a Context as a stack comes in handy in some custom template tags, as you’ll see below.


Context.flatten()
Using flatten() method you can get whole Context stack as one dictionary including builtin variables.

c = Context()
c['foo'] = 'first level'
c.update({'bar': 'second level'})
{'bar': 'second level'}
c.flatten()
{'True': True, 'None': None, 'foo': 'first level', 'False': False, 'bar': 'second level'}
A flatten() method is also internally used to make Context objects comparable.

c1 = Context()
c1['foo'] = 'first level'
c1['bar'] = 'second level'
c2 = Context()
c2.update({'bar': 'second level', 'foo': 'first level'})
{'foo': 'first level', 'bar': 'second level'}
c1 == c2
True
Result from flatten() can be useful in unit tests to compare Context against dict:

class ContextTest(unittest.TestCase):
    def test_against_dictionary(self):
        c1 = Context()
        c1['update'] = 'value'
        self.assertEqual(c1.flatten(), {
            'True': True, 'None': None, 'False': False,
            'update': 'value'})



Subclassing Context: class RequestContext

The first difference is that it takes an HttpRequest as its first argument. For example:

c = RequestContext(request, {
    'foo': 'bar',
})
The second difference is that it automatically populates the context with a few variables, according to your TEMPLATE_CONTEXT_PROCESSORS setting.

The TEMPLATE_CONTEXT_PROCESSORS setting is a tuple of callables – called context processors – that take a request object as their argument and return a dictionary of items to be merged into the context. By default, TEMPLATE_CONTEXT_PROCESSORS is set to:

("django.contrib.auth.context_processors.auth",
"django.core.context_processors.debug",
"django.core.context_processors.i18n",
"django.core.context_processors.media",
"django.core.context_processors.static",
"django.core.context_processors.tz",
"django.contrib.messages.context_processors.messages")

In addition to these, RequestContext always uses django.core.context_processors.csrf. This is a security related context processor required by the admin and other contrib apps, and, in case of accidental misconfiguration, it is deliberately hardcoded in and cannot be turned off by the TEMPLATE_CONTEXT_PROCESSORS setting.

Each processor is applied in order. That means, if one processor adds a variable to the context and a second processor adds a variable with the same name, the second will override the first. The default processors are explained below.



When context processors are applied

Context processors are applied after the context itself is processed. This means that a context processor may overwrite variables you’ve supplied to your Context or RequestContext

Also, you can give RequestContext a list of additional processors, using the optional, third positional argument, processors. In this example, the RequestContext instance gets a ip_address variable:

from django.http import HttpResponse
from django.template import RequestContext

def ip_address_processor(request):
    return {'ip_address': request.META['REMOTE_ADDR']}

def some_view(request):
    # ...
    c = RequestContext(request, {
        'foo': 'bar',
    }, [ip_address_processor])
    return HttpResponse(t.render(c))



If you’re using Django’s render_to_response() shortcut to populate a template with the contents of a dictionary, your template will be passed a Context instance by default (not a RequestContext). To use a RequestContext in your template rendering, use the render() shortcut which is the same as a call to render_to_response() with a context_instance argument that forces the use of a RequestContext.




Here’s what each of the default processors does:

django.contrib.auth.context_processors.auth
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain these variables:

•user – An auth.User instance representing the currently logged-in user (or an AnonymousUser instance, if the client isn’t logged in).
•perms – An instance of django.contrib.auth.context_processors.PermWrapper, representing the permissions that the currently logged-in user has.


django.core.context_processors.debug
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain these two variables – but only if your DEBUG setting is set to True and the request’s IP address (request.META['REMOTE_ADDR']) is in the INTERNAL_IPS setting:

•debug – True. You can use this in templates to test whether you’re in DEBUG mode.
•sql_queries – A list of {'sql': ..., 'time': ...} dictionaries, representing every SQL query that has happened so far during the request and how long it took. The list is in order by query.


django.core.context_processors.i18n
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain these two variables:

•LANGUAGES – The value of the LANGUAGES setting.
•LANGUAGE_CODE – request.LANGUAGE_CODE, if it exists. Otherwise, the value of the LANGUAGE_CODE setting.


django.core.context_processors.media
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain a variable MEDIA_URL, providing the value of the MEDIA_URL setting.

django.core.context_processors.static
static()[source]
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain a variable STATIC_URL, providing the value of the STATIC_URL setting.

django.core.context_processors.csrf
This processor adds a token that is needed by the csrf_token template tag for protection against Cross Site Request Forgeries.

django.core.context_processors.request
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain a variable request, which is the current HttpRequest. Note that this processor is not enabled by default; you’ll have to activate it.

django.contrib.messages.context_processors.messages
If TEMPLATE_CONTEXT_PROCESSORS contains this processor, every RequestContext will contain these two variables:

•messages – A list of messages (as strings) that have been set via the messages framework.
•DEFAULT_MESSAGE_LEVELS – A mapping of the message level names to their numeric value.

The DEFAULT_MESSAGE_LEVELS variable was added.



Writing your own context processors
It’s just a Python function that takes one argument, an HttpRequest object, and returns a dictionary that gets added to the template context. Each context processor must return a dictionary.

then add to  TEMPLATE_CONTEXT_PROCESSORS setting.




Loading templates
Save templates in a directory specified as a template directory.

Django searches for template directories in a number of places, depending on your template-loader settings , but the most basic way of specifying template directories is by using the TEMPLATE_DIRS setting.


The TEMPLATE_DIRS setting
Tell Django what your template directories are by using the TEMPLATE_DIRS setting in your settings file. This should be set to a list or tuple of strings that contain full paths to your template directory(ies). Example:

TEMPLATE_DIRS = (
    "/home/html/templates/lawrence.com",
    "/home/html/templates/default",
)
Your templates can go anywhere you want, as long as the directories and templates are readable by the Web server. They can have any extension you want, such as .html or .txt, or they can have no extension at all.


Template Method:

For example, if you call get_template('story_detail.html') and have the above TEMPLATE_DIRS setting, here are the files Django will look for, in order:

•/home/html/templates/lawrence.com/story_detail.html
•/home/html/templates/default/story_detail.html

If you call select_template(['story_253_detail.html', 'story_detail.html']), here’s what Django will look for:

•/home/html/templates/lawrence.com/story_253_detail.html
•/home/html/templates/default/story_253_detail.html
•/home/html/templates/lawrence.com/story_detail.html
•/home/html/templates/default/story_detail.html
When Django finds a template that exists, it stops looking.


You can use select_template() for super-flexible “templatability.” 
For example, if you’ve written a news story and want some stories to have custom templates, use something like select_template(['story_%s_detail.html' % story.id, 'story_detail.html']). That’ll allow you to use a custom template for an individual story, with a fallback template for stories that don’t have custom templates.




Using subdirectories
It’s possible – and preferable – to organize templates in subdirectories of the template directory. The convention is to make a subdirectory for each Django app, with subdirectories within those subdirectories as needed.

To load a template that’s within a subdirectory, just use a slash, like so:

get_template('news/story_detail.html')
Using the same TEMPLATE_DIRS setting from above, this example get_template() call will attempt to load the following templates:

•/home/html/templates/lawrence.com/news/story_detail.html
•/home/html/templates/default/news/story_detail.html



Loader types
By default, Django uses a filesystem-based template loader

Some of these other loaders are disabled by default, but you can activate them by editing your TEMPLATE_LOADERS setting. TEMPLATE_LOADERS should be a tuple of strings, where each string represents a template loader class. Here are the template loaders that come with Django:

django.template.loaders.filesystem.Loader
Loads templates from the filesystem, according to TEMPLATE_DIRS. This loader is enabled by default.

django.template.loaders.app_directories.Loader
This loader is enabled by default..Loads templates from Django apps on the filesystem. For each app in INSTALLED_APPS, the loader looks for a templates subdirectory. If the directory exists, Django looks for templates in there.

This means you can store templates with your individual apps. This also makes it easy to distribute Django apps with default templates.

For example, for this setting:

INSTALLED_APPS = ('myproject.polls', 'myproject.music')
...then get_template('foo.html') will look for foo.html in these directories, in this order:

•/path/to/myproject/polls/templates/
•/path/to/myproject/music/templates/
... and will use the one it finds first.

The order of INSTALLED_APPS is significant! 
For example, if you want to customize the Django admin, you might choose to override the standard admin/base_site.html template, from django.contrib.admin, with your own admin/base_site.html in myproject.polls. You must then make sure that your myproject.polls comes before django.contrib.admin in INSTALLED_APPS, otherwise django.contrib.admin’s will be loaded first and yours will be ignored.

Note that the loader performs an optimization when it is first imported: it caches a list of which INSTALLED_APPS packages have a templates subdirectory.





django.template.loaders.eggs.Loader
Just like app_directories above, but it loads templates from Python eggs rather than from the filesystem.

This loader is disabled by default.



django.template.loaders.cached.Loader
By default, the templating system will read and compile your templates every time they need to be rendered. While the Django templating system is quite fast, the overhead from reading and compiling templates can add up.

The cached template loader is a class-based loader that you configure with a list of other loaders that it should wrap. The wrapped loaders are used to locate unknown templates when they are first encountered. The cached loader then stores the compiled Template in memory. The cached Template instance is returned for subsequent requests to load the same template.

For example, to enable template caching with the filesystem and app_directories template loaders you might use the following settings:

TEMPLATE_LOADERS = (
    ('django.template.loaders.cached.Loader', (
        'django.template.loaders.filesystem.Loader',
        'django.template.loaders.app_directories.Loader',
    )),
)
Note

All of the built-in Django template tags are safe to use with the cached loader, but if you’re using custom template tags that come from third party packages, or that you wrote yourself, you should ensure that the Node implementation for each tag is thread-safe. For more information, see template tag thread safety considerations.

This loader is disabled by default.

Django uses the template loaders in order according to the TEMPLATE_LOADERS setting. It uses each loader until a loader finds a match.



Template origin
When TEMPLATE_DEBUG is True template objects will have an origin attribute depending on the source they are loaded from.

class loader.LoaderOrigin
Templates created from a template loader will use the django.template.loader.LoaderOrigin class.

name
The path to the template as returned by the template loader. For loaders that read from the file system, this is the full path to the template.

loadname
The relative path to the template as passed into the template loader.


class StringOrigin
Templates created from a Template class will use the django.template.StringOrigin class.

source
The string used to create the template.



The render_to_string shortcut
loader.render_to_string(template_name, dictionary=None, context_instance=None)
To cut down on the repetitive nature of loading and rendering templates, Django provides a shortcut function which largely automates the process: render_to_string() in django.template.loader, which loads a template, renders it and returns the resulting string:

from django.template.loader import render_to_string
rendered = render_to_string('my_template.html', {'foo': 'bar'})
The render_to_string shortcut takes one required argument – template_name, which should be the name of the template to load and render (or a list of template names, in which case Django will use the first template in the list that exists) – and two optional arguments:


dictionary
A dictionary to be used as variables and values for the template’s context. This can also be passed as the second positional argument.

context_instance
An instance of Context or a subclass (e.g., an instance of RequestContext) to use as the template’s context. This can also be passed as the third positional argument.
See also the render_to_response() shortcut, which calls render_to_string and feeds the result into an HttpResponse suitable for returning directly from a view.

















Custom template tags and filters


Code layout
Custom template tags and filters must live inside a Django app. If they relate to an existing app it makes sense to bundle them there; otherwise, you should create a new app to hold them.

The app should contain a templatetags directory, at the same level as models.py, views.py, etc. If this doesn’t already exist, create it - don’t forget the __init__.py file to ensure the directory is treated as a Python package. After adding this module, you will need to restart your server before you can use the tags or filters in templates.

Your custom tags and filters will live in a module inside the templatetags directory. The name of the module file is the name you’ll use to load the tags later, so be careful to pick a name that won’t clash with custom tags and filters in another app.

For example, if your custom tags/filters are in a file called poll_extras.py, your app layout might look like this:

polls/
    __init__.py
    models.py
    templatetags/
        __init__.py
        poll_extras.py
    views.py

And in your template you would use the following:

{% load poll_extras %}
The app that contains the custom tags must be in INSTALLED_APPS in order for the {% load %} tag to work. 


To be a valid tag library, the module must contain a module-level variable named register that is a template.Library instance, in which all the tags and filters are registered. So, near the top of your module, put the following:

from django import template

register = template.Library()




Writing custom template filters
Custom filters are just Python functions that take one or two arguments:

•The value of the variable (input) – not necessarily a string.
•The value of the argument – this can have a default value, or be left out altogether.

For example, in the filter {{ var|foo:"bar" }}, the filter foo would be passed the variable var and the argument "bar".

Usually any exception raised from a template filter will be exposed as a server error. Thus, filter functions should avoid raising exceptions if there is a reasonable fallback value to return. In case of input that represents a clear bug in a template, raising an exception may still be better than silent failure which hides the bug.

Here’s an example filter definition:

def cut(value, arg):
    """Removes all values of arg from the given string"""
    return value.replace(arg, '')

And here’s an example of how that filter would be used:

{{ somevariable|cut:"0" }}

Most filters don’t take arguments. In this case, just leave the argument out of your function. Example:

def lower(value): # Only one argument.
    """Converts a string into all lowercase"""
    return value.lower()


Registering custom filters  django.template.Library.filter()
Once you’ve written your filter definition, you need to register it with your Library instance, to make it available to Django’s template language:

register.filter('cut', cut)
register.filter('lower', lower)

The Library.filter() method takes two arguments:

1.The name of the filter – a string.
2.The compilation function – a Python function (not the name of the function as a string).
You can use register.filter() as a decorator instead:

@register.filter(name='cut')
def cut(value, arg):
    return value.replace(arg, '')

@register.filter
def lower(value):
    return value.lower()
If you leave off the name argument, as in the second example above, Django will use the function’s name as the filter name.

Finally, register.filter() also accepts three keyword arguments, is_safe, needs_autoescape, and expects_localtime. These arguments are described in filters and auto-escaping and filters and time zones below.



Template filters that expect strings  django.template.defaultfilters.stringfilter()
If you’re writing a template filter that only expects a string as the first argument, you should use the decorator stringfilter. This will convert an object to its string value before being passed to your function:

from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
@stringfilter
def lower(value):
    return value.lower()
This way, you’ll be able to pass, say, an integer to this filter, and it won’t cause an AttributeError (because integers don’t have lower() methods).



Filters and auto-escaping
Note that three types of strings can be passed around inside the template code:

•Raw strings are the native Python str or unicode types. On output, they’re escaped if auto-escaping is in effect and presented unchanged, otherwise.

•Safe strings are strings that have been marked safe from further escaping at output time. Any necessary escaping has already been done. They’re commonly used for output that contains raw HTML that is intended to be interpreted as-is on the client side.

Internally, these strings are of type SafeBytes or SafeText. They share a common base class of SafeData, so you can test for them using code like:

if isinstance(value, SafeData):
    # Do something with the "safe" string.
    ...
•Strings marked as “needing escaping” are always escaped on output, regardless of whether they are in an autoescape block or not. These strings are only escaped once, however, even if auto-escaping applies.

Internally, these strings are of type EscapeBytes or EscapeText. Generally you don’t have to worry about these; they exist for the implementation of the escape filter.

Template filter code falls into one of two situations:

1.Your filter does not introduce any HTML-unsafe characters (<, >, ', " or &) into the result that were not already present. In this case, you can let Django take care of all the auto-escaping handling for you. All you need to do is set the is_safe flag to True when you register your filter function, like so:

@register.filter(is_safe=True)
def myfilter(value):
    return value
This flag tells Django that if a “safe” string is passed into your filter, the result will still be “safe” and if a non-safe string is passed in, Django will automatically escape it, if necessary.

You can think of this as meaning “this filter is safe – it doesn’t introduce any possibility of unsafe HTML.”

The reason is_safe is necessary is because there are plenty of normal string operations that will turn a SafeData object back into a normal str or unicode object and, rather than try to catch them all, which would be very difficult, Django repairs the damage after the filter has completed.

For example, suppose you have a filter that adds the string xx to the end of any input. Since this introduces no dangerous HTML characters to the result (aside from any that were already present), you should mark your filter with is_safe:

@register.filter(is_safe=True)
def add_xx(value):
    return '%sxx' % value
When this filter is used in a template where auto-escaping is enabled, Django will escape the output whenever the input is not already marked as “safe”.

By default, is_safe is False, and you can omit it from any filters where it isn’t required.

Be careful when deciding if your filter really does leave safe strings as safe. If you’re removing characters, you might inadvertently leave unbalanced HTML tags or entities in the result. For example, removing a > from the input might turn <a> into <a, which would need to be escaped on output to avoid causing problems. Similarly, removing a semicolon (;) can turn &amp; into &amp, which is no longer a valid entity and thus needs further escaping. Most cases won’t be nearly this tricky, but keep an eye out for any problems like that when reviewing your code.

Marking a filter is_safe will coerce the filter’s return value to a string. If your filter should return a boolean or other non-string value, marking it is_safe will probably have unintended consequences (such as converting a boolean False to the string ‘False’).

2.Alternatively, your filter code can manually take care of any necessary escaping. This is necessary when you’re introducing new HTML markup into the result. You want to mark the output as safe from further escaping so that your HTML markup isn’t escaped further, so you’ll need to handle the input yourself.

To mark the output as a safe string, use django.utils.safestring.mark_safe().

Be careful, though. You need to do more than just mark the output as safe. You need to ensure it really is safe, and what you do depends on whether auto-escaping is in effect. The idea is to write filters that can operate in templates where auto-escaping is either on or off in order to make things easier for your template authors.

In order for your filter to know the current auto-escaping state, set the needs_autoescape flag to True when you register your filter function. (If you don’t specify this flag, it defaults to False). This flag tells Django that your filter function wants to be passed an extra keyword argument, called autoescape, that is True if auto-escaping is in effect and False otherwise.

For example, let’s write a filter that emphasizes the first character of a string:

from django import template
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(needs_autoescape=True)
def initial_letter_filter(text, autoescape=None):
    first, other = text[0], text[1:]
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
    result = '<strong>%s</strong>%s' % (esc(first), esc(other))
    return mark_safe(result)
The needs_autoescape flag and the autoescape keyword argument mean that our function will know whether automatic escaping is in effect when the filter is called. We use autoescape to decide whether the input data needs to be passed through django.utils.html.conditional_escape or not. (In the latter case, we just use the identity function as the “escape” function.) The conditional_escape() function is like escape() except it only escapes input that is not a SafeData instance. If a SafeData instance is passed to conditional_escape(), the data is returned unchanged.

Finally, in the above example, we remember to mark the result as safe so that our HTML is inserted directly into the template without further escaping.

There’s no need to worry about the is_safe flag in this case (although including it wouldn’t hurt anything). Whenever you manually handle the auto-escaping issues and return a safe string, the is_safe flag won’t change anything either way.

Warning

Avoiding XSS vulnerabilities when reusing built-in filters

Be careful when reusing Django’s built-in filters. You’ll need to pass autoescape=True to the filter in order to get the proper autoescaping behavior and avoid a cross-site script vulnerability.

For example, if you wanted to write a custom filter called urlize_and_linebreaks that combined the urlize and linebreaksbr filters, the filter would look like:

from django.template.defaultfilters import linebreaksbr, urlize

@register.filter
def urlize_and_linebreaks(text):
    return linebreaksbr(urlize(text, autoescape=True), autoescape=True)
Then:

{{ comment|urlize_and_linebreaks }}
would be equivalent to:

{{ comment|urlize|linebreaksbr }}



Filters and time zones
If you write a custom filter that operates on datetime objects, you’ll usually register it with the expects_localtime flag set to True:

@register.filter(expects_localtime=True)
def businesshours(value):
    try:
        return 9 <= value.hour < 17
    except AttributeError:
        return ''
When this flag is set, if the first argument to your filter is a time zone aware datetime, Django will convert it to the current time zone before passing it to your filter when appropriate, according to rules for time zones conversions in templates.




Writing custom template tags

For each template tag the template parser encounters, it calls a Python function with the tag contents and the parser object itself. This function is responsible for returning a Node instance based on the contents of the tag.

For example, let’s write a template tag, {% current_time %}, that displays the current date/time, formatted according to a parameter given in the tag, in strftime() syntax. It’s a good idea to decide the tag syntax before anything else. In our case, let’s say the tag should be used like this:

<p>The time is {% current_time "%Y-%m-%d %I:%M %p" %}.</p>

The parser for this function should grab the parameter and create a Node object:

from django import template
def do_current_time(parser, token):
    try:
        # split_contents() knows not to split quoted strings.
        tag_name, format_string = token.split_contents()
    except ValueError:
        raise template.TemplateSyntaxError(
            "%r tag requires a single argument" % token.contents.split()[0]
        )
    if not (format_string[0] == format_string[-1] and format_string[0] in ('"', "'")):
        raise template.TemplateSyntaxError(
            "%r tag's argument should be in quotes" % tag_name
        )
    return CurrentTimeNode(format_string[1:-1])
Notes:

•parser is the template parser object. We don’t need it in this example.
•token.contents is a string of the raw contents of the tag. In our example, it’s 'current_time "%Y-%m-%d %I:%M %p"'.
•The token.split_contents() method separates the arguments on spaces while keeping quoted strings together. The more straightforward token.contents.split() wouldn’t be as robust, as it would naively split on all spaces, including those within quoted strings. It’s a good idea to always use token.split_contents().
•This function is responsible for raising django.template.TemplateSyntaxError, with helpful messages, for any syntax error.
•The TemplateSyntaxError exceptions use the tag_name variable. Don’t hard-code the tag’s name in your error messages, because that couples the tag’s name to your function. token.contents.split()[0] will ‘’always’’ be the name of your tag – even when the tag has no arguments.
•The function returns a CurrentTimeNode with everything the node needs to know about this tag. In this case, it just passes the argument – "%Y-%m-%d %I:%M %p". The leading and trailing quotes from the template tag are removed in format_string[1:-1].
•The parsing is very low-level. The Django developers have experimented with writing small frameworks on top of this parsing system, using techniques such as EBNF grammars, but those experiments made the template engine too slow. It’s low-level because that’s fastest.


Writing the renderer
The second step in writing custom tags is to define a Node subclass that has a render() method.

Continuing the above example, we need to define CurrentTimeNode:

import datetime
from django import template

class CurrentTimeNode(template.Node):
    def __init__(self, format_string):
        self.format_string = format_string
    def render(self, context):
        return datetime.datetime.now().strftime(self.format_string)
Notes:

•__init__() gets the format_string from do_current_time(). Always pass any options/parameters/arguments to a Node via its __init__().
•The render() method is where the work actually happens.
•render() should generally fail silently, particularly in a production environment where DEBUG and TEMPLATE_DEBUG are False. In some cases however, particularly if TEMPLATE_DEBUG is True, this method may raise an exception to make debugging easier. For example, several core tags raise django.template.TemplateSyntaxError if they receive the wrong number or type of arguments.
Ultimately, this decoupling of compilation and rendering results in an efficient template system, because a template can render multiple contexts without having to be parsed multiple times.



Auto-escaping considerations
The output from template tags is not automatically run through the auto-escaping filters. However, there are still a couple of things you should keep in mind when writing a template tag.

If the render() function of your template stores the result in a context variable (rather than returning the result in a string), it should take care to call mark_safe() if appropriate. When the variable is ultimately rendered, it will be affected by the auto-escape setting in effect at the time, so content that should be safe from further escaping needs to be marked as such.

Also, if your template tag creates a new context for performing some sub-rendering, set the auto-escape attribute to the current context’s value. The __init__ method for the Context class takes a parameter called autoescape that you can use for this purpose. For example:

from django.template import Context

def render(self, context):
    # ...
    new_context = Context({'var': obj}, autoescape=context.autoescape)
    # ... Do something with new_context ...

This is not a very common situation, but it’s useful if you’re rendering a template yourself. For example:

def render(self, context):
    t = template.loader.get_template('small_fragment.html')
    return t.render(Context({'var': obj}, autoescape=context.autoescape))
If we had neglected to pass in the current context.autoescape value to our new Context in this example, the results would have always been automatically escaped, which may not be the desired behavior if the template tag is used inside a {% autoescape off %} block.




Thread-safety considerations
Once a node is parsed, its render method may be called any number of times. Since Django is sometimes run in multi-threaded environments, a single node may be simultaneously rendering with different contexts in response to two separate requests. Therefore, it’s important to make sure your template tags are thread safe.

To make sure your template tags are thread safe, you should never store state information on the node itself. For example, Django provides a builtin cycle template tag that cycles among a list of given strings each time it’s rendered:

{% for o in some_list %}
    <tr class="{% cycle 'row1' 'row2' %}">
        ...
    </tr>
{% endfor %}
A naive implementation of CycleNode might look something like this:

import itertools
from django import template

class CycleNode(template.Node):
    def __init__(self, cyclevars):
        self.cycle_iter = itertools.cycle(cyclevars)
    def render(self, context):
        return next(self.cycle_iter)
But, suppose we have two templates rendering the template snippet from above at the same time:

1.Thread 1 performs its first loop iteration, CycleNode.render() returns ‘row1’
2.Thread 2 performs its first loop iteration, CycleNode.render() returns ‘row2’
3.Thread 1 performs its second loop iteration, CycleNode.render() returns ‘row1’
4.Thread 2 performs its second loop iteration, CycleNode.render() returns ‘row2’
The CycleNode is iterating, but it’s iterating globally. As far as Thread 1 and Thread 2 are concerned, it’s always returning the same value. This is obviously not what we want!

To address this problem, Django provides a render_context that’s associated with the context of the template that is currently being rendered. The render_context behaves like a Python dictionary, and should be used to store Node state between invocations of the render method.

Let’s refactor our CycleNode implementation to use the render_context:

class CycleNode(template.Node):
    def __init__(self, cyclevars):
        self.cyclevars = cyclevars
    def render(self, context):
        if self not in context.render_context:
            context.render_context[self] = itertools.cycle(self.cyclevars)
        cycle_iter = context.render_context[self]
        return next(cycle_iter)
Note that it’s perfectly safe to store global information that will not change throughout the life of the Node as an attribute. In the case of CycleNode, the cyclevars argument doesn’t change after the Node is instantiated, so we don’t need to put it in the render_context. But state information that is specific to the template that is currently being rendered, like the current iteration of the CycleNode, should be stored in the render_context.


Notice how we used self to scope the CycleNode specific information within the render_context. There may be multiple CycleNodes in a given template, so we need to be careful not to clobber another node’s state information. The easiest way to do this is to always use self as the key into render_context. If you’re keeping track of several state variables, make render_context[self] a dictionary.



Registering the tag
Finally, register the tag with your module’s Library instance, as explained in “Writing custom template filters” above. Example:

register.tag('current_time', do_current_time)
The tag() method takes two arguments:

1.The name of the template tag – a string. If this is left out, the name of the compilation function will be used.
2.The compilation function – a Python function (not the name of the function as a string).
As with filter registration, it is also possible to use this as a decorator:

@register.tag(name="current_time")
def do_current_time(parser, token):
    ...

@register.tag
def shout(parser, token):
    ...
If you leave off the name argument, as in the second example above, Django will use the function’s name as the tag name.



Passing template variables to the tag
Although you can pass any number of arguments to a template tag using token.split_contents(), the arguments are all unpacked as string literals. A little more work is required in order to pass dynamic content (a template variable) to a template tag as an argument.

While the previous examples have formatted the current time into a string and returned the string, suppose you wanted to pass in a DateTimeField from an object and have the template tag format that date-time:

<p>This post was last updated at {% format_time blog_entry.date_updated "%Y-%m-%d %I:%M %p" %}.</p>

Initially, token.split_contents() will return three values:

1.The tag name format_time.
2.The string "blog_entry.date_updated" (without the surrounding quotes).
3.The formatting string "%Y-%m-%d %I:%M %p". The return value from split_contents() will include the leading and trailing quotes for string literals like this.

Now your tag should begin to look like this:

from django import template

def do_format_time(parser, token):
    try:
        # split_contents() knows not to split quoted strings.
        tag_name, date_to_be_formatted, format_string = token.split_contents()
    except ValueError:
        raise template.TemplateSyntaxError(
            "%r tag requires exactly two arguments" % token.contents.split()[0]
        )
    if not (format_string[0] == format_string[-1] and format_string[0] in ('"', "'")):
        raise template.TemplateSyntaxError(
            "%r tag's argument should be in quotes" % tag_name
        )
    return FormatTimeNode(date_to_be_formatted, format_string[1:-1])
You also have to change the renderer to retrieve the actual contents of the date_updated property of the blog_entry object. This can be accomplished by using the Variable() class in django.template.

To use the Variable class, simply instantiate it with the name of the variable to be resolved, and then call variable.resolve(context). So, for example:

class FormatTimeNode(template.Node):
    def __init__(self, date_to_be_formatted, format_string):
        self.date_to_be_formatted = template.Variable(date_to_be_formatted)
        self.format_string = format_string

    def render(self, context):
        try:
            actual_date = self.date_to_be_formatted.resolve(context)
            return actual_date.strftime(self.format_string)
        except template.VariableDoesNotExist:
            return ''
Variable resolution will throw a VariableDoesNotExist exception if it cannot resolve the string passed to it in the current context of the page.




Simple tags
django.template.Library.simple_tag()
Many template tags take a number of arguments – strings or template variables – and return a string after doing some processing based solely on the input arguments and some external information. For example, the current_time tag we wrote above is of this variety: we give it a format string, it returns the time as a string.

To ease the creation of these types of tags, Django provides a helper function, simple_tag. This function, which is a method of django.template.Library, takes a function that accepts any number of arguments, wraps it in a render function and the other necessary bits mentioned above and registers it with the template system.

Our earlier current_time function could thus be written like this:

import datetime
from django import template

register = template.Library()

def current_time(format_string):
    return datetime.datetime.now().strftime(format_string)

register.simple_tag(current_time)
The decorator syntax also works:

@register.simple_tag
def current_time(format_string):
    ...
A few things to note about the simple_tag helper function:

•Checking for the required number of arguments, etc., has already been done by the time our function is called, so we don’t need to do that.
•The quotes around the argument (if any) have already been stripped away, so we just receive a plain string.
•If the argument was a template variable, our function is passed the current value of the variable, not the variable itself.
If your template tag needs to access the current context, you can use the takes_context argument when registering your tag:

# The first argument *must* be called "context" here.
def current_time(context, format_string):
    timezone = context['timezone']
    return your_get_current_time_method(timezone, format_string)

register.simple_tag(takes_context=True)(current_time)
Or, using decorator syntax:

@register.simple_tag(takes_context=True)
def current_time(context, format_string):
    timezone = context['timezone']
    return your_get_current_time_method(timezone, format_string)
For more information on how the takes_context option works, see the section on inclusion tags.

If you need to rename your tag, you can provide a custom name for it:

register.simple_tag(lambda x: x - 1, name='minusone')

@register.simple_tag(name='minustwo')
def some_function(value):
    return value - 2
simple_tag functions may accept any number of positional or keyword arguments. For example:

@register.simple_tag
def my_tag(a, b, *args, **kwargs):
    warning = kwargs['warning']
    profile = kwargs['profile']
    ...
    return ...
Then in the template any number of arguments, separated by spaces, may be passed to the template tag. Like in Python, the values for keyword arguments are set using the equal sign (“=”) and must be provided after the positional arguments. For example:

{% my_tag 123 "abcd" book.title warning=message|lower profile=user.profile %}



Inclusion tags
Another common type of template tag is the type that displays some data by rendering another template. For example, Django’s admin interface uses custom template tags to display the buttons along the bottom of the “add/change” form pages. Those buttons always look the same, but the link targets change depending on the object being edited – so they’re a perfect case for using a small template that is filled with details from the current object. (In the admin’s case, this is the submit_row tag.)

These sorts of tags are called “inclusion tags”.

Writing inclusion tags is probably best demonstrated by example. Let’s write a tag that outputs a list of choices for a given Poll object, such as was created in the tutorials. We’ll use the tag like this:

{% show_results poll %}
...and the output will be something like this:

<ul>
  <li>First choice</li>
  <li>Second choice</li>
  <li>Third choice</li>
</ul>
First, define the function that takes the argument and produces a dictionary of data for the result. The important point here is we only need to return a dictionary, not anything more complex. This will be used as a template context for the template fragment. Example:

def show_results(poll):
    choices = poll.choice_set.all()
    return {'choices': choices}
Next, create the template used to render the tag’s output. This template is a fixed feature of the tag: the tag writer specifies it, not the template designer. Following our example, the template is very simple:

<ul>
{% for choice in choices %}
    <li> {{ choice }} </li>
{% endfor %}
</ul>
Now, create and register the inclusion tag by calling the inclusion_tag() method on a Library object. Following our example, if the above template is in a file called results.html in a directory that’s searched by the template loader, we’d register the tag like this:

# Here, register is a django.template.Library instance, as before
register.inclusion_tag('results.html')(show_results)
Alternatively it is possible to register the inclusion tag using a django.template.Template instance:

from django.template.loader import get_template
t = get_template('results.html')
register.inclusion_tag(t)(show_results)
As always, decorator syntax works as well, so we could have written:

@register.inclusion_tag('results.html')
def show_results(poll):
    ...
...when first creating the function.

Sometimes, your inclusion tags might require a large number of arguments, making it a pain for template authors to pass in all the arguments and remember their order. To solve this, Django provides a takes_context option for inclusion tags. If you specify takes_context in creating a template tag, the tag will have no required arguments, and the underlying Python function will have one argument – the template context as of when the tag was called.

For example, say you’re writing an inclusion tag that will always be used in a context that contains home_link and home_title variables that point back to the main page. Here’s what the Python function would look like:

# The first argument *must* be called "context" here.
def jump_link(context):
    return {
        'link': context['home_link'],
        'title': context['home_title'],
    }
# Register the custom tag as an inclusion tag with takes_context=True.
register.inclusion_tag('link.html', takes_context=True)(jump_link)
(Note that the first parameter to the function must be called context.)

In that register.inclusion_tag() line, we specified takes_context=True and the name of the template. Here’s what the template link.html might look like:

Jump directly to <a href="{{ link }}">{{ title }}</a>.
Then, any time you want to use that custom tag, load its library and call it without any arguments, like so:

{% jump_link %}
Note that when you’re using takes_context=True, there’s no need to pass arguments to the template tag. It automatically gets access to the context.

The takes_context parameter defaults to False. When it’s set to True, the tag is passed the context object, as in this example. That’s the only difference between this case and the previous inclusion_tag example.

inclusion_tag functions may accept any number of positional or keyword arguments. For example:

@register.inclusion_tag('my_template.html')
def my_tag(a, b, *args, **kwargs):
    warning = kwargs['warning']
    profile = kwargs['profile']
    ...
    return ...
Then in the template any number of arguments, separated by spaces, may be passed to the template tag. Like in Python, the values for keyword arguments are set using the equal sign (“=”) and must be provided after the positional arguments. For example:

{% my_tag 123 "abcd" book.title warning=message|lower profile=user.profile %}




Setting a variable in the context
The above examples simply output a value. Generally, it’s more flexible if your template tags set template variables instead of outputting values. That way, template authors can reuse the values that your template tags create.

To set a variable in the context, just use dictionary assignment on the context object in the render() method. Here’s an updated version of CurrentTimeNode that sets a template variable current_time instead of outputting it:

import datetime
from django import template

class CurrentTimeNode2(template.Node):
    def __init__(self, format_string):
        self.format_string = format_string
    def render(self, context):
        context['current_time'] = datetime.datetime.now().strftime(self.format_string)
        return ''
Note that render() returns the empty string. render() should always return string output. If all the template tag does is set a variable, render() should return the empty string.

Here’s how you’d use this new version of the tag:

{% current_time "%Y-%M-%d %I:%M %p" %}<p>The time is {{ current_time }}.</p>



Variable scope in context

Any variable set in the context will only be available in the same block of the template in which it was assigned. This behavior is intentional; it provides a scope for variables so that they don’t conflict with context in other blocks.

But, there’s a problem with CurrentTimeNode2: The variable name current_time is hard-coded. This means you’ll need to make sure your template doesn’t use {{ current_time }} anywhere else, because the {% current_time %} will blindly overwrite that variable’s value. A cleaner solution is to make the template tag specify the name of the output variable, like so:

{% current_time "%Y-%M-%d %I:%M %p" as my_current_time %}
<p>The current time is {{ my_current_time }}.</p>
To do that, you’ll need to refactor both the compilation function and Node class, like so:

class CurrentTimeNode3(template.Node):
    def __init__(self, format_string, var_name):
        self.format_string = format_string
        self.var_name = var_name
    def render(self, context):
        context[self.var_name] = datetime.datetime.now().strftime(self.format_string)
        return ''

import re
def do_current_time(parser, token):
    # This version uses a regular expression to parse tag contents.
    try:
        # Splitting by None == splitting by spaces.
        tag_name, arg = token.contents.split(None, 1)
    except ValueError:
        raise template.TemplateSyntaxError(
            "%r tag requires arguments" % token.contents.split()[0]
        )
    m = re.search(r'(.*?) as (\w+)', arg)
    if not m:
        raise template.TemplateSyntaxError("%r tag had invalid arguments" % tag_name)
    format_string, var_name = m.groups()
    if not (format_string[0] == format_string[-1] and format_string[0] in ('"', "'")):
        raise template.TemplateSyntaxError(
            "%r tag's argument should be in quotes" % tag_name
        )
    return CurrentTimeNode3(format_string[1:-1], var_name)
The difference here is that do_current_time() grabs the format string and the variable name, passing both to CurrentTimeNode3.

Finally, if you only need to have a simple syntax for your custom context-updating template tag, you might want to consider using an assignment tag.




Assignment tags
To ease the creation of tags setting a variable in the context, Django provides a helper function, assignment_tag. This function works the same way as simple_tag, except that it stores the tag’s result in a specified context variable instead of directly outputting it.

Our earlier current_time function could thus be written like this:

def get_current_time(format_string):
    return datetime.datetime.now().strftime(format_string)

register.assignment_tag(get_current_time)
The decorator syntax also works:

@register.assignment_tag
def get_current_time(format_string):
    ...
You may then store the result in a template variable using the as argument followed by the variable name, and output it yourself where you see fit:

{% get_current_time "%Y-%m-%d %I:%M %p" as the_time %}
<p>The time is {{ the_time }}.</p>
If your template tag needs to access the current context, you can use the takes_context argument when registering your tag:

# The first argument *must* be called "context" here.
def get_current_time(context, format_string):
    timezone = context['timezone']
    return your_get_current_time_method(timezone, format_string)

register.assignment_tag(takes_context=True)(get_current_time)
Or, using decorator syntax:

@register.assignment_tag(takes_context=True)
def get_current_time(context, format_string):
    timezone = context['timezone']
    return your_get_current_time_method(timezone, format_string)
For more information on how the takes_context option works, see the section on inclusion tags.

assignment_tag functions may accept any number of positional or keyword arguments. For example:

@register.assignment_tag
def my_tag(a, b, *args, **kwargs):
    warning = kwargs['warning']
    profile = kwargs['profile']
    ...
    return ...
Then in the template any number of arguments, separated by spaces, may be passed to the template tag. Like in Python, the values for keyword arguments are set using the equal sign (“=”) and must be provided after the positional arguments. For example:

{% my_tag 123 "abcd" book.title warning=message|lower profile=user.profile as the_result %}
Parsing until another block tag
Template tags can work in tandem. For instance, the standard {% comment %} tag hides everything until {% endcomment %}. To create a template tag such as this, use parser.parse() in your compilation function.

Here’s how a simplified {% comment %} tag might be implemented:

def do_comment(parser, token):
    nodelist = parser.parse(('endcomment',))
    parser.delete_first_token()
    return CommentNode()

class CommentNode(template.Node):
    def render(self, context):
        return ''
Note

The actual implementation of {% comment %} is slightly different in that it allows broken template tags to appear between {% comment %} and {% endcomment %}. It does so by calling parser.skip_past('endcomment') instead of parser.parse(('endcomment',)) followed by parser.delete_first_token(), thus avoiding the generation of a node list.

parser.parse() takes a tuple of names of block tags ‘’to parse until’‘. It returns an instance of django.template.NodeList, which is a list of all Node objects that the parser encountered ‘’before’’ it encountered any of the tags named in the tuple.

In "nodelist = parser.parse(('endcomment',))" in the above example, nodelist is a list of all nodes between the {% comment %} and {% endcomment %}, not counting {% comment %} and {% endcomment %} themselves.

After parser.parse() is called, the parser hasn’t yet “consumed” the {% endcomment %} tag, so the code needs to explicitly call parser.delete_first_token().

CommentNode.render() simply returns an empty string. Anything between {% comment %} and {% endcomment %} is ignored.




Parsing until another block tag, and saving contents
In the previous example, do_comment() discarded everything between {% comment %} and {% endcomment %}. Instead of doing that, it’s possible to do something with the code between block tags.

For example, here’s a custom template tag, {% upper %}, that capitalizes everything between itself and {% endupper %}.

Usage:

{% upper %}This will appear in uppercase, {{ your_name }}.{% endupper %}
As in the previous example, we’ll use parser.parse(). But this time, we pass the resulting nodelist to the Node:

def do_upper(parser, token):
    nodelist = parser.parse(('endupper',))
    parser.delete_first_token()
    return UpperNode(nodelist)

class UpperNode(template.Node):
    def __init__(self, nodelist):
        self.nodelist = nodelist
    def render(self, context):
        output = self.nodelist.render(context)
        return output.upper()
The only new concept here is the self.nodelist.render(context) in UpperNode.render().














FORM


HTML forms
In HTML, a form is a collection of elements inside <form>...</form> that allow a visitor to do things like enter text, select options, manipulate objects or controls, and so on, and then send that information back to the server.

Some of these form interface elements - text input or checkboxes - are fairly simple and are built into HTML itself. Others are much more complex; an interface that pops up a date picker or allows you to move a slider or manipulate controls will typically use JavaScript and CSS as well as HTML form <input> elements to achieve these effects.

As well as its <input> elements, a form must specify two things:

•where: the URL to which the data corresponding to the user’s input should be returned
•how: the HTTP method the data should be returned by




Django’s role in forms
Handling forms is a complex business. Consider Django’s admin, where numerous items of data of several different types may need to be prepared for display in a form, rendered as HTML, edited using a convenient interface, returned to the server, validated and cleaned up, and then saved or passed on for further processing.

Django’s form functionality can simplify and automate vast portions of this work, and can also do it more securely than most programmers would be able to do in code they wrote themselves.

Django handles three distinct parts of the work involved in forms:

•preparing and restructuring data to make it ready for rendering
•creating HTML forms for the data
•receiving and processing submitted forms and data from the client
It is possible to write code that does all of this manually, but Django can take care of it all for you.



The Django Form class
At the heart of this system of components is Django’s Form class. In much the same way that a Django model describes the logical structure of an object, its behavior, and the way its parts are represented to us, a Form class describes a form and determines how it works and appears.



Instantiating, processing, and rendering forms
When rendering an object in Django, we generally:

1.get hold of it in the view (fetch it from the database, for example)
2.pass it to the template context
3.expand it to HTML markup using template variables
Rendering a form in a template involves nearly the same work as rendering any other kind of object, but there are some key differences.

In the case of a model instance that contained no data, it would rarely if ever be useful to do anything with it in a template. On the other hand, it makes perfect sense to render an unpopulated form - that’s what we do when we want the user to populate it.

So when we handle a model instance in a view, we typically retrieve it from the database. When we’re dealing with a form we typically instantiate it in the view.

When we instantiate a form, we can opt to leave it empty or pre-populate it, for example with:

•data from a saved model instance (as in the case of admin forms for editing)
•data that we have collated from other sources
•data received from a previous HTML form submission
The last of these cases is the most interesting, because it’s what makes it possible for users not just to read a Web site, but to send information back to it too.


Building a form in Django

Example:
<form action="/your-name/" method="post">
    <label for="your_name">Your name: </label>
    <input id="your_name" type="text" name="your_name" value="{{ current_name }}">
    <input type="submit" value="OK">
</form>

Code:
from django import forms

class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)

The whole form, when rendered for the first time, will look like:

<label for="your_name">Your name: </label>
<input id="your_name" type="text" name="your_name" maxlength="100">
Note that it does not include the <form> tags, or a submit button. We’ll have to provide those ourselves in the template.


A Form instance has an is_valid() method, which runs validation routines for all its fields. When this method is called, if all fields contain valid data, it will:

•return True
•place the form’s data in its cleaned_data attribute.




The view
Form data sent back to a Django Web site is processed by a view, generally the same view which published the form. This allows us to reuse some of the same logic.

To handle the form we need to instantiate it in the view for the URL where we want it to be published:

from django.shortcuts import render
from django.http import HttpResponseRedirect

def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()

    return render(request, 'name.html', {'form': form})


If we arrive at this view with a GET request, it will create an empty form instance and place it in the template context to be rendered. 
If the form is submitted using a POST request, the view will once again create a form instance and populate it with data from the request: form = NameForm(request.POST) This is called “binding data to the form” (it is now a bound form).

We call the form’s is_valid() method; if it’s not True, we go back to the template with the form. This time the form is no longer empty (unbound) so the HTML form will be populated with the data previously submitted, where it can be edited and corrected as required.

If is_valid() is True, we’ll now be able to find all the validated form data in its cleaned_data attribute. We can use this data to update the database or do other processing before sending an HTTP redirect to the browser telling it where to go next.




name.html:

<form action="/your-name/" method="post">
    {% csrf_token %}
    {{ form }}
    <input type="submit" value="Submit" />
</form>



Forms and Cross Site Request Forgery protection

Django ships with an easy-to-use protection against Cross Site Request Forgeries. When submitting a form via POST with CSRF protection enabled you must use the csrf_token template tag as in the preceding example. 



HTML5 input types and browser validation

If your form includes a URLField, an EmailField or any integer field type, Django will use the url, email and number HTML5 input types. By default, browsers may apply their own validation on these fields, which may be stricter than Django’s validation. If you would like to disable this behavior, set the novalidate attribute on the form tag, or specify a different widget on the field, like TextInput.




Models and Forms

In fact if your form is going to be used to directly add or edit a Django model, a ModelForm can save you a great deal of time, effort, and code, because it will build a form, along with the appropriate fields and their attributes, from a Model class.



Bound and unbound form instances

•An unbound form has no data associated with it. When rendered to the user, it will be empty or will contain default values.
•A bound form has submitted data, and hence can be used to tell if that data is valid. If an invalid bound form is rendered, it can include inline error messages telling the user what data to correct.
The form’s is_bound attribute will tell you whether a form has data bound to it or not.

Example:


from django import forms

class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)
    sender = forms.EmailField()
    cc_myself = forms.BooleanField(required=False)


Widgets
Each form field has a corresponding Widget class, which in turn corresponds to an HTML form widget such as <input type="text">.

In most cases, the field will have a sensible default widget. For example, by default, a CharField will have a TextInput widget, that produces an <input type="text"> in the HTML. If you needed <textarea> instead, you’d specify the appropriate widget when defining your form field, as we have done for the message field.



Field data
Whatever the data submitted with a form, once it has been successfully validated by calling is_valid() (and is_valid() has returned True), the validated form data will be in the form.cleaned_data dictionary. This data will have been nicely converted into Python types for you.

You can still access the unvalidated data directly from request.POST at this point, but the validated data is better.

In the contact form example above, cc_myself will be a boolean value. Likewise, fields such as IntegerField and FloatField convert values to a Python int and float respectively.


from django.core.mail import send_mail

if form.is_valid():
    subject = form.cleaned_data['subject']
    message = form.cleaned_data['message']
    sender = form.cleaned_data['sender']
    cc_myself = form.cleaned_data['cc_myself']

    recipients = ['info@example.com']
    if cc_myself:
        recipients.append(sender)

    send_mail(subject, message, sender, recipients)
    return HttpResponseRedirect('/thanks/')



Working with form templates
Use {{ form }} will render its <label> and <input> elements appropriately.


Don’t forget that a form’s output does not include the surrounding <form> tags, or the form’s submit control. You will have to provide these yourself.

There are other output options though for the <label>/<input> pairs:

•{{ form.as_table }} will render them as table cells wrapped in <tr> tags
•{{ form.as_p }} will render them wrapped in <p> tags
•{{ form.as_ul }} will render them wrapped in <li> tags
Note that you’ll have to provide the surrounding <table> or <ul> elements yourself.

Here’s the output of {{ form.as_p }} for our ContactForm instance:

<p><label for="id_subject">Subject:</label>
    <input id="id_subject" type="text" name="subject" maxlength="100" /></p>
<p><label for="id_message">Message:</label>
    <input type="text" name="message" id="id_message" /></p>
<p><label for="id_sender">Sender:</label>
    <input type="email" name="sender" id="id_sender" /></p>
<p><label for="id_cc_myself">Cc myself:</label>
    <input type="checkbox" name="cc_myself" id="id_cc_myself" /></p>
Note that each form field has an ID attribute set to id_<field-name>, which is referenced by the accompanying label tag. This is important in ensuring that forms are accessible to assistive technology such as screen reader software. You can also customize the way in which labels and ids are generated.




Rendering fields manually
We don’t have to let Django unpack the form’s fields; we can do it manually if we like (allowing us to reorder the fields, for example). 
Each field is available as an attribute of the form using {{ form.name_of_field }}, and in a Django template, will be rendered appropriately. For example:

{{ form.non_field_errors }}
<div class="fieldWrapper">
    {{ form.subject.errors }}
    <label for="{{ form.subject.id_for_label }}">Email subject:</label>
    {{ form.subject }}
</div>
<div class="fieldWrapper">
    {{ form.message.errors }}
    <label for="{{ form.message.id_for_label }}">Your message:</label>
    {{ form.message }}
</div>
<div class="fieldWrapper">
    {{ form.sender.errors }}
    <label for="{{ form.sender.id_for_label }}">Your email address:</label>
    {{ form.sender }}
</div>
<div class="fieldWrapper">
    {{ form.cc_myself.errors }}
    <label for="{{ form.cc_myself.id_for_label }}">CC yourself?</label>
    {{ form.cc_myself }}
</div>
Complete <label> elements can also be generated using the label_tag(). For example:

<div class="fieldWrapper">
    {{ form.subject.errors }}
    {{ form.subject.label_tag }}
    {{ form.subject }}
</div>


Note {{ form.non_field_errors }} at the top of the form and the template lookup for errors on each field.

Using {{ form.name_of_field.errors }} displays a list of form errors, rendered as an unordered list. This might look like:

<ul class="errorlist">
    <li>Sender is required.</li>
</ul>
The list has a CSS class of errorlist to allow you to style its appearance. If you wish to further customize the display of errors you can do so by looping over them:

{% if form.subject.errors %}
    <ol>
    {% for error in form.subject.errors %}
        <li><strong>{{ error|escape }}</strong></li>
    {% endfor %}
    </ol>
{% endif %}


Looping over the form’s fields
If you’re using the same HTML for each of your form fields, you can reduce duplicate code by looping through each field in turn using a {% for %} loop:

{% for field in form %}
    <div class="fieldWrapper">
        {{ field.errors }}
        {{ field.label_tag }} {{ field }}
    </div>
{% endfor %}


Useful attributes on {{ field }} include:

{{ field.label }}
The label of the field, e.g. Email address.

{{ field.label_tag }}
The field’s label wrapped in the appropriate HTML <label> tag.
This includes the form’s label_suffix. For example, the default label_suffix is a colon:

<label for="id_email">Email address:</label>

{{ field.id_for_label }}
The ID that will be used for this field (id_email in the example above). If you are constructing the label manually, you may want to use this in lieu of label_tag. It’s also useful, for example, if you have some inline JavaScript and want to avoid hardcoding the field’s ID.

{{ field.value }}
The value of the field. e.g someone@example.com.

{{ field.html_name }}
The name of the field that will be used in the input element’s name field. This takes the form prefix into account, if it has been set.

{{ field.help_text }}
Any help text that has been associated with the field.

{{ field.errors }}
Outputs a <ul class="errorlist"> containing any validation errors corresponding to this field. You can customize the presentation of the errors with a {% for error in field.errors %} loop. In this case, each object in the loop is a simple string containing the error message.

{{ field.is_hidden }}
This attribute is True if the form field is a hidden field and False otherwise. It’s not particularly useful as a template variable, but could be useful in conditional tests such as:
{% if field.is_hidden %}
   {# Do something special #}
{% endif %}

{{ field.field }}
The Field instance from the form class that this BoundField wraps. You can use it to access Field attributes, e.g. {{ char_field.field.max_length }}.



Looping over hidden and visible fields
Django provides two methods on a form that allow you to loop over the hidden and visible fields independently: hidden_fields() and visible_fields(). 


{# Include the hidden fields #}
{% for hidden in form.hidden_fields %}
{{ hidden }}
{% endfor %}
{# Include the visible fields #}
{% for field in form.visible_fields %}
    <div class="fieldWrapper">
        {{ field.errors }}
        {{ field.label_tag }} {{ field }}
    </div>
{% endfor %}
This example does not handle any errors in the hidden fields. Usually, an error in a hidden field is a sign of form tampering, since normal form interaction won’t alter them. However, you could easily insert some error displays for those form errors, as well.



Reusable form templates
If your site uses the same rendering logic for forms in multiple places, you can reduce duplication by saving the form’s loop in a standalone template and using the include tag to reuse it in other templates:

# In your form template:
{% include "form_snippet.html" %}

# In form_snippet.html:
{% for field in form %}
    <div class="fieldWrapper">
        {{ field.errors }}
        {{ field.label_tag }} {{ field }}
    </div>
{% endfor %}
If the form object passed to a template has a different name within the context, you can alias it using the with argument of the include tag:

{% include "form_snippet.html" with form=comment_form %}


















Bound and unbound forms
A Form instance is either bound to a set of data, or unbound.

•If it’s bound to a set of data, it’s capable of validating that data and rendering the form as HTML with the data displayed in the HTML.
•If it’s unbound, it cannot do validation (because there’s no data to validate!), but it can still render the blank form as HTML.


class Form
To create an unbound Form instance, simply instantiate the class:

f = ContactForm()
To bind data to a form, pass the data as a dictionary as the first parameter to your Form class constructor:

data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
f = ContactForm(data)
In this dictionary, the keys are the field names, which correspond to the attributes in your Form class. The values are the data you’re trying to validate. These will usually be strings, but there’s no requirement that they be strings; the type of data you pass depends on the Field, as we’ll see in a moment.



Form.is_bound
If you need to distinguish between bound and unbound form instances at runtime, check the value of the form’s is_bound attribute:

f = ContactForm()
f.is_bound
False
f = ContactForm({'subject': 'hello'})
f.is_bound
True
Note that passing an empty dictionary creates a bound form with empty data:

f = ContactForm({})
f.is_bound
True



Using forms to validate data
Form.clean()
Implement a clean() method on your Form when you must add custom validation for fields that are interdependent. 


Form.is_valid()
The primary task of a Form object is to validate data. With a bound Form instance, call the is_valid() method to run validation and return a boolean designating whether the data was valid:

data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
f = ContactForm(data)
f.is_valid()
True
Let’s try with some invalid data. In this case, subject is blank (an error, because all fields are required by default) and sender is not a valid email address:

data = {'subject': '',
...         'message': 'Hi there',
...         'sender': 'invalid email address',
...         'cc_myself': True}
f = ContactForm(data)
f.is_valid()
False


Form.errors
Access the errors attribute to get a dictionary of error messages:

f.errors
{'sender': [u'Enter a valid email address.'], 'subject': [u'This field is required.']}
In this dictionary, the keys are the field names, and the values are lists of Unicode strings representing the error messages. The error messages are stored in lists because a field can have multiple error messages.

You can access errors without having to call is_valid() first. The form’s data will be validated the first time either you call is_valid() or access errors.

The validation routines will only get called once, regardless of how many times you access errors or call is_valid(). This means that if validation has side effects, those side effects will only be triggered once.



Form.errors.as_data()
Returns a dict that maps fields to their original ValidationError instances.

f.errors.as_data()
{'sender': [ValidationError(['Enter a valid email address.'])],
'subject': [ValidationError(['This field is required.'])]}
Use this method anytime you need to identify an error by its code. This enables things like rewriting the error’s message or writing custom logic in a view when a given error is present. It can also be used to serialize the errors in a custom format (e.g. XML); for instance, as_json() relies on as_data().

The need for the as_data() method is due to backwards compatibility. Previously ValidationError instances were lost as soon as their rendered error messages were added to the Form.errors dictionary. Ideally Form.errors would have stored ValidationError instances and methods with an as_ prefix could render them, but it had to be done the other way around in order not to break code that expects rendered error messages in Form.errors.




Form.errors.as_json(escape_html=False)
Returns the errors serialized as JSON.

f.errors.as_json()
{"sender": [{"message": "Enter a valid email address.", "code": "invalid"}],
"subject": [{"message": "This field is required.", "code": "required"}]}
By default, as_json() does not escape its output. If you are using it for something like AJAX requests to a form view where the client interprets the response and inserts errors into the page, you’ll want to be sure to escape the results on the client-side to avoid the possibility of a cross-site scripting attack. It’s trivial to do so using a JavaScript library like jQuery - simply use $(el).text(errorText) rather than .html().

If for some reason you don’t want to use client-side escaping, you can also set escape_html=True and error messages will be escaped so you can use them directly in HTML.


Form.add_error(field, error)

This method allows adding errors to specific fields from within the Form.clean() method, or from outside the form altogether; for instance from a view.

The field argument is the name of the field to which the errors should be added. If its value is None the error will be treated as a non-field error as returned by Form.non_field_errors().

The error argument can be a simple string, or preferably an instance of ValidationError. See Raising ValidationError for best practices when defining form errors.

Note that Form.add_error() automatically removes the relevant field from cleaned_data.



Form.non_field_errors()
This method returns the list of errors from Form.errors that aren’t associated with a particular field. This includes ValidationErrors that are raised in Form.clean() and errors added using Form.add_error(None, "...").



Behavior of unbound forms
It’s meaningless to validate a form with no data, but, for the record, here’s what happens with unbound forms:

f = ContactForm()
f.is_valid()
False
f.errors
{}




Form.initial
Use initial to declare the initial value of form fields at runtime. For example, you might want to fill in a username field with the username of the current session.

To accomplish this, use the initial argument to a Form. This argument, if given, should be a dictionary mapping field names to initial values. Only include the fields for which you’re specifying an initial value; it’s not necessary to include every field in your form. For example:

f = ContactForm(initial={'subject': 'Hi there!'})
These values are only displayed for unbound forms, and they’re not used as fallback values if a particular value isn’t provided.

Note that if a Field defines initial and you include initial when instantiating the Form, then the latter initial will have precedence. In this example, initial is provided both at the field level and at the form instance level, and the latter gets precedence:

from django import forms
class CommentForm(forms.Form):
...     name = forms.CharField(initial='class')
...     url = forms.URLField()
...     comment = forms.CharField()
f = CommentForm(initial={'name': 'instance'}, auto_id=False)
print(f)
<tr><th>Name:</th><td><input type="text" name="name" value="instance" /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url" /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" /></td></tr>



Form.has_changed()
Use the has_changed() method on your Form when you need to check if the form data has been changed from the initial data.

data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
f = ContactForm(data, initial=data)
f.has_changed()
False
When the form is submitted, we reconstruct it and provide the original data so that the comparison can be done:

f = ContactForm(request.POST, initial=data)
f.has_changed()
has_changed() will be True if the data from request.POST differs from what was provided in initial or False otherwise.



Form.fields
You can access the fields of Form instance from its fields attribute:

for row in f.fields.values(): print(row)
...
<django.forms.fields.CharField object at 0x7ffaac632510>
<django.forms.fields.URLField object at 0x7ffaac632f90>
<django.forms.fields.CharField object at 0x7ffaac3aa050>
f.fields['name']
<django.forms.fields.CharField object at 0x7ffaac6324d0>
You can alter the field of Form instance to change the way it is presented in the form:

f.as_table().split('\n')[0]
'<tr><th>Name:</th><td><input name="name" type="text" value="instance" /></td></tr>'
f.fields['name'].label = "Username"
f.as_table().split('\n')[0]
'<tr><th>Username:</th><td><input name="name" type="text" value="instance" /></td></tr>'
Beware not to alter the base_fields attribute because this modification will influence all subsequent ContactForm instances within the same Python process:

f.base_fields['name'].label = "Username"
another_f = CommentForm(auto_id=False)
another_f.as_table().split('\n')[0]
'<tr><th>Username:</th><td><input name="name" type="text" value="class" /></td></tr>'



Form.cleaned_data
Each field in a Form class is responsible not only for validating data, but also for “cleaning” it – normalizing it to a consistent format. This is a nice feature, because it allows data for a particular field to be input in a variety of ways, always resulting in consistent output.

For example, DateField normalizes input into a Python datetime.date object. Regardless of whether you pass it a string in the format '1994-07-15', a datetime.date object, or a number of other formats, DateField will always normalize it to a datetime.date object as long as it’s valid.

Once you’ve created a Form instance with a set of data and validated it, you can access the clean data via its cleaned_data attribute:

data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
f = ContactForm(data)
f.is_valid()
True
f.cleaned_data
{'cc_myself': True, 'message': u'Hi there', 'sender': u'foo@example.com', 'subject': u'hello'}
Note that any text-based field – such as CharField or EmailField – always cleans the input into a Unicode string. We’ll cover the encoding implications later in this document.

If your data does not validate, the cleaned_data dictionary contains only the valid fields:

data = {'subject': '',
...         'message': 'Hi there',
...         'sender': 'invalid email address',
...         'cc_myself': True}
f = ContactForm(data)
f.is_valid()
False
f.cleaned_data
{'cc_myself': True, 'message': u'Hi there'}
cleaned_data will always only contain a key for fields defined in the Form, even if you pass extra data when you define the Form. In this example, we pass a bunch of extra fields to the ContactForm constructor, but cleaned_data contains only the form’s fields:

data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True,
...         'extra_field_1': 'foo',
...         'extra_field_2': 'bar',
...         'extra_field_3': 'baz'}
f = ContactForm(data)
f.is_valid()
True
f.cleaned_data # Doesn't contain extra_field_1, etc.
{'cc_myself': True, 'message': u'Hi there', 'sender': u'foo@example.com', 'subject': u'hello'}
When the Form is valid, cleaned_data will include a key and value for all its fields, even if the data didn’t include a value for some optional fields. In this example, the data dictionary doesn’t include a value for the nick_name field, but cleaned_data includes it, with an empty value:

from django.forms import Form
class OptionalPersonForm(Form):
...     first_name = CharField()
...     last_name = CharField()
...     nick_name = CharField(required=False)
data = {'first_name': u'John', 'last_name': u'Lennon'}
f = OptionalPersonForm(data)
f.is_valid()
True
f.cleaned_data
{'nick_name': u'', 'first_name': u'John', 'last_name': u'Lennon'}
In this above example, the cleaned_data value for nick_name is set to an empty string, because nick_name is CharField, and CharFields treat empty values as an empty string. Each field type knows what its “blank” value is – e.g., for DateField, it’s None instead of the empty string. For full details on each field’s behavior in this case, see the “Empty value” note for each field in the “Built-in Field classes” section below.



Outputting forms as HTML
The second task of a Form object is to render itself as HTML. To do so, simply print it:

f = ContactForm()
print(f)
<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" /></td></tr>
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" /></td></tr>
<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" /></td></tr>
<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" /></td></tr>
If the form is bound to data, the HTML output will include that data appropriately. For example, if a field is represented by an <input type="text">, the data will be in the value attribute. If a field is represented by an <input type="checkbox">, then that HTML will include checked="checked" if appropriate:

data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
f = ContactForm(data)
print(f)
<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" value="hello" /></td></tr>
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" value="Hi there" /></td></tr>
<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" value="foo@example.com" /></td></tr>
<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" checked="checked" /></td></tr>


This default output is a two-column HTML table, with a <tr> for each field. Notice the following:

•For flexibility, the output does not include the <table> and </table> tags, nor does it include the <form> and </form> tags or an <input type="submit"> tag. It’s your job to do that.
•Each field type has a default HTML representation. CharField is represented by an <input type="text"> and EmailField by an <input type="email">. BooleanField is represented by an <input type="checkbox">. Note these are merely sensible defaults; you can specify which HTML to use for a given field by using widgets, which we’ll explain shortly.
•The HTML name for each tag is taken directly from its attribute name in the ContactForm class.
•The text label for each field – e.g. 'Subject:', 'Message:' and 'Cc myself:' is generated from the field name by converting all underscores to spaces and upper-casing the first letter. Again, note these are merely sensible defaults; you can also specify labels manually.
•Each text label is surrounded in an HTML <label> tag, which points to the appropriate form field via its id. Its id, in turn, is generated by prepending 'id_' to the field name. The id attributes and <label> tags are included in the output by default, to follow best practices, but you can change that behavior.
Although <table> output is the default output style when you print a form, other output styles are available. Each style is available as a method on a form object, and each rendering method returns a Unicode object.




Form.as_p()
as_p() renders the form as a series of <p> tags, with each <p> containing one field:

f = ContactForm()
f.as_p()
u'<p><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" /></p>\n<p><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" /></p>\n<p><label for="id_sender">Sender:</label> <input type="text" name="sender" id="id_sender" /></p>\n<p><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></p>'
print(f.as_p())
<p><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" /></p>
<p><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" /></p>
<p><label for="id_sender">Sender:</label> <input type="email" name="sender" id="id_sender" /></p>
<p><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></p>



Form.as_ul()
as_ul() renders the form as a series of <li> tags, with each <li> containing one field. It does not include the <ul> or </ul>, so that you can specify any HTML attributes on the <ul> for flexibility:

f = ContactForm()
f.as_ul()
u'<li><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" /></li>\n<li><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" /></li>\n<li><label for="id_sender">Sender:</label> <input type="email" name="sender" id="id_sender" /></li>\n<li><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></li>'
print(f.as_ul())
<li><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" /></li>
<li><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" /></li>
<li><label for="id_sender">Sender:</label> <input type="email" name="sender" id="id_sender" /></li>
<li><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></li>



Form.as_table()
Finally, as_table() outputs the form as an HTML <table>. This is exactly the same as print. In fact, when you print a form object, it calls its as_table() method behind the scenes:

f = ContactForm()
f.as_table()
u'<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" /></td></tr>\n<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" /></td></tr>\n<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" /></td></tr>\n<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" /></td></tr>'
print(f.as_table())
<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" /></td></tr>
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" /></td></tr>
<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" /></td></tr>
<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" /></td></tr>



Form.error_css_class
Form.required_css_class
It’s pretty common to style form rows and fields that are required or have errors. For example, you might want to present required form rows in bold and highlight errors in red.

The Form class has a couple of hooks you can use to add class attributes to required rows or to rows with errors: simply set the Form.error_css_class and/or Form.required_css_class attributes:

from django.forms import Form

class ContactForm(Form):
    error_css_class = 'error'
    required_css_class = 'required'

    # ... and the rest of your fields here
Once you’ve done that, rows will be given "error" and/or "required" classes, as needed. The HTML will look something like:

f = ContactForm(data)
print(f.as_table())
<tr class="required"><th><label for="id_subject">Subject:</label>    ...
<tr class="required"><th><label for="id_message">Message:</label>    ...
<tr class="required error"><th><label for="id_sender">Sender:</label>      ...
<tr><th><label for="id_cc_myself">Cc myself:<label> ...



Form.auto_id
By default, the form rendering methods include:

•HTML id attributes on the form elements.
•The corresponding <label> tags around the labels. An HTML <label> tag designates which label text is associated with which form element. This small enhancement makes forms more usable and more accessible to assistive devices. It’s always a good idea to use <label> tags.
The id attribute values are generated by prepending id_ to the form field names. This behavior is configurable, though, if you want to change the id convention or remove HTML id attributes and <label> tags entirely.

Use the auto_id argument to the Form constructor to control the id and label behavior. This argument must be True, False or a string.

If auto_id is False, then the form output will not include <label> tags nor id attributes:

f = ContactForm(auto_id=False)
print(f.as_table())
<tr><th>Subject:</th><td><input type="text" name="subject" maxlength="100" /></td></tr>
<tr><th>Message:</th><td><input type="text" name="message" /></td></tr>
<tr><th>Sender:</th><td><input type="email" name="sender" /></td></tr>
<tr><th>Cc myself:</th><td><input type="checkbox" name="cc_myself" /></td></tr>
print(f.as_ul())
<li>Subject: <input type="text" name="subject" maxlength="100" /></li>
<li>Message: <input type="text" name="message" /></li>
<li>Sender: <input type="email" name="sender" /></li>
<li>Cc myself: <input type="checkbox" name="cc_myself" /></li>
print(f.as_p())
<p>Subject: <input type="text" name="subject" maxlength="100" /></p>
<p>Message: <input type="text" name="message" /></p>
<p>Sender: <input type="email" name="sender" /></p>
<p>Cc myself: <input type="checkbox" name="cc_myself" /></p>
If auto_id is set to True, then the form output will include <label> tags and will simply use the field name as its id for each form field:

f = ContactForm(auto_id=True)
print(f.as_table())
<tr><th><label for="subject">Subject:</label></th><td><input id="subject" type="text" name="subject" maxlength="100" /></td></tr>
<tr><th><label for="message">Message:</label></th><td><input type="text" name="message" id="message" /></td></tr>
<tr><th><label for="sender">Sender:</label></th><td><input type="email" name="sender" id="sender" /></td></tr>
<tr><th><label for="cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="cc_myself" /></td></tr>
print(f.as_ul())
<li><label for="subject">Subject:</label> <input id="subject" type="text" name="subject" maxlength="100" /></li>
<li><label for="message">Message:</label> <input type="text" name="message" id="message" /></li>
<li><label for="sender">Sender:</label> <input type="email" name="sender" id="sender" /></li>
<li><label for="cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="cc_myself" /></li>
print(f.as_p())
<p><label for="subject">Subject:</label> <input id="subject" type="text" name="subject" maxlength="100" /></p>
<p><label for="message">Message:</label> <input type="text" name="message" id="message" /></p>
<p><label for="sender">Sender:</label> <input type="email" name="sender" id="sender" /></p>
<p><label for="cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="cc_myself" /></p>
If auto_id is set to a string containing the format character '%s', then the form output will include <label> tags, and will generate id attributes based on the format string. For example, for a format string 'field_%s', a field named subject will get the id value 'field_subject'. Continuing our example:

f = ContactForm(auto_id='id_for_%s')
print(f.as_table())
<tr><th><label for="id_for_subject">Subject:</label></th><td><input id="id_for_subject" type="text" name="subject" maxlength="100" /></td></tr>
<tr><th><label for="id_for_message">Message:</label></th><td><input type="text" name="message" id="id_for_message" /></td></tr>
<tr><th><label for="id_for_sender">Sender:</label></th><td><input type="email" name="sender" id="id_for_sender" /></td></tr>
<tr><th><label for="id_for_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></td></tr>
print(f.as_ul())
<li><label for="id_for_subject">Subject:</label> <input id="id_for_subject" type="text" name="subject" maxlength="100" /></li>
<li><label for="id_for_message">Message:</label> <input type="text" name="message" id="id_for_message" /></li>
<li><label for="id_for_sender">Sender:</label> <input type="email" name="sender" id="id_for_sender" /></li>
<li><label for="id_for_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></li>
print(f.as_p())
<p><label for="id_for_subject">Subject:</label> <input id="id_for_subject" type="text" name="subject" maxlength="100" /></p>
<p><label for="id_for_message">Message:</label> <input type="text" name="message" id="id_for_message" /></p>
<p><label for="id_for_sender">Sender:</label> <input type="email" name="sender" id="id_for_sender" /></p>
<p><label for="id_for_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></p>
If auto_id is set to any other true value – such as a string that doesn’t include %s – then the library will act as if auto_id is True.

By default, auto_id is set to the string 'id_%s'.




Form.label_suffix
A translatable string (defaults to a colon (:) in English) that will be appended after any label name when a form is rendered.



It’s possible to customize that character, or omit it entirely, using the label_suffix parameter:

f = ContactForm(auto_id='id_for_%s', label_suffix='')
print(f.as_ul())
<li><label for="id_for_subject">Subject</label> <input id="id_for_subject" type="text" name="subject" maxlength="100" /></li>
<li><label for="id_for_message">Message</label> <input type="text" name="message" id="id_for_message" /></li>
<li><label for="id_for_sender">Sender</label> <input type="email" name="sender" id="id_for_sender" /></li>
<li><label for="id_for_cc_myself">Cc myself</label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></li>
f = ContactForm(auto_id='id_for_%s', label_suffix=' ->')
print(f.as_ul())
<li><label for="id_for_subject">Subject -></label> <input id="id_for_subject" type="text" name="subject" maxlength="100" /></li>
<li><label for="id_for_message">Message -></label> <input type="text" name="message" id="id_for_message" /></li>
<li><label for="id_for_sender">Sender -></label> <input type="email" name="sender" id="id_for_sender" /></li>
<li><label for="id_for_cc_myself">Cc myself -></label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></li>
Note that the label suffix is added only if the last character of the label isn’t a punctuation character (in English, those are ., !, ? or :).




Notes on field ordering
In the as_p(), as_ul() and as_table() shortcuts, the fields are displayed in the order in which you define them in your form class. For example, in the ContactForm example, the fields are defined in the order subject, message, sender, cc_myself. To reorder the HTML output, just change the order in which those fields are listed in the class.


How errors are displayed
If you render a bound Form object, the act of rendering will automatically run the form’s validation if it hasn’t already happened, and the HTML output will include the validation errors as a <ul class="errorlist"> near the field. The particular positioning of the error messages depends on the output method you’re using:

data = {'subject': '',
...         'message': 'Hi there',
...         'sender': 'invalid email address',
...         'cc_myself': True}
f = ContactForm(data, auto_id=False)
print(f.as_table())
<tr><th>Subject:</th><td><ul class="errorlist"><li>This field is required.</li></ul><input type="text" name="subject" maxlength="100" /></td></tr>
<tr><th>Message:</th><td><input type="text" name="message" value="Hi there" /></td></tr>
<tr><th>Sender:</th><td><ul class="errorlist"><li>Enter a valid email address.</li></ul><input type="email" name="sender" value="invalid email address" /></td></tr>
<tr><th>Cc myself:</th><td><input checked="checked" type="checkbox" name="cc_myself" /></td></tr>
print(f.as_ul())
<li><ul class="errorlist"><li>This field is required.</li></ul>Subject: <input type="text" name="subject" maxlength="100" /></li>
<li>Message: <input type="text" name="message" value="Hi there" /></li>
<li><ul class="errorlist"><li>Enter a valid email address.</li></ul>Sender: <input type="email" name="sender" value="invalid email address" /></li>
<li>Cc myself: <input checked="checked" type="checkbox" name="cc_myself" /></li>
print(f.as_p())
<p><ul class="errorlist"><li>This field is required.</li></ul></p>
<p>Subject: <input type="text" name="subject" maxlength="100" /></p>
<p>Message: <input type="text" name="message" value="Hi there" /></p>
<p><ul class="errorlist"><li>Enter a valid email address.</li></ul></p>
<p>Sender: <input type="email" name="sender" value="invalid email address" /></p>
<p>Cc myself: <input checked="checked" type="checkbox" name="cc_myself" /></p>



Customizing the error list format
By default, forms use django.forms.utils.ErrorList to format validation errors. If you’d like to use an alternate class for displaying errors, you can pass that in at construction time (replace __str__ by __unicode__ on Python 2):

from django.forms.utils import ErrorList
class DivErrorList(ErrorList):
...     def __str__(self):              # __unicode__ on Python 2
...         return self.as_divs()
...     def as_divs(self):
...         if not self: return ''
...         return '<div class="errorlist">%s</div>' % ''.join(['<div class="error">%s</div>' % e for e in self])
f = ContactForm(data, auto_id=False, error_class=DivErrorList)
f.as_p()
<div class="errorlist"><div class="error">This field is required.</div></div>
<p>Subject: <input type="text" name="subject" maxlength="100" /></p>
<p>Message: <input type="text" name="message" value="Hi there" /></p>
<div class="errorlist"><div class="error">Enter a valid email address.</div></div>
<p>Sender: <input type="email" name="sender" value="invalid email address" /></p>
<p>Cc myself: <input checked="checked" type="checkbox" name="cc_myself" /></p>



class BoundField
Used to display HTML or access attributes for a single field of a Form instance.

The __str__() (__unicode__ on Python 2) method of this object displays the HTML for this field.

To retrieve a single BoundField, use dictionary lookup syntax on your form using the field’s name as the key:

form = ContactForm()
print(form['subject'])
<input id="id_subject" type="text" name="subject" maxlength="100" />
To retrieve all BoundField objects, iterate the form:

form = ContactForm()
for boundfield in form: print(boundfield)
<input id="id_subject" type="text" name="subject" maxlength="100" />
<input type="text" name="message" id="id_message" />
<input type="email" name="sender" id="id_sender" />
<input type="checkbox" name="cc_myself" id="id_cc_myself" />
The field-specific output honors the form object’s auto_id setting:

f = ContactForm(auto_id=False)
print(f['message'])
<input type="text" name="message" />
f = ContactForm(auto_id='id_%s')
print(f['message'])
<input type="text" name="message" id="id_message" />
For a field’s list of errors, access the field’s errors attribute.


BoundField.errors
A list-like object that is displayed as an HTML <ul class="errorlist"> when printed:

data = {'subject': 'hi', 'message': '', 'sender': '', 'cc_myself': ''}
f = ContactForm(data, auto_id=False)
print(f['message'])
<input type="text" name="message" />
f['message'].errors
[u'This field is required.']
print(f['message'].errors)
<ul class="errorlist"><li>This field is required.</li></ul>
f['subject'].errors
[]
print(f['subject'].errors)

str(f['subject'].errors)
''


BoundField.label_tag(contents=None, attrs=None, label_suffix=None)
To separately render the label tag of a form field, you can call its label_tag method:

f = ContactForm(data)
print(f['message'].label_tag())
<label for="id_message">Message:</label>
Optionally, you can provide the contents parameter which will replace the auto-generated label tag. An optional attrs dictionary may contain additional attributes for the <label> tag.




BoundField.css_classes()
When you use Django’s rendering shortcuts, CSS classes are used to indicate required form fields or fields that contain errors. If you’re manually rendering a form, you can access these CSS classes using the css_classes method:

f = ContactForm(data)
f['message'].css_classes()
'required'
If you want to provide some additional classes in addition to the error and required classes that may be required, you can provide those classes as an argument:

f = ContactForm(data)
f['message'].css_classes('foo bar')
'foo bar required'


BoundField.value()
Use this method to render the raw value of this field as it would be rendered by a Widget:

initial = {'subject': 'welcome'}
unbound_form = ContactForm(initial=initial)
bound_form = ContactForm(data, initial=initial)
print(unbound_form['subject'].value())
welcome
print(bound_form['subject'].value())
hi


BoundField.id_for_label
Use this property to render the ID of this field. For example, if you are manually constructing a <label> in your template (despite the fact that label_tag() will do this for you):

<label for="{{ form.my_field.id_for_label }}">...</label>{{ my_field }}
By default, this will be the field’s name prefixed by id_ (“id_my_field” for the example above). You may modify the ID by setting attrs on the field’s widget. For example, declaring a field like this:

my_field = forms.CharField(widget=forms.TextInput(attrs={'id': 'myFIELD'}))
and using the template above, would render something like:

<label for="myFIELD">...</label><input id="myFIELD" type="text" name="my_field" />


Binding uploaded files to a form
Dealing with forms that have FileField and ImageField fields is a little more complicated than a normal form.

Firstly, in order to upload files, you’ll need to make sure that your <form> element correctly defines the enctype as "multipart/form-data":

<form enctype="multipart/form-data" method="post" action="/foo/">
Secondly, when you use the form, you need to bind the file data. File data is handled separately to normal form data, so when your form contains a FileField and ImageField, you will need to specify a second argument when you bind your form. So if we extend our ContactForm to include an ImageField called mugshot, we need to bind the file data containing the mugshot image:

# Bound form with an image field
from django.core.files.uploadedfile import SimpleUploadedFile
data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
file_data = {'mugshot': SimpleUploadedFile('face.jpg', <file data>)}
f = ContactFormWithMugshot(data, file_data)
In practice, you will usually specify request.FILES as the source of file data (just like you use request.POST as the source of form data):

# Bound form with an image field, data from the request
f = ContactFormWithMugshot(request.POST, request.FILES)
Constructing an unbound form is the same as always – just omit both form data and file data:

# Unbound form with an image field
f = ContactFormWithMugshot()
Testing for multipart forms
Form.is_multipart()
If you’re writing reusable views or templates, you may not know ahead of time whether your form is a multipart form or not. The is_multipart() method tells you whether the form requires multipart encoding for submission:

f = ContactFormWithMugshot()
f.is_multipart()
True
Here’s an example of how you might use this in a template:

{% if form.is_multipart %}
    <form enctype="multipart/form-data" method="post" action="/foo/">
{% else %}
    <form method="post" action="/foo/">
{% endif %}
{{ form }}
</form>


Subclassing forms
If you have multiple Form classes that share fields, you can use subclassing to remove redundancy.

When you subclass a custom Form class, the resulting subclass will include all fields of the parent class(es), followed by the fields you define in the subclass.

In this example, ContactFormWithPriority contains all the fields from ContactForm, plus an additional field, priority. The ContactForm fields are ordered first:

class ContactFormWithPriority(ContactForm):
...     priority = forms.CharField()
f = ContactFormWithPriority(auto_id=False)
print(f.as_ul())
<li>Subject: <input type="text" name="subject" maxlength="100" /></li>
<li>Message: <input type="text" name="message" /></li>
<li>Sender: <input type="email" name="sender" /></li>
<li>Cc myself: <input type="checkbox" name="cc_myself" /></li>
<li>Priority: <input type="text" name="priority" /></li>
It’s possible to subclass multiple forms, treating forms as “mix-ins.” In this example, BeatleForm subclasses both PersonForm and InstrumentForm (in that order), and its field list includes the fields from the parent classes:

from django.forms import Form
class PersonForm(Form):
...     first_name = CharField()
...     last_name = CharField()
class InstrumentForm(Form):
...     instrument = CharField()
class BeatleForm(PersonForm, InstrumentForm):
...     haircut_type = CharField()
b = BeatleForm(auto_id=False)
print(b.as_ul())
<li>First name: <input type="text" name="first_name" /></li>
<li>Last name: <input type="text" name="last_name" /></li>
<li>Instrument: <input type="text" name="instrument" /></li>
<li>Haircut type: <input type="text" name="haircut_type" /></li>

•It’s possible to declaratively remove a Field inherited from a parent class by setting the name to be None on the subclass. For example:

from django import forms

class ParentForm(forms.Form):
...     name = forms.CharField()
...     age = forms.IntegerField()

class ChildForm(ParentForm):
...     name = None

ChildForm().fields.keys()
... ['age']



Form.prefix
You can put several Django forms inside one <form> tag. To give each Form its own namespace, use the prefix keyword argument:

mother = PersonForm(prefix="mother")
father = PersonForm(prefix="father")
print(mother.as_ul())
<li><label for="id_mother-first_name">First name:</label> <input type="text" name="mother-first_name" id="id_mother-first_name" /></li>
<li><label for="id_mother-last_name">Last name:</label> <input type="text" name="mother-last_name" id="id_mother-last_name" /></li>
print(father.as_ul())
<li><label for="id_father-first_name">First name:</label> <input type="text" name="father-first_name" id="id_father-first_name" /></li>
<li><label for="id_father-last_name">Last name:</label> <input type="text" name="father-last_name" id="id_father-last_name" /></li>












Form fields


Field.clean(value)
Although the primary way you’ll use Field classes is in Form classes, you can also instantiate them and use them directly to get a better idea of how they work. Each Field instance has a clean() method, which takes a single argument and either raises a django.forms.ValidationError exception or returns the clean value:

from django import forms
f = forms.EmailField()
f.clean('foo@example.com')
u'foo@example.com'
f.clean('invalid email address')
Traceback (most recent call last):
...
ValidationError: [u'Enter a valid email address.']



Core field arguments
Each Field class constructor takes at least these arguments. Some Field classes take additional, field-specific arguments, but the following should always be accepted:


Field.required
By default, each Field class assumes the value is required, so if you pass an empty value – either None or the empty string ("") – then clean() will raise a ValidationError exception:

from django import forms
f = forms.CharField()
f.clean('foo')
u'foo'
f.clean('')
Traceback (most recent call last):
...
ValidationError: [u'This field is required.']
f.clean(None)
Traceback (most recent call last):
...
ValidationError: [u'This field is required.']
f.clean(' ')
u' '
f.clean(0)
u'0'
f.clean(True)
u'True'
f.clean(False)
u'False'
To specify that a field is not required, pass required=False to the Field constructor:

f = forms.CharField(required=False)
f.clean('foo')
u'foo'
f.clean('')
u''
f.clean(None)
u''
f.clean(0)
u'0'
f.clean(True)
u'True'
f.clean(False)
u'False'
If a Field has required=False and you pass clean() an empty value, then clean() will return a normalized empty value rather than raising ValidationError. For CharField, this will be a Unicode empty string. For other Field classes, it might be None. (This varies from field to field.)



Field.label
The label argument lets you specify the “human-friendly” label for this field. This is used when the Field is displayed in a Form.

the default label for a Field is generated from the field name by converting all underscores to spaces and upper-casing the first letter. Specify label if that default behavior doesn’t result in an adequate label.

Here’s a full example Form that implements label for two of its fields. We’ve specified auto_id=False to simplify the output:

from django import forms
class CommentForm(forms.Form):
...     name = forms.CharField(label='Your name')
...     url = forms.URLField(label='Your Web site', required=False)
...     comment = forms.CharField()
f = CommentForm(auto_id=False)
print(f)
<tr><th>Your name:</th><td><input type="text" name="name" /></td></tr>
<tr><th>Your Web site:</th><td><input type="url" name="url" /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" /></td></tr>



Field.initial
The initial argument lets you specify the initial value to use when rendering this Field in an unbound Form.

To specify dynamic initial data, see the Form.initial parameter.

The use-case for this is when you want to display an “empty” form in which a field is initialized to a particular value. For example:

from django import forms
class CommentForm(forms.Form):
...     name = forms.CharField(initial='Your name')
...     url = forms.URLField(initial='http://')
...     comment = forms.CharField()
f = CommentForm(auto_id=False)
print(f)
<tr><th>Name:</th><td><input type="text" name="name" value="Your name" /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url" value="http://" /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" /></td></tr>
You may be thinking, why not just pass a dictionary of the initial values as data when displaying the form? Well, if you do that, you’ll trigger validation, and the HTML output will include any validation errors:

class CommentForm(forms.Form):
...     name = forms.CharField()
...     url = forms.URLField()
...     comment = forms.CharField()
default_data = {'name': 'Your name', 'url': 'http://'}
f = CommentForm(default_data, auto_id=False)
print(f)
<tr><th>Name:</th><td><input type="text" name="name" value="Your name" /></td></tr>
<tr><th>Url:</th><td><ul class="errorlist"><li>Enter a valid URL.</li></ul><input type="url" name="url" value="http://" /></td></tr>
<tr><th>Comment:</th><td><ul class="errorlist"><li>This field is required.</li></ul><input type="text" name="comment" /></td></tr>
This is why initial values are only displayed for unbound forms. For bound forms, the HTML output will use the bound data.

Also note that initial values are not used as “fallback” data in validation if a particular field’s value is not given. initial values are only intended for initial form display:

class CommentForm(forms.Form):
...     name = forms.CharField(initial='Your name')
...     url = forms.URLField(initial='http://')
...     comment = forms.CharField()
data = {'name': '', 'url': '', 'comment': 'Foo'}
f = CommentForm(data)
f.is_valid()
False
# The form does *not* fall back to using the initial values.
f.errors
{'url': [u'This field is required.'], 'name': [u'This field is required.']}
Instead of a constant, you can also pass any callable:

import datetime
class DateForm(forms.Form):
...     day = forms.DateField(initial=datetime.date.today)
print(DateForm())
<tr><th>Day:</th><td><input type="text" name="day" value="12/23/2008" /><td></tr>
The callable will be evaluated only when the unbound form is displayed, not when it is defined.




Field.widget
The widget argument lets you specify a Widget class to use when rendering this Field. 




Field.help_text
The help_text argument lets you specify descriptive text for this Field. If you provide help_text, it will be displayed next to the Field when the Field is rendered by one of the convenience Form methods (e.g., as_ul()).

Here’s a full example Form that implements help_text for two of its fields. We’ve specified auto_id=False to simplify the output:

from django import forms
class HelpTextContactForm(forms.Form):
...     subject = forms.CharField(max_length=100, help_text='100 characters max.')
...     message = forms.CharField()
...     sender = forms.EmailField(help_text='A valid email address, please.')
...     cc_myself = forms.BooleanField(required=False)
f = HelpTextContactForm(auto_id=False)
print(f.as_table())
<tr><th>Subject:</th><td><input type="text" name="subject" maxlength="100" /><br /><span class="helptext">100 characters max.</span></td></tr>
<tr><th>Message:</th><td><input type="text" name="message" /></td></tr>
<tr><th>Sender:</th><td><input type="email" name="sender" /><br />A valid email address, please.</td></tr>
<tr><th>Cc myself:</th><td><input type="checkbox" name="cc_myself" /></td></tr>
print(f.as_ul()))
<li>Subject: <input type="text" name="subject" maxlength="100" /> <span class="helptext">100 characters max.</span></li>
<li>Message: <input type="text" name="message" /></li>
<li>Sender: <input type="email" name="sender" /> A valid email address, please.</li>
<li>Cc myself: <input type="checkbox" name="cc_myself" /></li>
print(f.as_p())
<p>Subject: <input type="text" name="subject" maxlength="100" /> <span class="helptext">100 characters max.</span></p>
<p>Message: <input type="text" name="message" /></p>
<p>Sender: <input type="email" name="sender" /> A valid email address, please.</p>
<p>Cc myself: <input type="checkbox" name="cc_myself" /></p>



Field.error_messages
The error_messages argument lets you override the default messages that the field will raise. Pass in a dictionary with keys matching the error messages you want to override. For example, here is the default error message:

from django import forms
generic = forms.CharField()
generic.clean('')
Traceback (most recent call last):
  ...
ValidationError: [u'This field is required.']
And here is a custom error message:

name = forms.CharField(error_messages={'required': 'Please enter your name'})
name.clean('')
Traceback (most recent call last):
  ...
ValidationError: [u'Please enter your name']
In the built-in Field classes section below, each Field defines the error message keys it uses.


Field.validators
The validators argument lets you provide a list of validation functions for this field.




Field.localize
The localize argument enables the localization of form data, input as well as the rendered output.





Built-in Field classes


BooleanField
class BooleanField(**kwargs)
•Default widget: CheckboxInput
•Empty value: False
•Normalizes to: A Python True or False value.
•Validates that the value is True (e.g. the check box is checked) if the field has required=True.
•Error message keys: required
Note

Since all Field subclasses have required=True by default, the validation condition here is important. If you want to include a boolean in your form that can be either True or False (e.g. a checked or unchecked checkbox), you must remember to pass in required=False when creating the BooleanField.



CharField
class CharField(**kwargs)
•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates max_length or min_length, if they are provided. Otherwise, all inputs are valid.
•Error message keys: required, max_length, min_length
Has two optional arguments for validation:

max_length
min_length
If provided, these arguments ensure that the string is at most or at least the given length.



ChoiceField
class ChoiceField(**kwargs)
•Default widget: Select
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value exists in the list of choices.
•Error message keys: required, invalid_choice
The invalid_choice error message may contain %(value)s, which will be replaced with the selected choice.

Takes one extra required argument:

choices
An iterable (e.g., a list or tuple) of 2-tuples to use as choices for this field. This argument accepts the same formats as the choices argument to a model field. See the model field reference documentation on choices for more details.



TypedChoiceField
class TypedChoiceField(**kwargs)
Just like a ChoiceField, except TypedChoiceField takes two extra arguments, coerce and empty_value.

•Default widget: Select
•Empty value: Whatever you’ve given as empty_value
•Normalizes to: A value of the type provided by the coerce argument.
•Validates that the given value exists in the list of choices and can be coerced.
•Error message keys: required, invalid_choice
Takes extra arguments:

coerce
A function that takes one argument and returns a coerced value. Examples include the built-in int, float, bool and other types. Defaults to an identity function. Note that coercion happens after input validation, so it is possible to coerce to a value not present in choices.

empty_value
The value to use to represent “empty.” Defaults to the empty string; None is another common choice here. Note that this value will not be coerced by the function given in the coerce argument, so choose it accordingly.



DateField
class DateField(**kwargs)
•Default widget: DateInput
•Empty value: None
•Normalizes to: A Python datetime.date object.
•Validates that the given value is either a datetime.date, datetime.datetime or string formatted in a particular date format.
•Error message keys: required, invalid
Takes one optional argument:

input_formats
A list of formats used to attempt to convert a string to a valid datetime.date object.

If no input_formats argument is provided, the default input formats are:

['%Y-%m-%d',      # '2006-10-25'
'%m/%d/%Y',       # '10/25/2006'
'%m/%d/%y']       # '10/25/06'
Additionally, if you specify USE_L10N=False in your settings, the following will also be included in the default input formats:

['%b %d %Y',      # 'Oct 25 2006'
'%b %d, %Y',      # 'Oct 25, 2006'
'%d %b %Y',       # '25 Oct 2006'
'%d %b, %Y',      # '25 Oct, 2006'
'%B %d %Y',       # 'October 25 2006'
'%B %d, %Y',      # 'October 25, 2006'
'%d %B %Y',       # '25 October 2006'
'%d %B, %Y']      # '25 October, 2006'




DateTimeField
class DateTimeField(**kwargs)
•Default widget: DateTimeInput
•Empty value: None
•Normalizes to: A Python datetime.datetime object.
•Validates that the given value is either a datetime.datetime, datetime.date or string formatted in a particular datetime format.
•Error message keys: required, invalid
Takes one optional argument:

input_formats
A list of formats used to attempt to convert a string to a valid datetime.datetime object.

If no input_formats argument is provided, the default input formats are:

['%Y-%m-%d %H:%M:%S',    # '2006-10-25 14:30:59'
'%Y-%m-%d %H:%M',        # '2006-10-25 14:30'
'%Y-%m-%d',              # '2006-10-25'
'%m/%d/%Y %H:%M:%S',     # '10/25/2006 14:30:59'
'%m/%d/%Y %H:%M',        # '10/25/2006 14:30'
'%m/%d/%Y',              # '10/25/2006'
'%m/%d/%y %H:%M:%S',     # '10/25/06 14:30:59'
'%m/%d/%y %H:%M',        # '10/25/06 14:30'
'%m/%d/%y']              # '10/25/06'



DecimalField
class DecimalField(**kwargs)
•Default widget: NumberInput when Field.localize is False, else TextInput.
•Empty value: None
•Normalizes to: A Python decimal.
•Validates that the given value is a decimal. Leading and trailing whitespace is ignored.
•Error message keys: required, invalid, max_value, min_value, max_digits, max_decimal_places, max_whole_digits
The max_value and min_value error messages may contain %(limit_value)s, which will be substituted by the appropriate limit.

Similarly, the max_digits, max_decimal_places and max_whole_digits error messages may contain %(max)s.

Takes four optional arguments:

max_value
min_value
These control the range of values permitted in the field, and should be given as decimal.Decimal values.

max_digits
The maximum number of digits (those before the decimal point plus those after the decimal point, with leading zeros stripped) permitted in the value.

decimal_places
The maximum number of decimal places permitted.



EmailField
class EmailField(**kwargs)
•Default widget: EmailInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value is a valid email address, using a moderately complex regular expression.
•Error message keys: required, invalid
Has two optional arguments for validation, max_length and min_length. If provided, these arguments ensure that the string is at most or at least the given length.



FileField
class FileField(**kwargs)
•Default widget: ClearableFileInput
•Empty value: None
•Normalizes to: An UploadedFile object that wraps the file content and file name into a single object.
•Can validate that non-empty file data has been bound to the form.
•Error message keys: required, invalid, missing, empty, max_length
Has two optional arguments for validation, max_length and allow_empty_file. If provided, these ensure that the file name is at most the given length, and that validation will succeed even if the file content is empty.

When you use a FileField in a form, you must also remember to bind the file data to the form.

The max_length error refers to the length of the filename. In the error message for that key, %(max)d will be replaced with the maximum filename length and %(length)d will be replaced with the current filename length.



FilePathField
class FilePathField(**kwargs)
•Default widget: Select
•Empty value: None
•Normalizes to: A unicode object
•Validates that the selected choice exists in the list of choices.
•Error message keys: required, invalid_choice
The field allows choosing from files inside a certain directory. It takes three extra arguments; only path is required:

path
The absolute path to the directory whose contents you want listed. This directory must exist.

recursive
If False (the default) only the direct contents of path will be offered as choices. If True, the directory will be descended into recursively and all descendants will be listed as choices.

match
A regular expression pattern; only files with names matching this expression will be allowed as choices.

allow_files
Optional. Either True or False. Default is True. Specifies whether files in the specified location should be included. Either this or allow_folders must be True.

allow_folders
Optional. Either True or False. Default is False. Specifies whether folders in the specified location should be included. Either this or allow_files must be True.




FloatField
class FloatField(**kwargs)
•Default widget: NumberInput when Field.localize is False, else TextInput.
•Empty value: None
•Normalizes to: A Python float.
•Validates that the given value is an float. Leading and trailing whitespace is allowed, as in Python’s float() function.
•Error message keys: required, invalid, max_value, min_value
Takes two optional arguments for validation, max_value and min_value. These control the range of values permitted in the field.



ImageField
class ImageField(**kwargs)
•Default widget: ClearableFileInput
•Empty value: None
•Normalizes to: An UploadedFile object that wraps the file content and file name into a single object.
•Validates that file data has been bound to the form, and that the file is of an image format understood by Pillow/PIL.
•Error message keys: required, invalid, missing, empty, invalid_image
Using an ImageField requires that either Pillow (recommended) or the Python Imaging Library (PIL) are installed and supports the image formats you use. If you encounter a corrupt image error when you upload an image, it usually means either Pillow or PIL doesn’t understand its format. To fix this, install the appropriate library and reinstall Pillow or PIL.

When you use an ImageField on a form, you must also remember to bind the file data to the form.



IntegerField
class IntegerField(**kwargs)
•Default widget: NumberInput when Field.localize is False, else TextInput.
•Empty value: None
•Normalizes to: A Python integer or long integer.
•Validates that the given value is an integer. Leading and trailing whitespace is allowed, as in Python’s int() function.
•Error message keys: required, invalid, max_value, min_value
The max_value and min_value error messages may contain %(limit_value)s, which will be substituted by the appropriate limit.

Takes two optional arguments for validation:

max_value
min_value
These control the range of values permitted in the field.



IPAddressField
class IPAddressField(**kwargs)
Deprecated since version 1.7: 
This field has been deprecated in favor of GenericIPAddressField.

•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value is a valid IPv4 address, using a regular expression.
•Error message keys: required, invalid


GenericIPAddressField
class GenericIPAddressField(**kwargs)
A field containing either an IPv4 or an IPv6 address.

•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object. IPv6 addresses are normalized as described below.
•Validates that the given value is a valid IP address.
•Error message keys: required, invalid
The IPv6 address normalization follows RFC 4291 section 2.2, including using the IPv4 format suggested in paragraph 3 of that section, like ::ffff:192.0.2.0. For example, 2001:0::0:01 would be normalized to 2001::1, and ::ffff:0a0a:0a0a to ::ffff:10.10.10.10. All characters are converted to lowercase.

Takes two optional arguments:

protocol
Limits valid inputs to the specified protocol. Accepted values are both (default), IPv4 or IPv6. Matching is case insensitive.

unpack_ipv4
Unpacks IPv4 mapped addresses like ::ffff:192.0.2.1. If this option is enabled that address would be unpacked to 192.0.2.1. Default is disabled. Can only be used when protocol is set to 'both'.

MultipleChoiceField
•Default widget: SelectMultiple
•Empty value: [] (an empty list)
•Normalizes to: A list of Unicode objects.
•Validates that every value in the given list of values exists in the list of choices.
•Error message keys: required, invalid_choice, invalid_list
The invalid_choice error message may contain %(value)s, which will be replaced with the selected choice.

Takes one extra required argument, choices, as for ChoiceField.


TypedMultipleChoiceField
Just like a MultipleChoiceField, except TypedMultipleChoiceField takes two extra arguments, coerce and empty_value.

•Default widget: SelectMultiple
•Empty value: Whatever you’ve given as empty_value
•Normalizes to: A list of values of the type provided by the coerce argument.
•Validates that the given values exists in the list of choices and can be coerced.
•Error message keys: required, invalid_choice
The invalid_choice error message may contain %(value)s, which will be replaced with the selected choice.

Takes two extra arguments, coerce and empty_value, as for TypedChoiceField.



NullBooleanField
class NullBooleanField(**kwargs)
•Default widget: NullBooleanSelect
•Empty value: None
•Normalizes to: A Python True, False or None value.
•Validates nothing (i.e., it never raises a ValidationError).
RegexField
class RegexField(**kwargs)
•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value matches against a certain regular expression.
•Error message keys: required, invalid
Takes one required argument:

regex
A regular expression specified either as a string or a compiled regular expression object.

Also takes max_length and min_length, which work just as they do for CharField.

The optional argument error_message is also accepted for backwards compatibility. The preferred way to provide an error message is to use the error_messages argument, passing a dictionary with 'invalid' as a key and the error message as the value.



SlugField
class SlugField(**kwargs)
•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value contains only letters, numbers, underscores, and hyphens.
•Error messages: required, invalid
This field is intended for use in representing a model SlugField in forms.



TimeField
class TimeField(**kwargs)
•Default widget: TextInput
•Empty value: None
•Normalizes to: A Python datetime.time object.
•Validates that the given value is either a datetime.time or string formatted in a particular time format.
•Error message keys: required, invalid
Takes one optional argument:

input_formats
A list of formats used to attempt to convert a string to a valid datetime.time object.

If no input_formats argument is provided, the default input formats are:

'%H:%M:%S',     # '14:30:59'
'%H:%M',        # '14:30'
URLField
class URLField(**kwargs)
•Default widget: URLInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value is a valid URL.
•Error message keys: required, invalid
Takes the following optional arguments:

max_length
min_length
These are the same as CharField.max_length and CharField.min_length.




ComboField
class ComboField(**kwargs)
•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: A Unicode object.
•Validates that the given value against each of the fields specified as an argument to the ComboField.
•Error message keys: required, invalid
Takes one extra required argument:

fields
The list of fields that should be used to validate the field’s value (in the order in which they are provided).

from django.forms import ComboField
f = ComboField(fields=[CharField(max_length=20), EmailField()])
f.clean('test@example.com')
u'test@example.com'
f.clean('longemailaddress@example.com')
Traceback (most recent call last):
...
ValidationError: [u'Ensure this value has at most 20 characters (it has 28).']




MultiValueField
class MultiValueField(fields=(), **kwargs)
•Default widget: TextInput
•Empty value: '' (an empty string)
•Normalizes to: the type returned by the compress method of the subclass.
•Validates that the given value against each of the fields specified as an argument to the MultiValueField.
•Error message keys: required, invalid, incomplete
Aggregates the logic of multiple fields that together produce a single value.

This field is abstract and must be subclassed. In contrast with the single-value fields, subclasses of MultiValueField must not implement clean() but instead - implement compress().

Takes one extra required argument:

fields
A tuple of fields whose values are cleaned and subsequently combined into a single value. Each value of the field is cleaned by the corresponding field in fields – the first value is cleaned by the first field, the second value is cleaned by the second field, etc. Once all fields are cleaned, the list of clean values is combined into a single value by compress().

Also takes one extra optional argument:

require_all_fields
Defaults to True, in which case a required validation error will be raised if no value is supplied for any field.

When set to False, the Field.required attribute can be set to False for individual fields to make them optional. If no value is supplied for a required field, an incomplete validation error will be raised.

A default incomplete error message can be defined on the MultiValueField subclass, or different messages can be defined on each individual field. For example:

from django.core.validators import RegexValidator

class PhoneField(MultiValueField):
    def __init__(self, *args, **kwargs):
        # Define one message for all fields.
        error_messages = {
            'incomplete': 'Enter a country calling code and a phone number.',
        }
        # Or define a different message for each field.
        fields = (
            CharField(error_messages={'incomplete': 'Enter a country calling code.'},
                      validators=[RegexValidator(r'^\d+$', 'Enter a valid country calling code.')]),
            CharField(error_messages={'incomplete': 'Enter a phone number.'},
                      validators=[RegexValidator(r'^\d+$', 'Enter a valid phone number.')]),
            CharField(validators=[RegexValidator(r'^\d+$', 'Enter a valid extension.')],
                      required=False),
        )
        super(PhoneField, self).__init__(
            error_messages=error_messages, fields=fields,
            require_all_fields=False, *args, **kwargs)
widget
Must be a subclass of django.forms.MultiWidget. Default value is TextInput, which probably is not very useful in this case.

compress(data_list)
Takes a list of valid values and returns a “compressed” version of those values – in a single value. For example, SplitDateTimeField is a subclass which combines a time field and a date field into a datetime object.

This method must be implemented in the subclasses.



SplitDateTimeField
class SplitDateTimeField(**kwargs)
•Default widget: SplitDateTimeWidget
•Empty value: None
•Normalizes to: A Python datetime.datetime object.
•Validates that the given value is a datetime.datetime or string formatted in a particular datetime format.
•Error message keys: required, invalid, invalid_date, invalid_time
Takes two optional arguments:

input_date_formats
A list of formats used to attempt to convert a string to a valid datetime.date object.

If no input_date_formats argument is provided, the default input formats for DateField are used.

input_time_formats
A list of formats used to attempt to convert a string to a valid datetime.time object.

If no input_time_formats argument is provided, the default input formats for TimeField are used.




Fields which handle relationships
Two fields are available for representing relationships between models: ModelChoiceField and ModelMultipleChoiceField. Both of these fields require a single queryset parameter that is used to create the choices for the field. Upon form validation, these fields will place either one model object (in the case of ModelChoiceField) or multiple model objects (in the case of ModelMultipleChoiceField) into the cleaned_data dictionary of the form.

For more complex uses, you can specify queryset=None when declaring the form field and then populate the queryset in the form’s __init__() method:

class FooMultipleChoiceForm(forms.Form):
    foo_select = forms.ModelMultipleChoiceField(queryset=None)

    def __init__(self, *args, **kwargs):
        super(FooMultipleChoiceForm, self).__init__(*args, **kwargs)
        self.fields['foo_select'].queryset = ...


ModelChoiceField
class ModelChoiceField(**kwargs)
•Default widget: Select
•Empty value: None
•Normalizes to: A model instance.
•Validates that the given id exists in the queryset.
•Error message keys: required, invalid_choice
Allows the selection of a single model object, suitable for representing a foreign key. Note that the default widget for ModelChoiceField becomes impractical when the number of entries increases. You should avoid using it for more than 100 items.

A single argument is required:

queryset
A QuerySet of model objects from which the choices for the field will be derived, and which will be used to validate the user’s selection.

ModelChoiceField also takes two optional arguments:

empty_label
By default the <select> widget used by ModelChoiceField will have an empty choice at the top of the list. You can change the text of this label (which is "---------" by default) with the empty_label attribute, or you can disable the empty label entirely by setting empty_label to None:

# A custom empty label
field1 = forms.ModelChoiceField(queryset=..., empty_label="(Nothing)")

# No empty label
field2 = forms.ModelChoiceField(queryset=..., empty_label=None)
Note that if a ModelChoiceField is required and has a default initial value, no empty choice is created (regardless of the value of empty_label).

to_field_name
This optional argument is used to specify the field to use as the value of the choices in the field’s widget. Be sure it’s a unique field for the model, otherwise the selected value could match more than one object. By default it is set to None, in which case the primary key of each object will be used. For example:

# No custom to_field_name
field1 = forms.ModelChoiceField(queryset=...)
would yield:

<select id="id_field1" name="field1">
<option value="obj1.pk">Object1</option>
<option value="obj2.pk">Object2</option>
...
</select>
and:

# to_field_name provided
field2 = forms.ModelChoiceField(queryset=..., to_field_name="name")
would yield:

<select id="id_field2" name="field2">
<option value="obj1.name">Object1</option>
<option value="obj2.name">Object2</option>
...
</select>
The __str__ (__unicode__ on Python 2) method of the model will be called to generate string representations of the objects for use in the field’s choices; to provide customized representations, subclass ModelChoiceField and override label_from_instance. This method will receive a model object, and should return a string suitable for representing it. For example:

from django.forms import ModelChoiceField

class MyModelChoiceField(ModelChoiceField):
    def label_from_instance(self, obj):
        return "My Object #%i" % obj.id


ModelMultipleChoiceField
class ModelMultipleChoiceField(**kwargs)
•Default widget: SelectMultiple
•Empty value: An empty QuerySet (self.queryset.none())
•Normalizes to: A QuerySet of model instances.
•Validates that every id in the given list of values exists in the queryset.
•Error message keys: required, list, invalid_choice, invalid_pk_value
Changed in Django 1.6: 
The invalid_choice message may contain %(value)s and the invalid_pk_value message may contain %(pk)s, which will be substituted by the appropriate values.

Allows the selection of one or more model objects, suitable for representing a many-to-many relation. As with ModelChoiceField, you can use label_from_instance to customize the object representations, and queryset is a required parameter:

queryset
A QuerySet of model objects from which the choices for the field will be derived, and which will be used to validate the user’s selection.






















Widgets
Form fields deal with the logic of input validation and are used directly in templates. Widgets deal with rendering of HTML form input elements on the web page and extraction of raw submitted data. However, widgets do need to be assigned to form fields.

Specifying widgets
Whenever you specify a field on a form, Django will use a default widget that is appropriate to the type of data that is to be displayed. To find which widget is used on which field, see the documentation about Built-in Field classes.

However, if you want to use a different widget for a field, you can just use the widget argument on the field definition. For example:

from django import forms

class CommentForm(forms.Form):
    name = forms.CharField()
    url = forms.URLField()
    comment = forms.CharField(widget=forms.Textarea)
This would specify a form with a comment that uses a larger Textarea widget, rather than the default TextInput widget.



Setting arguments for widgets
Many widgets have optional extra arguments; they can be set when defining the widget on the field. In the following example, the years attribute is set for a SelectDateWidget:

from django import forms
from django.forms.extras.widgets import SelectDateWidget

BIRTH_YEAR_CHOICES = ('1980', '1981', '1982')
FAVORITE_COLORS_CHOICES = (('blue', 'Blue'),
                            ('green', 'Green'),
                            ('black', 'Black'))

class SimpleForm(forms.Form):
    birth_year = forms.DateField(widget=SelectDateWidget(years=BIRTH_YEAR_CHOICES))
    favorite_colors = forms.MultipleChoiceField(required=False,
    widget=forms.CheckboxSelectMultiple, choices=FAVORITE_COLORS_CHOICES)



Widgets inheriting from the Select widget
Widgets inheriting from the Select widget deal with choices. They present the user with a list of options to choose from. The different widgets present this choice differently; the Select widget itself uses a <select> HTML list representation, while RadioSelect uses radio buttons.

Select widgets are used by default on ChoiceField fields. The choices displayed on the widget are inherited from the ChoiceField and changing ChoiceField.choices will update Select.choices. For example:

from django import forms
CHOICES = (('1', 'First',), ('2', 'Second',))
choice_field = forms.ChoiceField(widget=forms.RadioSelect, choices=CHOICES)
choice_field.choices
[('1', 'First'), ('2', 'Second')]
choice_field.widget.choices
[('1', 'First'), ('2', 'Second')]
choice_field.widget.choices = ()
choice_field.choices = (('1', 'First and only',),)
choice_field.widget.choices
[('1', 'First and only')]
Widgets which offer a choices attribute can however be used with fields which are not based on choice – such as a CharField – but it is recommended to use a ChoiceField-based field when the choices are inherent to the model and not just the representational widget.



Customizing widget instances
When Django renders a widget as HTML, it only renders very minimal markup - Django doesn’t add class names, or any other widget-specific attributes. This means, for example, that all TextInput widgets will appear the same on your Web pages.

There are two ways to customize widgets: per widget instance and per widget class.



Styling widget instances

from django import forms

class CommentForm(forms.Form):
    name = forms.CharField()
    url = forms.URLField()
    comment = forms.CharField()
This form will include three default TextInput widgets, with default rendering – no CSS class, no extra attributes. This means that the input boxes provided for each widget will be rendered exactly the same:

f = CommentForm(auto_id=False)
f.as_table()
<tr><th>Name:</th><td><input type="text" name="name" /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url"/></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" /></td></tr>

On a real Web page, you probably don’t want every widget to look the same. You might want a larger input element for the comment, and you might want the ‘name’ widget to have some special CSS class. It is also possible to specify the ‘type’ attribute to take advantage of the new HTML5 input types. To do this, you use the Widget.attrs argument when creating the widget:

class CommentForm(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'special'}))
    url = forms.URLField()
    comment = forms.CharField(widget=forms.TextInput(attrs={'size': '40'}))
Django will then include the extra attributes in the rendered output:

f = CommentForm(auto_id=False)
f.as_table()
<tr><th>Name:</th><td><input type="text" name="name" class="special"/></td></tr>
<tr><th>Url:</th><td><input type="url" name="url"/></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" size="40"/></td></tr>



Styling widget classes
With widgets, it is possible to add assets (css and javascript) and more deeply customize their appearance and behavior.

In a nutshell, you will need to subclass the widget and either define a “Media” inner class or create a “media” property.




Base Widget classes
Base widget classes Widget and MultiWidget are subclassed by all the built-in widgets and may serve as a foundation for custom widgets.

class Widget(attrs=None)
This abstract class cannot be rendered, but provides the basic attribute attrs. You may also implement or override the render() method on custom widgets.

attrs
A dictionary containing HTML attributes to be set on the rendered widget.

from django import forms
name = forms.TextInput(attrs={'size': 10, 'title': 'Your name',})
name.render('name', 'A name')
u'<input title="Your name" type="text" name="name" value="A name" size="10" />'


render(name, value, attrs=None)
Returns HTML for the widget, as a Unicode string. This method must be implemented by the subclass, otherwise NotImplementedError will be raised.

The ‘value’ given is not guaranteed to be valid input, therefore subclass implementations should program defensively.


value_from_datadict(data, files, name)
Given a dictionary of data and this widget’s name, returns the value of this widget. files may contain data coming from request.FILES. Returns None if a value wasn’t provided. Note also that value_from_datadict may be called more than once during handling of form data, so if you customize it and add expensive processing, you should implement some caching mechanism yourself.



class MultiWidget(widgets, attrs=None)
A widget that is composed of multiple widgets. MultiWidget works hand in hand with the MultiValueField.

MultiWidget has one required argument:

widgets
An iterable containing the widgets needed.

And one required method:

decompress(value)
This method takes a single “compressed” value from the field and returns a list of “decompressed” values. The input value can be assumed valid, but not necessarily non-empty.

This method must be implemented by the subclass, and since the value may be empty, the implementation must be defensive.

The rationale behind “decompression” is that it is necessary to “split” the combined value of the form field into the values for each widget.

An example of this is how SplitDateTimeWidget turns a datetime value into a list with date and time split into two separate values:

from django.forms import MultiWidget

class SplitDateTimeWidget(MultiWidget):

    # ...

    def decompress(self, value):
        if value:
            return [value.date(), value.time().replace(microsecond=0)]
        return [None, None]

Note that MultiValueField has a complementary method compress() with the opposite responsibility - to combine cleaned values of all member fields into one.



Other methods that may be useful to override include:

render(name, value, attrs=None)
Argument value is handled differently in this method from the subclasses of Widget because it has to figure out how to split a single value for display in multiple widgets.

The value argument used when rendering can be one of two things:

•A list.
•A single value (e.g., a string) that is the “compressed” representation of a list of values.
If value is a list, the output of render() will be a concatenation of rendered child widgets. If value is not a list, it will first be processed by the method decompress() to create the list and then rendered.

When render() executes its HTML rendering, each value in the list is rendered with the corresponding widget – the first value is rendered in the first widget, the second value is rendered in the second widget, etc.

Unlike in the single value widgets, method render() need not be implemented in the subclasses.


format_output(rendered_widgets)
Given a list of rendered widgets (as strings), returns a Unicode string representing the HTML for the whole lot.

This hook allows you to format the HTML design of the widgets any way you’d like.

Here’s an example widget which subclasses MultiWidget to display a date with the day, month, and year in different select boxes. This widget is intended to be used with a DateField rather than a MultiValueField, thus we have implemented value_from_datadict():

from datetime import date
from django.forms import widgets

class DateSelectorWidget(widgets.MultiWidget):
    def __init__(self, attrs=None):
        # create choices for days, months, years
        # example below, the rest snipped for brevity.
        years = [(year, year) for year in (2011, 2012, 2013)]
        _widgets = (
            widgets.Select(attrs=attrs, choices=days),
            widgets.Select(attrs=attrs, choices=months),
            widgets.Select(attrs=attrs, choices=years),
        )
        super(DateSelectorWidget, self).__init__(_widgets, attrs)

    def decompress(self, value):
        if value:
            return [value.day, value.month, value.year]
        return [None, None, None]

    def format_output(self, rendered_widgets):
        return u''.join(rendered_widgets)

    def value_from_datadict(self, data, files, name):
        datelist = [
            widget.value_from_datadict(data, files, name + '_%s' % i)
            for i, widget in enumerate(self.widgets)]
        try:
            D = date(day=int(datelist[0]), month=int(datelist[1]),
                    year=int(datelist[2]))
        except ValueError:
            return ''
        else:
            return str(D)
The constructor creates several Select widgets in a tuple. The super class uses this tuple to setup the widget.

The format_output() method is fairly vanilla here (in fact, it’s the same as what’s been implemented as the default for MultiWidget), but the idea is that you could add custom HTML between the widgets should you wish.

The required method decompress() breaks up a datetime.date value into the day, month, and year values corresponding to each widget. Note how the method handles the case where value is None.

The default implementation of value_from_datadict() returns a list of values corresponding to each Widget. This is appropriate when using a MultiWidget with a MultiValueField, but since we want to use this widget with a DateField which takes a single value, we have overridden this method to combine the data of all the subwidgets into a datetime.date. The method extracts data from the POST dictionary and constructs and validates the date. If it is valid, we return the string, otherwise, we return an empty string which will cause form.is_valid to return False.




Built-in widgets


Widgets handling input of text
These widgets make use of the HTML elements input and textarea.

TextInput
class TextInput
Text input: <input type="text" ...>


NumberInput
class NumberInput
New in Django 1.6. 
Text input: <input type="number" ...>

Beware that not all browsers support entering localized numbers in number input types. Django itself avoids using them for fields having their localize property to True.


EmailInput
class EmailInput
New in Django 1.6. 
Text input: <input type="email" ...>


URLInput
class URLInput
New in Django 1.6. 
Text input: <input type="url" ...>


PasswordInput
class PasswordInput
Password input: <input type='password' ...>

Takes one optional argument:

render_value
Determines whether the widget will have a value filled in when the form is re-displayed after a validation error (default is False).


HiddenInput
class HiddenInput
Hidden input: <input type='hidden' ...>

Note that there also is a MultipleHiddenInput widget that encapsulates a set of hidden input elements.


DateInput
class DateInput
Date input as a simple text box: <input type='text' ...>

Takes same arguments as TextInput, with one more optional argument:

format
The format in which this field’s initial value will be displayed.

If no format argument is provided, the default format is the first format found in DATE_INPUT_FORMATS and respects Format localization.


DateTimeInput
class DateTimeInput
Date/time input as a simple text box: <input type='text' ...>

Takes same arguments as TextInput, with one more optional argument:

format
The format in which this field’s initial value will be displayed.

If no format argument is provided, the default format is the first format found in DATETIME_INPUT_FORMATS and respects Format localization.


TimeInput
class TimeInput
Time input as a simple text box: <input type='text' ...>

Takes same arguments as TextInput, with one more optional argument:

format
The format in which this field’s initial value will be displayed.

If no format argument is provided, the default format is the first format found in TIME_INPUT_FORMATS and respects Format localization.


Textarea
class Textarea
Text area: <textarea>...</textarea>



Selector and checkbox widgets

CheckboxInput
class CheckboxInput
Checkbox: <input type='checkbox' ...>

Takes one optional argument:

check_test
A callable that takes the value of the CheckboxInput and returns True if the checkbox should be checked for that value.


Select
class Select
Select widget: <select><option ...>...</select>

choices
This attribute is optional when the form field does not have a choices attribute. If it does, it will override anything you set here when the attribute is updated on the Field.


NullBooleanSelect
class NullBooleanSelect
Select widget with options ‘Unknown’, ‘Yes’ and ‘No’


SelectMultiple
class SelectMultiple
Similar to Select, but allows multiple selection: <select multiple='multiple'>...</select>


RadioSelect
class RadioSelect
Similar to Select, but rendered as a list of radio buttons within <li> tags:

<ul>
  <li><input type='radio' name='...'></li>
  ...
</ul>
For more granular control over the generated markup, you can loop over the radio buttons in the template. Assuming a form myform with a field beatles that uses a RadioSelect as its widget:

{% for radio in myform.beatles %}
<div class="myradio">
    {{ radio }}
</div>
{% endfor %}
This would generate the following HTML:

<div class="myradio">
    <label for="id_beatles_0"><input id="id_beatles_0" name="beatles" type="radio" value="john" /> John</label>
</div>
<div class="myradio">
    <label for="id_beatles_1"><input id="id_beatles_1" name="beatles" type="radio" value="paul" /> Paul</label>
</div>
<div class="myradio">
    <label for="id_beatles_2"><input id="id_beatles_2" name="beatles" type="radio" value="george" /> George</label>
</div>
<div class="myradio">
    <label for="id_beatles_3"><input id="id_beatles_3" name="beatles" type="radio" value="ringo" /> Ringo</label>
</div>
That included the <label> tags. To get more granular, you can use each radio button’s tag, choice_label and id_for_label attributes. For example, this template...

{% for radio in myform.beatles %}
    <label for="{{ radio.id_for_label }}">
        {{ radio.choice_label }}
        <span class="radio">{{ radio.tag }}</span>
    </label>
{% endfor %}
...will result in the following HTML:

<label for="id_beatles_0">
    John
    <span class="radio"><input id="id_beatles_0" name="beatles" type="radio" value="john" /></span>
</label>

<label for="id_beatles_1">
    Paul
    <span class="radio"><input id="id_beatles_1" name="beatles" type="radio" value="paul" /></span>
</label>

<label for="id_beatles_2">
    George
    <span class="radio"><input id="id_beatles_2" name="beatles" type="radio" value="george" /></span>
</label>

<label for="id_beatles_3">
    Ringo
    <span class="radio"><input id="id_beatles_3" name="beatles" type="radio" value="ringo" /></span>
</label>
If you decide not to loop over the radio buttons – e.g., if your template simply includes {{ myform.beatles }} – they’ll be output in a <ul> with <li> tags, as above.


The outer <ul> container will now receive the id attribute defined on the widget.


When looping over the radio buttons, the label and input tags include for and id attributes, respectively. Each radio button has an id_for_label attribute to output the element’s ID.



CheckboxSelectMultiple
class CheckboxSelectMultiple
Similar to SelectMultiple, but rendered as a list of check buttons:

<ul>
  <li><input type='checkbox' name='...' ></li>
  ...
</ul>
Changed in Django 1.6. 
The outer <ul> container will now receive the id attribute defined on the widget.

Like RadioSelect, you can now loop over the individual checkboxes making up the lists. See the documentation of RadioSelect for more details.


When looping over the checkboxes, the label and input tags include for and id attributes, respectively. Each checkbox has an id_for_label attribute to output the element’s ID.

File upload widgets

FileInput
class FileInput
File upload input: <input type='file' ...>

ClearableFileInput
class ClearableFileInput
File upload input: <input type='file' ...>, with an additional checkbox input to clear the field’s value, if the field is not required and has initial data.



Composite widgets

MultipleHiddenInput
class MultipleHiddenInput
Multiple <input type='hidden' ...> widgets.

A widget that handles multiple hidden widgets for fields that have a list of values.

choices
This attribute is optional when the form field does not have a choices attribute. If it does, it will override anything you set here when the attribute is updated on the Field.


SplitDateTimeWidget
class SplitDateTimeWidget
Wrapper (using MultiWidget) around two widgets: DateInput for the date, and TimeInput for the time.

SplitDateTimeWidget has two optional attributes:

date_format
Similar to DateInput.format

time_format
Similar to TimeInput.format


SplitHiddenDateTimeWidget
class SplitHiddenDateTimeWidget
Similar to SplitDateTimeWidget, but uses HiddenInput for both date and time.


SelectDateWidget
class SelectDateWidget[source]
Wrapper around three Select widgets: one each for month, day, and year. Note that this widget lives in a separate file from the standard widgets.

Takes one optional argument:

years
An optional list/tuple of years to use in the “year” select box. The default is a list containing the current year and the next 9 years.

months
An optional dict of months to use in the “months” select box.

The keys of the dict correspond to the month number (1-indexed) and the values are the displayed months.

MONTHS = {
    1:_('jan'), 2:_('feb'), 3:_('mar'), 4:_('apr'),
    5:_('may'), 6:_('jun'), 7:_('jul'), 8:_('aug'),
    9:_('sep'), 10:_('oct'), 11:_('nov'), 12:_('dec')
}
























Creating forms from models-ModelForm

example:

from django.forms import ModelForm
from myapp.models import Article

# Create the form class.
class ArticleForm(ModelForm):
...     class Meta:
...         model = Article
...         fields = ['pub_date', 'headline', 'content', 'reporter']

# Creating a form to add an article.
form = ArticleForm()

# Creating a form to change an existing article.
article = Article.objects.get(pk=1)
form = ArticleForm(instance=article)



Field types
The generated Form class will have a form field for every model field specified, in the order specified in the fields attribute.

Each model field has a corresponding default form field. For example, a CharField on a model is represented as a CharField on a form. A model ManyToManyField is represented as a MultipleChoiceField. Here is the full list of conversions:

Model field 			Form field 
AutoField 			Not represented in the form 
BigIntegerField 		IntegerField with min_value set to -9223372036854775808 and max_value set to 9223372036854775807. 
BooleanField 			BooleanField 
CharField 			CharField with max_length set to the model field’s max_length 
CommaSeparatedIntegerField 	CharField 
DateField 			DateField 
DateTimeField 			DateTimeField 
DecimalField 			DecimalField 
EmailField 			EmailField 
FileField 			FileField 
FilePathField 			FilePathField 
FloatField 			FloatField 
ForeignKey 			ModelChoiceField 
ImageField 			ImageField 
IntegerField 			IntegerField 
IPAddressField 			IPAddressField 
GenericIPAddressField 		GenericIPAddressField 
ManyToManyField 		ModelMultipleChoiceField (see below) 
NullBooleanField 		NullBooleanField 
PositiveIntegerField 		IntegerField 
PositiveSmallIntegerField 	IntegerField 
SlugField 			SlugField 
SmallIntegerField 		IntegerField 
TextField 			CharField with widget=forms.Textarea 
TimeField 			TimeField 
URLField 			URLField 


the ForeignKey and ManyToManyField model field types are special cases:

•ForeignKey is represented by django.forms.ModelChoiceField, which is a ChoiceField whose choices are a model QuerySet.
•ManyToManyField is represented by django.forms.ModelMultipleChoiceField, which is a MultipleChoiceField whose choices are a model QuerySet.
In addition, each generated form field has attributes set as follows:

•If the model field has blank=True, then required is set to False on the form field. Otherwise, required=True.
•The form field’s label is set to the verbose_name of the model field, with the first character capitalized.
•The form field’s help_text is set to the help_text of the model field.
•If the model field has choices set, then the form field’s widget will be set to Select, with choices coming from the model field’s choices. The choices will normally include the blank choice which is selected by default. If the field is required, this forces the user to make a selection. The blank choice will not be included if the model field has blank=False and an explicit default value (the default value will be initially selected instead).
Finally, note that you can override the form field used for a given model field. See Overriding the default fields below.

Example:

from django.db import models
from django.forms import ModelForm

TITLE_CHOICES = (
    ('MR', 'Mr.'),
    ('MRS', 'Mrs.'),
    ('MS', 'Ms.'),
)

class Author(models.Model):
    name = models.CharField(max_length=100)
    title = models.CharField(max_length=3, choices=TITLE_CHOICES)
    birth_date = models.DateField(blank=True, null=True)

    def __str__(self):              # __unicode__ on Python 2
        return self.name

class Book(models.Model):
    name = models.CharField(max_length=100)
    authors = models.ManyToManyField(Author)

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = ['name', 'title', 'birth_date']

class BookForm(ModelForm):
    class Meta:
        model = Book
        fields = ['name', 'authors']
With these models, the ModelForm subclasses above would be roughly equivalent to this (the only difference being the save() method, which we’ll discuss in a moment.):

from django import forms

class AuthorForm(forms.Form):
    name = forms.CharField(max_length=100)
    title = forms.CharField(max_length=3,
                widget=forms.Select(choices=TITLE_CHOICES))
    birth_date = forms.DateField(required=False)

class BookForm(forms.Form):
    name = forms.CharField(max_length=100)
    authors = forms.ModelMultipleChoiceField(queryset=Author.objects.all())



Validation on a ModelForm
There are two main steps involved in validating a ModelForm:

1.Validating the form
2.Validating the model instance
Just like normal form validation, model form validation is triggered implicitly when calling is_valid() or accessing the errors attribute and explicitly when calling full_clean(), although you will typically not use the latter method in practice.

Model validation (Model.full_clean()) is triggered from within the form validation step, right after the form’s clean() method is called.


The cleaning process modifies the model instance passed to the ModelForm constructor in various ways. For instance, any date fields on the model are converted into actual date objects. Failed validation may leave the underlying model instance in an inconsistent state and therefore it’s not recommended to reuse it.




Overriding the clean() method
You can override the clean() method on a model form to provide additional validation in the same way you can on a normal form.

A model form instance attached to a model object will contain an instance attribute that gives its methods access to that specific model instance.

The ModelForm.clean() method sets a flag that makes the model validation step validate the uniqueness of model fields that are marked as unique, unique_together or unique_for_date|month|year.

If you would like to override the clean() method and maintain this validation, you must call the parent class’s clean() method.




Interaction with model validation
As part of the validation process, ModelForm will call the clean() method of each field on your model that has a corresponding field on your form. If you have excluded any model fields, validation will not be run on those fields. See the form validation documentation for more on how field cleaning and validation work.

The model’s clean() method will be called before any uniqueness checks are made. See Validating objects for more information on the model’s clean() hook.



Considerations regarding model’s error_messages
Error messages defined at the form field level or at the form Meta level always take precedence over the error messages defined at the model field level.

Error messages defined on model fields are only used when the ValidationError is raised during the model validation step and no corresponding error messages are defined at the form level.

You can override the error messages from NON_FIELD_ERRORS raised by model validation by adding the NON_FIELD_ERRORS key to the error_messages dictionary of the ModelForm’s inner Meta class:

from django.forms import ModelForm
from django.core.exceptions import NON_FIELD_ERRORS

class ArticleForm(ModelForm):
    class Meta:
        error_messages = {
            NON_FIELD_ERRORS: {
                'unique_together': "%(model_name)s's %(field_labels)s are not unique.",
            }
        }


The save() method
Every ModelForm also has a save() method. This method creates and saves a database object from the data bound to the form. A subclass of ModelForm can accept an existing model instance as the keyword argument instance; if this is supplied, save() will update that instance. If it’s not supplied, save() will create a new instance of the specified model:

from myapp.models import Article
from myapp.forms import ArticleForm

# Create a form instance from POST data.
f = ArticleForm(request.POST)

# Save a new Article object from the form's data.
new_article = f.save()

# Create a form to edit an existing Article, but use
# POST data to populate the form.
a = Article.objects.get(pk=1)
f = ArticleForm(request.POST, instance=a)
f.save()
Note that if the form hasn’t been validated, calling save() will do so by checking form.errors. A ValueError will be raised if the data in the form doesn’t validate – i.e., if form.errors evaluates to True.

This save() method accepts an optional commit keyword argument, which accepts either True or False. If you call save() with commit=False, then it will return an object that hasn’t yet been saved to the database. In this case, it’s up to you to call save() on the resulting model instance. This is useful if you want to do custom processing on the object before saving it, or if you want to use one of the specialized model saving options. commit is True by default.

Another side effect of using commit=False is seen when your model has a many-to-many relation with another model. If your model has a many-to-many relation and you specify commit=False when you save a form, Django cannot immediately save the form data for the many-to-many relation. This is because it isn’t possible to save many-to-many data for an instance until the instance exists in the database.

To work around this problem, every time you save a form using commit=False, Django adds a save_m2m() method to your ModelForm subclass. After you’ve manually saved the instance produced by the form, you can invoke save_m2m() to save the many-to-many form data. For example:

# Create a form instance with POST data.
f = AuthorForm(request.POST)

# Create, but don't save the new author instance.
new_author = f.save(commit=False)

# Modify the author in some way.
new_author.some_field = 'some_value'

# Save the new instance.
new_author.save()

# Now, save the many-to-many data for the form.
f.save_m2m()
Calling save_m2m() is only required if you use save(commit=False). When you use a simple save() on a form, all data – including many-to-many data – is saved without the need for any additional method calls. For example:

# Create a form instance with POST data.
a = Author()
f = AuthorForm(request.POST, instance=a)

# Create and save the new author instance. There's no need to do anything else.
new_author = f.save()
Other than the save() and save_m2m() methods, a ModelForm works exactly the same way as any other forms form. For example, the is_valid() method is used to check for validity, the is_multipart() method is used to determine whether a form requires multipart file upload (and hence whether request.FILES must be passed to the form), etc. See Binding uploaded files to a form for more information.




Selecting the fields to use
It is strongly recommended that you explicitly set all fields that should be edited in the form using the fields attribute. Failure to do so can easily lead to security problems when a form unexpectedly allows a user to set certain fields, especially when new fields are added to a model. Depending on how the form is rendered, the problem may not even be visible on the web page.

The alternative approach would be to include all fields automatically, or blacklist only some. This fundamental approach is known to be much less secure and has led to serious exploits on major websites (e.g. GitHub).

There are, however, two shortcuts available for cases where you can guarantee these security concerns do not apply to you:

1.Set the fields attribute to the special value '__all__' to indicate that all fields in the model should be used. For example:

from django.forms import ModelForm

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = '__all__'
2.Set the exclude attribute of the ModelForm’s inner Meta class to a list of fields to be excluded from the form.

For example:

class PartialAuthorForm(ModelForm):
    class Meta:
        model = Author
        exclude = ['title']
Since the Author model has the 3 fields name, title and birth_date, this will result in the fields name and birth_date being present on the form.

If either of these are used, the order the fields appear in the form will be the order the fields are defined in the model, with ManyToManyField instances appearing last.

In addition, Django applies the following rule: if you set editable=False on the model field, any form created from the model via ModelForm will not include that field.

Any fields not included in a form by the above logic will not be set by the form’s save() method. Also, if you manually add the excluded fields back to the form, they will not be initialized from the model instance.

Django will prevent any attempt to save an incomplete model, so if the model does not allow the missing fields to be empty, and does not provide a default value for the missing fields, any attempt to save() a ModelForm with missing fields will fail. To avoid this failure, you must instantiate your model with initial values for the missing, but required fields:

author = Author(title='Mr')
form = PartialAuthorForm(request.POST, instance=author)
form.save()

Alternatively, you can use save(commit=False) and manually set any extra required fields:

form = PartialAuthorForm(request.POST)
author = form.save(commit=False)
author.title = 'Mr'
author.save()
See the section on saving forms for more details on using save(commit=False).





Overriding the default fields

To specify a custom widget for a field, use the widgets attribute of the inner Meta class. This should be a dictionary mapping field names to widget classes or instances.

For example, if you want the CharField for the name attribute of Author to be represented by a <textarea> instead of its default <input type="text">, you can override the field’s widget:

from django.forms import ModelForm, Textarea
from myapp.models import Author

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = ('name', 'title', 'birth_date')
        widgets = {
            'name': Textarea(attrs={'cols': 80, 'rows': 20}),
        }
The widgets dictionary accepts either widget instances (e.g., Textarea(...)) or classes (e.g., Textarea).

The labels, help_texts and error_messages options were added.

Similarly, you can specify the labels, help_texts and error_messages attributes of the inner Meta class if you want to further customize a field.

For example if you wanted to customize the wording of all user facing strings for the name field:

from django.utils.translation import ugettext_lazy as _

class AuthorForm(ModelForm):
    class Meta:
        model = Author
        fields = ('name', 'title', 'birth_date')
        labels = {
            'name': _('Writer'),
        }
        help_texts = {
            'name': _('Some useful help text.'),
        }
        error_messages = {
            'name': {
                'max_length': _("This writer's name is too long."),
            },
        }
Finally, if you want complete control over of a field – including its type, validators, etc. – you can do this by declaratively specifying fields like you would in a regular Form.

For example, if you wanted to use MySlugFormField for the slug field, you could do the following:

from django.forms import ModelForm
from myapp.models import Article

class ArticleForm(ModelForm):
    slug = MySlugFormField()

    class Meta:
        model = Article
        fields = ['pub_date', 'headline', 'content', 'reporter', 'slug']
If you want to specify a field’s validators, you can do so by defining the field declaratively and setting its validators parameter:

from django.forms import ModelForm, CharField
from myapp.models import Article

class ArticleForm(ModelForm):
    slug = CharField(validators=[validate_slug])

    class Meta:
        model = Article
        fields = ['pub_date', 'headline', 'content', 'reporter', 'slug']
Note

When you explicitly instantiate a form field like this, it is important to understand how ModelForm and regular Form are related.

ModelForm is a regular Form which can automatically generate certain fields. The fields that are automatically generated depend on the content of the Meta class and on which fields have already been defined declaratively. Basically, ModelForm will only generate fields that are missing from the form, or in other words, fields that weren’t defined declaratively.

Fields defined declaratively are left as-is, therefore any customizations made to Meta attributes such as widgets, labels, help_texts, or error_messages are ignored; these only apply to fields that are generated automatically.

Similarly, fields defined declaratively do not draw their attributes like max_length or required from the corresponding model. If you want to maintain the behavior specified in the model, you must set the relevant arguments explicitly when declaring the form field.

For example, if the Article model looks like this:

class Article(models.Model):
    headline = models.CharField(max_length=200, null=True, blank=True,
                                help_text="Use puns liberally")
    content = models.TextField()
and you want to do some custom validation for headline, while keeping the blank and help_text values as specified, you might define ArticleForm like this:

class ArticleForm(ModelForm):
    headline = MyFormField(max_length=200, required=False,
                           help_text="Use puns liberally")

    class Meta:
        model = Article
        fields = ['headline', 'content']
You must ensure that the type of the form field can be used to set the contents of the corresponding model field. When they are not compatible, you will get a ValueError as no implicit conversion takes place.

See the form field documentation for more information on fields and their arguments.



Enabling localization of fields
By default, the fields in a ModelForm will not localize their data. To enable localization for fields, you can use the localized_fields attribute on the Meta class.

from django.forms import ModelForm
from myapp.models import Author
class AuthorForm(ModelForm):
...     class Meta:
...         model = Author
...         localized_fields = ('birth_date',)
If localized_fields is set to the special value '__all__', all fields will be localized.




Form inheritance
As with basic forms, you can extend and reuse ModelForms by inheriting them. This is useful if you need to declare extra fields or extra methods on a parent class for use in a number of forms derived from models. For example, using the previous ArticleForm class:

class EnhancedArticleForm(ArticleForm):
...     def clean_pub_date(self):
...         ...
This creates a form that behaves identically to ArticleForm, except there’s some extra validation and cleaning for the pub_date field.

You can also subclass the parent’s Meta inner class if you want to change the Meta.fields or Meta.excludes lists:

class RestrictedArticleForm(EnhancedArticleForm):
...     class Meta(ArticleForm.Meta):
...         exclude = ('body',)
This adds the extra method from the EnhancedArticleForm and modifies the original ArticleForm.Meta to remove one field.

There are a couple of things to note, however.

•Normal Python name resolution rules apply. If you have multiple base classes that declare a Meta inner class, only the first one will be used. This means the child’s Meta, if it exists, otherwise the Meta of the first parent, etc.

•It’s possible to inherit from both Form and ModelForm simultaneously, however, you must ensure that ModelForm appears first in the MRO. This is because these classes rely on different metaclasses and a class can only have one metaclass.
•It’s possible to declaratively remove a Field inherited from a parent class by setting the name to be None on the subclass.

You can only use this technique to opt out from a field defined declaratively by a parent class; it won’t prevent the ModelForm metaclass from generating a default field. To opt-out from default fields, see Controlling which fields are used with fields and exclude.




Providing initial values
As with regular forms, it’s possible to specify initial data for forms by specifying an initial parameter when instantiating the form. Initial values provided this way will override both initial values from the form field and values from an attached model instance. For example:

article = Article.objects.get(pk=1)
article.headline
'My headline'
form = ArticleForm(initial={'headline': 'Initial headline'}, instance=article)
form['headline'].value()
'Initial headline'



ModelForm factory function
You can create forms from a given model using the standalone function modelform_factory(), instead of using a class definition. This may be more convenient if you do not have many customizations to make:

from django.forms.models import modelform_factory
from myapp.models import Book
BookForm = modelform_factory(Book, fields=("author", "title"))
This can also be used to make simple modifications to existing forms, for example by specifying the widgets to be used for a given field:

from django.forms import Textarea
Form = modelform_factory(Book, form=BookForm,
...                          widgets={"title": Textarea()})
The fields to include can be specified using the fields and exclude keyword arguments, or the corresponding attributes on the ModelForm inner Meta class. Please see the ModelForm Selecting the fields to use documentation.

... or enable localization for specific fields:

Form = modelform_factory(Author, form=AuthorForm, localized_fields=("birth_date",))



Model formsets
class models.BaseModelFormSet
Like regular formsets, Django provides a couple of enhanced formset classes that make it easy to work with Django models. Let’s reuse the Author model from above:

from django.forms.models import modelformset_factory
from myapp.models import Author
AuthorFormSet = modelformset_factory(Author)
This will create a formset that is capable of working with the data associated with the Author model. It works just like a regular formset:

formset = AuthorFormSet()
print(formset)
<input type="hidden" name="form-TOTAL_FORMS" value="1" id="id_form-TOTAL_FORMS" /><input type="hidden" name="form-INITIAL_FORMS" value="0" id="id_form-INITIAL_FORMS" /><input type="hidden" name="form-MAX_NUM_FORMS" id="id_form-MAX_NUM_FORMS" />
<tr><th><label for="id_form-0-name">Name:</label></th><td><input id="id_form-0-name" type="text" name="form-0-name" maxlength="100" /></td></tr>
<tr><th><label for="id_form-0-title">Title:</label></th><td><select name="form-0-title" id="id_form-0-title">
<option value="" selected="selected">---------</option>
<option value="MR">Mr.</option>
<option value="MRS">Mrs.</option>
<option value="MS">Ms.</option>
</select></td></tr>
<tr><th><label for="id_form-0-birth_date">Birth date:</label></th><td><input type="text" name="form-0-birth_date" id="id_form-0-birth_date" /><input type="hidden" name="form-0-id" id="id_form-0-id" /></td></tr>
Note

modelformset_factory() uses formset_factory() to generate formsets. This means that a model formset is just an extension of a basic formset that knows how to interact with a particular model.







Changing the queryset
By default, when you create a formset from a model, the formset will use a queryset that includes all objects in the model (e.g., Author.objects.all()). You can override this behavior by using the queryset argument:

formset = AuthorFormSet(queryset=Author.objects.filter(name__startswith='O'))
Alternatively, you can create a subclass that sets self.queryset in __init__:

from django.forms.models import BaseModelFormSet
from myapp.models import Author

class BaseAuthorFormSet(BaseModelFormSet):
    def __init__(self, *args, **kwargs):
        super(BaseAuthorFormSet, self).__init__(*args, **kwargs)
        self.queryset = Author.objects.filter(name__startswith='O')
Then, pass your BaseAuthorFormSet class to the factory function:

AuthorFormSet = modelformset_factory(Author, formset=BaseAuthorFormSet)
If you want to return a formset that doesn’t include any pre-existing instances of the model, you can specify an empty QuerySet:

AuthorFormSet(queryset=Author.objects.none())











Changing the form
By default, when you use modelformset_factory, a model form will be created using modelform_factory(). Often, it can be useful to specify a custom model form. For example, you can create a custom model form that has custom validation:

class AuthorForm(forms.ModelForm):
    class Meta:
        model = Author
        fields = ('name', 'title')

    def clean_name(self):
        # custom validation for the name field
        ...
Then, pass your model form to the factory function:

AuthorFormSet = modelformset_factory(Author, form=AuthorForm)
It is not always necessary to define a custom model form. The modelformset_factory function has several arguments which are passed through to modelform_factory, which are described below.

Controlling which fields are used with fields and exclude
By default, a model formset uses all fields in the model that are not marked with editable=False. However, this can be overridden at the formset level:

AuthorFormSet = modelformset_factory(Author, fields=('name', 'title'))
Using fields restricts the formset to use only the given fields. Alternatively, you can take an “opt-out” approach, specifying which fields to exclude:

AuthorFormSet = modelformset_factory(Author, exclude=('birth_date',))


Specifying widgets to use in the form with widgets
Using the widgets parameter, you can specify a dictionary of values to customize the ModelForm’s widget class for a particular field. This works the same way as the widgets dictionary on the inner Meta class of a ModelForm works:

AuthorFormSet = modelformset_factory(
...     Author, widgets={'name': Textarea(attrs={'cols': 80, 'rows': 20})
Enabling localization for fields with localized_fields
New in Django 1.6. 
Using the localized_fields parameter, you can enable localization for fields in the form.

AuthorFormSet = modelformset_factory(
...     Author, localized_fields=('value',))
If localized_fields is set to the special value '__all__', all fields will be localized.



Providing initial values
As with regular formsets, it’s possible to specify initial data for forms in the formset by specifying an initial parameter when instantiating the model formset class returned by modelformset_factory(). However, with model formsets, the initial values only apply to extra forms, those that aren’t attached to an existing model instance. If the extra forms with initial data aren’t changed by the user, they won’t be validated or saved.



Saving objects in the formset
As with a ModelForm, you can save the data as a model object. This is done with the formset’s save() method:

# Create a formset instance with POST data.
formset = AuthorFormSet(request.POST)

# Assuming all is valid, save the data.
instances = formset.save()
The save() method returns the instances that have been saved to the database. If a given instance’s data didn’t change in the bound data, the instance won’t be saved to the database and won’t be included in the return value (instances, in the above example).

When fields are missing from the form (for example because they have been excluded), these fields will not be set by the save() method. You can find more information about this restriction, which also holds for regular ModelForms, in Selecting the fields to use.

Pass commit=False to return the unsaved model instances:

# don't save to the database
instances = formset.save(commit=False)
for instance in instances:
...     # do something with instance
...     instance.save()
This gives you the ability to attach data to the instances before saving them to the database. If your formset contains a ManyToManyField, you’ll also need to call formset.save_m2m() to ensure the many-to-many relationships are saved properly.

After calling save(), your model formset will have three new attributes containing the formset’s changes:

models.BaseModelFormSet.changed_objects
models.BaseModelFormSet.deleted_objects
models.BaseModelFormSet.new_objects



Limiting the number of editable objects
As with regular formsets, you can use the max_num and extra parameters to modelformset_factory() to limit the number of extra forms displayed.

max_num does not prevent existing objects from being displayed:

Author.objects.order_by('name')
[<Author: Charles Baudelaire>, <Author: Paul Verlaine>, <Author: Walt Whitman>]

AuthorFormSet = modelformset_factory(Author, max_num=1)
formset = AuthorFormSet(queryset=Author.objects.order_by('name'))
[x.name for x in formset.get_queryset()]
[u'Charles Baudelaire', u'Paul Verlaine', u'Walt Whitman']
If the value of max_num is greater than the number of existing related objects, up to extra additional blank forms will be added to the formset, so long as the total number of forms does not exceed max_num:

AuthorFormSet = modelformset_factory(Author, max_num=4, extra=2)
formset = AuthorFormSet(queryset=Author.objects.order_by('name'))
for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-name">Name:</label></th><td><input id="id_form-0-name" type="text" name="form-0-name" value="Charles Baudelaire" maxlength="100" /><input type="hidden" name="form-0-id" value="1" id="id_form-0-id" /></td></tr>
<tr><th><label for="id_form-1-name">Name:</label></th><td><input id="id_form-1-name" type="text" name="form-1-name" value="Paul Verlaine" maxlength="100" /><input type="hidden" name="form-1-id" value="3" id="id_form-1-id" /></td></tr>
<tr><th><label for="id_form-2-name">Name:</label></th><td><input id="id_form-2-name" type="text" name="form-2-name" value="Walt Whitman" maxlength="100" /><input type="hidden" name="form-2-id" value="2" id="id_form-2-id" /></td></tr>
<tr><th><label for="id_form-3-name">Name:</label></th><td><input id="id_form-3-name" type="text" name="form-3-name" maxlength="100" /><input type="hidden" name="form-3-id" id="id_form-3-id" /></td></tr>
A max_num value of None (the default) puts a high limit on the number of forms displayed (1000). In practice this is equivalent to no limit.



Using a model formset in a view
Model formsets are very similar to formsets. Let’s say we want to present a formset to edit Author model instances:

from django.forms.models import modelformset_factory
from django.shortcuts import render_to_response
from myapp.models import Author

def manage_authors(request):
    AuthorFormSet = modelformset_factory(Author)
    if request.method == 'POST':
        formset = AuthorFormSet(request.POST, request.FILES)
        if formset.is_valid():
            formset.save()
            # do something.
    else:
        formset = AuthorFormSet()
    return render_to_response("manage_authors.html", {
        "formset": formset,
    })
As you can see, the view logic of a model formset isn’t drastically different than that of a “normal” formset. The only difference is that we call formset.save() to save the data into the database. (This was described above, in Saving objects in the formset.)



Overriding clean() on a ModelFormSet
Just like with ModelForms, by default the clean() method of a ModelFormSet will validate that none of the items in the formset violate the unique constraints on your model (either unique, unique_together or unique_for_date|month|year). If you want to override the clean() method on a ModelFormSet and maintain this validation, you must call the parent class’s clean method:

from django.forms.models import BaseModelFormSet

class MyModelFormSet(BaseModelFormSet):
    def clean(self):
        super(MyModelFormSet, self).clean()
        # example custom validation across forms in the formset
        for form in self.forms:
            # your custom formset validation
            ...
Also note that by the time you reach this step, individual model instances have already been created for each Form. Modifying a value in form.cleaned_data is not sufficient to affect the saved value. If you wish to modify a value in ModelFormSet.clean() you must modify form.instance:

from django.forms.models import BaseModelFormSet

class MyModelFormSet(BaseModelFormSet):
    def clean(self):
        super(MyModelFormSet, self).clean()

        for form in self.forms:
            name = form.cleaned_data['name'].upper()
            form.cleaned_data['name'] = name
            # update the instance value.
            form.instance.name = name


Using a custom queryset
As stated earlier, you can override the default queryset used by the model formset:

from django.forms.models import modelformset_factory
from django.shortcuts import render_to_response
from myapp.models import Author

def manage_authors(request):
    AuthorFormSet = modelformset_factory(Author)
    if request.method == "POST":
        formset = AuthorFormSet(request.POST, request.FILES,
                                queryset=Author.objects.filter(name__startswith='O'))
        if formset.is_valid():
            formset.save()
            # Do something.
    else:
        formset = AuthorFormSet(queryset=Author.objects.filter(name__startswith='O'))
    return render_to_response("manage_authors.html", {
        "formset": formset,
    })
Note that we pass the queryset argument in both the POST and GET cases in this example.

Using the formset in the template
There are three ways to render a formset in a Django template.

First, you can let the formset do most of the work:

<form method="post" action="">
    {{ formset }}
</form>
Second, you can manually render the formset, but let the form deal with itself:

<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        {{ form }}
    {% endfor %}
</form>
When you manually render the forms yourself, be sure to render the management form as shown above. See the management form documentation.

Third, you can manually render each field:

<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        {% for field in form %}
            {{ field.label_tag }} {{ field }}
        {% endfor %}
    {% endfor %}
</form>
If you opt to use this third method and you don’t iterate over the fields with a {% for %} loop, you’ll need to render the primary key field. For example, if you were rendering the name and age fields of a model:

<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        {{ form.id }}
        <ul>
            <li>{{ form.name }}</li>
            <li>{{ form.age }}</li>
        </ul>
    {% endfor %}
</form>
Notice how we need to explicitly render {{ form.id }}. This ensures that the model formset, in the POST case, will work correctly. (This example assumes a primary key named id. If you’ve explicitly defined your own primary key that isn’t called id, make sure it gets rendered.)



Inline formsets
class models.BaseInlineFormSet
Inline formsets is a small abstraction layer on top of model formsets. These simplify the case of working with related objects via a foreign key. Suppose you have these two models:

from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author)
    title = models.CharField(max_length=100)
If you want to create a formset that allows you to edit books belonging to a particular author, you could do this:

from django.forms.models import inlineformset_factory
BookFormSet = inlineformset_factory(Author, Book)
author = Author.objects.get(name=u'Mike Royko')
formset = BookFormSet(instance=author)
Note

inlineformset_factory() uses modelformset_factory() and marks can_delete=True.




Overriding methods on an InlineFormSet
When overriding methods on InlineFormSet, you should subclass BaseInlineFormSet rather than BaseModelFormSet.

For example, if you want to override clean():

from django.forms.models import BaseInlineFormSet

class CustomInlineFormSet(BaseInlineFormSet):
    def clean(self):
        super(CustomInlineFormSet, self).clean()
        # example custom validation across forms in the formset
        for form in self.forms:
            # your custom formset validation
            ...
See also Overriding clean() on a ModelFormSet.

Then when you create your inline formset, pass in the optional argument formset:

from django.forms.models import inlineformset_factory
BookFormSet = inlineformset_factory(Author, Book, formset=CustomInlineFormSet)
author = Author.objects.get(name=u'Mike Royko')
formset = BookFormSet(instance=author)


More than one foreign key to the same model
If your model contains more than one foreign key to the same model, you’ll need to resolve the ambiguity manually using fk_name. For example, consider the following model:

class Friendship(models.Model):
    from_friend = models.ForeignKey(Friend)
    to_friend = models.ForeignKey(Friend)
    length_in_months = models.IntegerField()
To resolve this, you can use fk_name to inlineformset_factory():

FriendshipFormSet = inlineformset_factory(Friend, Friendship, fk_name="from_friend")
Using an inline formset in a view
You may want to provide a view that allows a user to edit the related objects of a model. Here’s how you can do that:

def manage_books(request, author_id):
    author = Author.objects.get(pk=author_id)
    BookInlineFormSet = inlineformset_factory(Author, Book)
    if request.method == "POST":
        formset = BookInlineFormSet(request.POST, request.FILES, instance=author)
        if formset.is_valid():
            formset.save()
            # Do something. Should generally end with a redirect. For example:
            return HttpResponseRedirect(author.get_absolute_url())
    else:
        formset = BookInlineFormSet(instance=author)
    return render_to_response("manage_books.html", {
        "formset": formset,
    })
Notice how we pass instance in both the POST and GET cases.

Specifying widgets to use in the inline form
inlineformset_factory uses modelformset_factory and passes most of its arguments to modelformset_factory. This means you can use the widgets parameter in much the same way as passing it to modelformset_factory. See Specifying widgets to use in the form with widgets above.






















Form Assets (the Media class)
Rendering an attractive and easy-to-use Web form requires more than just HTML - it also requires CSS stylesheets, and if you want to use fancy “Web2.0” widgets, you may also need to include some JavaScript on each page. The exact combination of CSS and JavaScript that is required for any given page will depend upon the widgets that are in use on that page.

This is where asset definitions come in. Django allows you to associate different files – like stylesheets and scripts – with the forms and widgets that require those assets. For example, if you want to use a calendar to render DateFields, you can define a custom Calendar widget. This widget can then be associated with the CSS and JavaScript that is required to render the calendar. When the Calendar widget is used on a form, Django is able to identify the CSS and JavaScript files that are required, and provide the list of file names in a form suitable for easy inclusion on your Web page.



Assets and Django Admin

The Django Admin application defines a number of customized widgets for calendars, filtered selections, and so on. These widgets define asset requirements, and the Django Admin uses the custom widgets in place of the Django defaults. The Admin templates will only include those files that are required to render the widgets on any given page.

If you like the widgets that the Django Admin application uses, feel free to use them in your own application! They’re all stored in django.contrib.admin.widgets.



Which JavaScript toolkit?

Many JavaScript toolkits exist, and many of them include widgets (such as calendar widgets) that can be used to enhance your application. Django has deliberately avoided blessing any one JavaScript toolkit. Each toolkit has its own relative strengths and weaknesses - use whichever toolkit suits your requirements. Django is able to integrate with any JavaScript toolkit.


Assets as a static definition
The easiest way to define assets is as a static definition. Using this method, the declaration is an inner Media class. The properties of the inner class define the requirements.


from django import forms

class CalendarWidget(forms.TextInput):
    class Media:
        css = {
            'all': ('pretty.css',)
        }
        js = ('animations.js', 'actions.js')
This code defines a CalendarWidget, which will be based on TextInput. Every time the CalendarWidget is used on a form, that form will be directed to include the CSS file pretty.css, and the JavaScript files animations.js and actions.js.

This static definition is converted at runtime into a widget property named media. The list of assets for a CalendarWidget instance can be retrieved through this property:

w = CalendarWidget()
print(w.media)
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://static.example.com/actions.js"></script>


Here’s a list of all possible Media options. There are no required options.

css
A dictionary describing the CSS files required for various forms of output media.

The values in the dictionary should be a tuple/list of file names. See the section on paths for details of how to specify paths to these files.

The keys in the dictionary are the output media types. These are the same types accepted by CSS files in media declarations: ‘all’, ‘aural’, ‘braille’, ‘embossed’, ‘handheld’, ‘print’, ‘projection’, ‘screen’, ‘tty’ and ‘tv’. If you need to have different stylesheets for different media types, provide a list of CSS files for each output medium. The following example would provide two CSS options – one for the screen, and one for print:

class Media:
    css = {
        'screen': ('pretty.css',),
        'print': ('newspaper.css',)
    }
If a group of CSS files are appropriate for multiple output media types, the dictionary key can be a comma separated list of output media types. In the following example, TV’s and projectors will have the same media requirements:

class Media:
    css = {
        'screen': ('pretty.css',),
        'tv,projector': ('lo_res.css',),
        'print': ('newspaper.css',)
    }
If this last CSS definition were to be rendered, it would become the following HTML:

<link href="http://static.example.com/pretty.css" type="text/css" media="screen" rel="stylesheet" />
<link href="http://static.example.com/lo_res.css" type="text/css" media="tv,projector" rel="stylesheet" />
<link href="http://static.example.com/newspaper.css" type="text/css" media="print" rel="stylesheet" />



js
A tuple describing the required JavaScript files. See the section on paths for details of how to specify paths to these files.



extend
A boolean defining inheritance behavior for Media declarations.

By default, any object using a static Media definition will inherit all the assets associated with the parent widget. This occurs regardless of how the parent defines its own requirements. For example, if we were to extend our basic Calendar widget from the example above:

class FancyCalendarWidget(CalendarWidget):
...     class Media:
...         css = {
...             'all': ('fancy.css',)
...         }
...         js = ('whizbang.js',)

w = FancyCalendarWidget()
print(w.media)
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
<link href="http://static.example.com/fancy.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://static.example.com/actions.js"></script>
<script type="text/javascript" src="http://static.example.com/whizbang.js"></script>
The FancyCalendar widget inherits all the assets from its parent widget. If you don’t want Media to be inherited in this way, add an extend=False declaration to the Media declaration:

class FancyCalendarWidget(CalendarWidget):
...     class Media:
...         extend = False
...         css = {
...             'all': ('fancy.css',)
...         }
...         js = ('whizbang.js',)

w = FancyCalendarWidget()
print(w.media)
<link href="http://static.example.com/fancy.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/whizbang.js"></script>
If you require even more control over inheritance, define your assets using a dynamic property. Dynamic properties give you complete control over which files are inherited, and which are not.



Media as a dynamic property
If you need to perform some more sophisticated manipulation of asset requirements, you can define the media property directly. This is done by defining a widget property that returns an instance of forms.Media. The constructor for forms.Media accepts css and js keyword arguments in the same format as that used in a static media definition.

For example, the static definition for our Calendar Widget could also be defined in a dynamic fashion:

class CalendarWidget(forms.TextInput):
    def _media(self):
        return forms.Media(css={'all': ('pretty.css',)},
                           js=('animations.js', 'actions.js'))
    media = property(_media)



Paths in asset definitions
Paths used to specify assets can be either relative or absolute. If a path starts with /, http:// or https://, it will be interpreted as an absolute path, and left as-is. All other paths will be prepended with the value of the appropriate prefix.

As part of the introduction of the staticfiles app two new settings were added to refer to “static files” (images, CSS, Javascript, etc.) that are needed to render a complete web page: STATIC_URL and STATIC_ROOT.

To find the appropriate prefix to use, Django will check if the STATIC_URL setting is not None and automatically fall back to using MEDIA_URL. For example, if the MEDIA_URL for your site was 'http://uploads.example.com/' and STATIC_URL was None:

from django import forms
class CalendarWidget(forms.TextInput):
...     class Media:
...         css = {
...             'all': ('/css/pretty.css',),
...         }
...         js = ('animations.js', 'http://othersite.com/actions.js')

w = CalendarWidget()
print(w.media)
<link href="/css/pretty.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://uploads.example.com/animations.js"></script>
<script type="text/javascript" src="http://othersite.com/actions.js"></script>
But if STATIC_URL is 'http://static.example.com/':

w = CalendarWidget()
print(w.media)
<link href="/css/pretty.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://othersite.com/actions.js"></script>



Media objects
When you interrogate the media attribute of a widget or form, the value that is returned is a forms.Media object. As we have already seen, the string representation of a Media object is the HTML required to include the relevant files in the <head> block of your HTML page.

However, Media objects have some other interesting properties.


Subsets of assets
If you only want files of a particular type, you can use the subscript operator to filter out a medium of interest. For example:

w = CalendarWidget()
print(w.media)
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://static.example.com/actions.js"></script>

print(w.media['css'])
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
When you use the subscript operator, the value that is returned is a new Media object – but one that only contains the media of interest.



Combining Media objects
Media objects can also be added together. When two Media objects are added, the resulting Media object contains the union of the assets specified by both:

from django import forms
class CalendarWidget(forms.TextInput):
...     class Media:
...         css = {
...             'all': ('pretty.css',)
...         }
...         js = ('animations.js', 'actions.js')

class OtherWidget(forms.TextInput):
...     class Media:
...         js = ('whizbang.js',)

w1 = CalendarWidget()
w2 = OtherWidget()
print(w1.media + w2.media)
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://static.example.com/actions.js"></script>
<script type="text/javascript" src="http://static.example.com/whizbang.js"></script>



Media on Forms
Widgets aren’t the only objects that can have media definitions – forms can also define media. The rules for media definitions on forms are the same as the rules for widgets: declarations can be static or dynamic; path and inheritance rules for those declarations are exactly the same.

Regardless of whether you define a media declaration, all Form objects have a media property. The default value for this property is the result of adding the media definitions for all widgets that are part of the form:

from django import forms
class ContactForm(forms.Form):
...     date = DateField(widget=CalendarWidget)
...     name = CharField(max_length=40, widget=OtherWidget)

f = ContactForm()
f.media
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://static.example.com/actions.js"></script>
<script type="text/javascript" src="http://static.example.com/whizbang.js"></script>
If you want to associate additional assets with a form – for example, CSS for form layout – simply add a Media declaration to the form:

class ContactForm(forms.Form):
...     date = DateField(widget=CalendarWidget)
...     name = CharField(max_length=40, widget=OtherWidget)
...
...     class Media:
...         css = {
...             'all': ('layout.css',)
...         }

f = ContactForm()
f.media
<link href="http://static.example.com/pretty.css" type="text/css" media="all" rel="stylesheet" />
<link href="http://static.example.com/layout.css" type="text/css" media="all" rel="stylesheet" />
<script type="text/javascript" src="http://static.example.com/animations.js"></script>
<script type="text/javascript" src="http://static.example.com/actions.js"></script>
<script type="text/javascript" src="http://static.example.com/whizbang.js"></script>




























Formsets
class BaseFormSet
A formset is a layer of abstraction to work with multiple forms on the same page. It can be best compared to a data grid. Let’s say you have the following form:

from django import forms
class ArticleForm(forms.Form):
...     title = forms.CharField()
...     pub_date = forms.DateField()
You might want to allow the user to create several articles at once. To create a formset out of an ArticleForm you would do:

from django.forms.formsets import formset_factory
ArticleFormSet = formset_factory(ArticleForm)

You now have created a formset named ArticleFormSet. The formset gives you the ability to iterate over the forms in the formset and display them as you would with a regular form:

formset = ArticleFormSet()
for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" id="id_form-0-pub_date" /></td></tr>

it only displayed one empty form. The number of empty forms that is displayed is controlled by the extra parameter. By default, formset_factory() defines one extra form; the following example will display two blank forms:

ArticleFormSet = formset_factory(ArticleForm, extra=2)
Iterating over the formset will render the forms in the order they were created. You can change this order by providing an alternate implementation for the __iter__() method.

Formsets can also be indexed into, which returns the corresponding form. If you override __iter__, you will need to also override __getitem__ to have matching behavior.



Using initial data with a formset
Initial data is what drives the main usability of a formset. As shown above you can define the number of extra forms. What this means is that you are telling the formset how many additional forms to show in addition to the number of forms it generates from the initial data. Let’s take a look at an example:

import datetime
from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm, extra=2)
formset = ArticleFormSet(initial=[
...     {'title': u'Django is now open source',
...      'pub_date': datetime.date.today(),}
... ])

for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" value="Django is now open source" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" value="2008-05-12" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-1-title">Title:</label></th><td><input type="text" name="form-1-title" id="id_form-1-title" /></td></tr>
<tr><th><label for="id_form-1-pub_date">Pub date:</label></th><td><input type="text" name="form-1-pub_date" id="id_form-1-pub_date" /></td></tr>
<tr><th><label for="id_form-2-title">Title:</label></th><td><input type="text" name="form-2-title" id="id_form-2-title" /></td></tr>
<tr><th><label for="id_form-2-pub_date">Pub date:</label></th><td><input type="text" name="form-2-pub_date" id="id_form-2-pub_date" /></td></tr>
There are now a total of three forms showing above. One for the initial data that was passed in and two extra forms. Also note that we are passing in a list of dictionaries as the initial data.




Limiting the maximum number of forms
The max_num parameter to formset_factory() gives you the ability to limit the number of forms the formset will display:

from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm, extra=2, max_num=1)
formset = ArticleFormSet()
for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" id="id_form-0-pub_date" /></td></tr>
If the value of max_num is greater than the number of existing items in the initial data, up to extra additional blank forms will be added to the formset, so long as the total number of forms does not exceed max_num. For example, if extra=2 and max_num=2 and the formset is initialized with one initial item, a form for the initial item and one blank form will be displayed.

If the number of items in the initial data exceeds max_num, all initial data forms will be displayed regardless of the value of max_num and no extra forms will be displayed. For example, if extra=3 and max_num=1 and the formset is initialized with two initial items, two forms with the initial data will be displayed.

A max_num value of None (the default) puts a high limit on the number of forms displayed (1000). In practice this is equivalent to no limit.

By default, max_num only affects how many forms are displayed and does not affect validation. If validate_max=True is passed to the formset_factory(), then max_num will affect validation. See Validating the number of forms in a formset.

The validate_max parameter was added to formset_factory(). Also, the behavior of FormSet was brought in line with that of ModelFormSet so that it displays initial data regardless of max_num.


Formset validation
Validation with a formset is almost identical to a regular Form. There is an is_valid method on the formset to provide a convenient way to validate all forms in the formset:

from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm)
data = {
...     'form-TOTAL_FORMS': u'1',
...     'form-INITIAL_FORMS': u'0',
...     'form-MAX_NUM_FORMS': u'',
... }
formset = ArticleFormSet(data)
formset.is_valid()
True
We passed in no data to the formset which is resulting in a valid form. The formset is smart enough to ignore extra forms that were not changed. If we provide an invalid article:

data = {
...     'form-TOTAL_FORMS': u'2',
...     'form-INITIAL_FORMS': u'0',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'Test',
...     'form-0-pub_date': u'1904-06-16',
...     'form-1-title': u'Test',
...     'form-1-pub_date': u'', # <-- this date is missing but required
... }
formset = ArticleFormSet(data)
formset.is_valid()
False
formset.errors
[{}, {'pub_date': [u'This field is required.']}]
As we can see, formset.errors is a list whose entries correspond to the forms in the formset. Validation was performed for each of the two forms, and the expected error message appears for the second item.




BaseFormSet.total_error_count()

To check how many errors there are in the formset, we can use the total_error_count method:

# Using the previous example
formset.errors
[{}, {'pub_date': [u'This field is required.']}]
len(formset.errors)
2
formset.total_error_count()
1
We can also check if form data differs from the initial data (i.e. the form was sent without any data):

data = {
...     'form-TOTAL_FORMS': u'1',
...     'form-INITIAL_FORMS': u'0',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'',
...     'form-0-pub_date': u'',
... }
formset = ArticleFormSet(data)
formset.has_changed()
False



Understanding the ManagementForm
You may have noticed the additional data (form-TOTAL_FORMS, form-INITIAL_FORMS and form-MAX_NUM_FORMS) that was required in the formset’s data above. This data is required for the ManagementForm. This form is used by the formset to manage the collection of forms contained in the formset. If you don’t provide this management data, an exception will be raised:

data = {
...     'form-0-title': u'Test',
...     'form-0-pub_date': u'',
... }
formset = ArticleFormSet(data)
formset.is_valid()
Traceback (most recent call last):
...
django.forms.utils.ValidationError: [u'ManagementForm data is missing or has been tampered with']
It is used to keep track of how many form instances are being displayed. If you are adding new forms via JavaScript, you should increment the count fields in this form as well. On the other hand, if you are using JavaScript to allow deletion of existing objects, then you need to ensure the ones being removed are properly marked for deletion by including form-#-DELETE in the POST data. It is expected that all forms are present in the POST data regardless.

The management form is available as an attribute of the formset itself. When rendering a formset in a template, you can include all the management data by rendering {{ my_formset.management_form }} (substituting the name of your formset as appropriate).



total_form_count and initial_form_count
BaseFormSet has a couple of methods that are closely related to the ManagementForm, total_form_count and initial_form_count.

total_form_count returns the total number of forms in this formset. initial_form_count returns the number of forms in the formset that were pre-filled, and is also used to determine how many forms are required. You will probably never need to override either of these methods, so please be sure you understand what they do before doing so.



empty_form
BaseFormSet provides an additional attribute empty_form which returns a form instance with a prefix of __prefix__ for easier use in dynamic forms with JavaScript.




Custom formset validation
A formset has a clean method similar to the one on a Form class. This is where you define your own validation that works at the formset level:

from django.forms.formsets import BaseFormSet
from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm

class BaseArticleFormSet(BaseFormSet):
...     def clean(self):
...         """Checks that no two articles have the same title."""
...         if any(self.errors):
...             # Don't bother validating the formset unless each form is valid on its own
...             return
...         titles = []
...         for form in self.forms:
...             title = form.cleaned_data['title']
...             if title in titles:
...                 raise forms.ValidationError("Articles in a set must have distinct titles.")
...             titles.append(title)

ArticleFormSet = formset_factory(ArticleForm, formset=BaseArticleFormSet)
data = {
...     'form-TOTAL_FORMS': u'2',
...     'form-INITIAL_FORMS': u'0',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'Test',
...     'form-0-pub_date': u'1904-06-16',
...     'form-1-title': u'Test',
...     'form-1-pub_date': u'1912-06-23',
... }
formset = ArticleFormSet(data)
formset.is_valid()
False
formset.errors
[{}, {}]
formset.non_form_errors()
[u'Articles in a set must have distinct titles.']
The formset clean method is called after all the Form.clean methods have been called. The errors will be found using the non_form_errors() method on the formset.



Validating the number of forms in a formset
Django provides a couple ways to validate the minimum or maximum number of submitted forms. Applications which need more customizable validation of the number of forms should use custom formset validation.

validate_max
If validate_max=True is passed to formset_factory(), validation will also check that the number of forms in the data set, minus those marked for deletion, is less than or equal to max_num.

from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm, max_num=1, validate_max=True)
data = {
...     'form-TOTAL_FORMS': u'2',
...     'form-INITIAL_FORMS': u'0',
...     'form-MIN_NUM_FORMS': u'',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'Test',
...     'form-0-pub_date': u'1904-06-16',
...     'form-1-title': u'Test 2',
...     'form-1-pub_date': u'1912-06-23',
... }
formset = ArticleFormSet(data)
formset.is_valid()
False
formset.errors
[{}, {}]
formset.non_form_errors()
[u'Please submit 1 or fewer forms.']
validate_max=True validates against max_num strictly even if max_num was exceeded because the amount of initial data supplied was excessive.


Regardless of validate_max, if the number of forms in a data set exceeds max_num by more than 1000, then the form will fail to validate as if validate_max were set, and additionally only the first 1000 forms above max_num will be validated. The remainder will be truncated entirely. This is to protect against memory exhaustion attacks using forged POST requests.


The validate_max parameter was added to formset_factory().



validate_min

If validate_min=True is passed to formset_factory(), validation will also check that the number of forms in the data set, minus those marked for deletion, is greater than or equal to min_num.

from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm, min_num=3, validate_min=True)
data = {
...     'form-TOTAL_FORMS': u'2',
...     'form-INITIAL_FORMS': u'0',
...     'form-MIN_NUM_FORMS': u'',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'Test',
...     'form-0-pub_date': u'1904-06-16',
...     'form-1-title': u'Test 2',
...     'form-1-pub_date': u'1912-06-23',
... }
formset = ArticleFormSet(data)
formset.is_valid()
False
formset.errors
[{}, {}]
formset.non_form_errors()
[u'Please submit 3 or more forms.']
Changed in Django 1.7: 
The min_num and validate_min parameters were added to formset_factory().




Dealing with ordering and deletion of forms
The formset_factory() provides two optional parameters can_order and can_delete to help with ordering of forms in formsets and deletion of forms from a formset.

can_order
BaseFormSet.can_order
Default: False

Lets you create a formset with the ability to order:

from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm, can_order=True)
formset = ArticleFormSet(initial=[
...     {'title': u'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': u'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" value="Article #1" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" value="2008-05-10" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-0-ORDER">Order:</label></th><td><input type="number" name="form-0-ORDER" value="1" id="id_form-0-ORDER" /></td></tr>
<tr><th><label for="id_form-1-title">Title:</label></th><td><input type="text" name="form-1-title" value="Article #2" id="id_form-1-title" /></td></tr>
<tr><th><label for="id_form-1-pub_date">Pub date:</label></th><td><input type="text" name="form-1-pub_date" value="2008-05-11" id="id_form-1-pub_date" /></td></tr>
<tr><th><label for="id_form-1-ORDER">Order:</label></th><td><input type="number" name="form-1-ORDER" value="2" id="id_form-1-ORDER" /></td></tr>
<tr><th><label for="id_form-2-title">Title:</label></th><td><input type="text" name="form-2-title" id="id_form-2-title" /></td></tr>
<tr><th><label for="id_form-2-pub_date">Pub date:</label></th><td><input type="text" name="form-2-pub_date" id="id_form-2-pub_date" /></td></tr>
<tr><th><label for="id_form-2-ORDER">Order:</label></th><td><input type="number" name="form-2-ORDER" id="id_form-2-ORDER" /></td></tr>
This adds an additional field to each form. This new field is named ORDER and is an forms.IntegerField. For the forms that came from the initial data it automatically assigned them a numeric value. Let’s look at what will happen when the user changes these values:

data = {
...     'form-TOTAL_FORMS': u'3',
...     'form-INITIAL_FORMS': u'2',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'Article #1',
...     'form-0-pub_date': u'2008-05-10',
...     'form-0-ORDER': u'2',
...     'form-1-title': u'Article #2',
...     'form-1-pub_date': u'2008-05-11',
...     'form-1-ORDER': u'1',
...     'form-2-title': u'Article #3',
...     'form-2-pub_date': u'2008-05-01',
...     'form-2-ORDER': u'0',
... }

formset = ArticleFormSet(data, initial=[
...     {'title': u'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': u'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
formset.is_valid()
True
for form in formset.ordered_forms:
...     print(form.cleaned_data)
{'pub_date': datetime.date(2008, 5, 1), 'ORDER': 0, 'title': u'Article #3'}
{'pub_date': datetime.date(2008, 5, 11), 'ORDER': 1, 'title': u'Article #2'}
{'pub_date': datetime.date(2008, 5, 10), 'ORDER': 2, 'title': u'Article #1'}
can_delete
BaseFormSet.can_delete
Default: False

Lets you create a formset with the ability to select forms for deletion:

from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
ArticleFormSet = formset_factory(ArticleForm, can_delete=True)
formset = ArticleFormSet(initial=[
...     {'title': u'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': u'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" value="Article #1" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" value="2008-05-10" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-0-DELETE">Delete:</label></th><td><input type="checkbox" name="form-0-DELETE" id="id_form-0-DELETE" /></td></tr>
<tr><th><label for="id_form-1-title">Title:</label></th><td><input type="text" name="form-1-title" value="Article #2" id="id_form-1-title" /></td></tr>
<tr><th><label for="id_form-1-pub_date">Pub date:</label></th><td><input type="text" name="form-1-pub_date" value="2008-05-11" id="id_form-1-pub_date" /></td></tr>
<tr><th><label for="id_form-1-DELETE">Delete:</label></th><td><input type="checkbox" name="form-1-DELETE" id="id_form-1-DELETE" /></td></tr>
<tr><th><label for="id_form-2-title">Title:</label></th><td><input type="text" name="form-2-title" id="id_form-2-title" /></td></tr>
<tr><th><label for="id_form-2-pub_date">Pub date:</label></th><td><input type="text" name="form-2-pub_date" id="id_form-2-pub_date" /></td></tr>
<tr><th><label for="id_form-2-DELETE">Delete:</label></th><td><input type="checkbox" name="form-2-DELETE" id="id_form-2-DELETE" /></td></tr>
Similar to can_order this adds a new field to each form named DELETE and is a forms.BooleanField. When data comes through marking any of the delete fields you can access them with deleted_forms:

data = {
...     'form-TOTAL_FORMS': u'3',
...     'form-INITIAL_FORMS': u'2',
...     'form-MAX_NUM_FORMS': u'',
...     'form-0-title': u'Article #1',
...     'form-0-pub_date': u'2008-05-10',
...     'form-0-DELETE': u'on',
...     'form-1-title': u'Article #2',
...     'form-1-pub_date': u'2008-05-11',
...     'form-1-DELETE': u'',
...     'form-2-title': u'',
...     'form-2-pub_date': u'',
...     'form-2-DELETE': u'',
... }

formset = ArticleFormSet(data, initial=[
...     {'title': u'Article #1', 'pub_date': datetime.date(2008, 5, 10)},
...     {'title': u'Article #2', 'pub_date': datetime.date(2008, 5, 11)},
... ])
[form.cleaned_data for form in formset.deleted_forms]
[{'DELETE': True, 'pub_date': datetime.date(2008, 5, 10), 'title': u'Article #1'}]
If you are using a ModelFormSet, model instances for deleted forms will be deleted when you call formset.save().


If you call formset.save(commit=False), objects will not be deleted automatically. You’ll need to call delete() on each of the formset.deleted_objects to actually delete them:

instances = formset.save(commit=False)
for obj in formset.deleted_objects:
...     obj.delete()
If you want to maintain backwards compatibility with Django 1.6 and earlier, you can do something like this:

try:
    # For Django 1.7+
    for obj in formset.deleted_objects:
        obj.delete()
except AssertionError:
    # Django 1.6 and earlier already deletes the objects, trying to
    # delete them a second time raises an AssertionError.
    pass
On the other hand, if you are using a plain FormSet, it’s up to you to handle formset.deleted_forms, perhaps in your formset’s save() method, as there’s no general notion of what it means to delete a form.




Adding additional fields to a formset
If you need to add additional fields to the formset this can be easily accomplished. The formset base class provides an add_fields method. You can simply override this method to add your own fields or even redefine the default fields/attributes of the order and deletion fields:

from django.forms.formsets import BaseFormSet
from django.forms.formsets import formset_factory
from myapp.forms import ArticleForm
class BaseArticleFormSet(BaseFormSet):
...     def add_fields(self, form, index):
...         super(BaseArticleFormSet, self).add_fields(form, index)
...         form.fields["my_field"] = forms.CharField()

ArticleFormSet = formset_factory(ArticleForm, formset=BaseArticleFormSet)
formset = ArticleFormSet()
for form in formset:
...     print(form.as_table())
<tr><th><label for="id_form-0-title">Title:</label></th><td><input type="text" name="form-0-title" id="id_form-0-title" /></td></tr>
<tr><th><label for="id_form-0-pub_date">Pub date:</label></th><td><input type="text" name="form-0-pub_date" id="id_form-0-pub_date" /></td></tr>
<tr><th><label for="id_form-0-my_field">My field:</label></th><td><input type="text" name="form-0-my_field" id="id_form-0-my_field" /></td></tr>
Using a formset in views and templates
Using a formset inside a view is as easy as using a regular Form class. The only thing you will want to be aware of is making sure to use the management form inside the template. Let’s look at a sample view:

from django.forms.formsets import formset_factory
from django.shortcuts import render_to_response
from myapp.forms import ArticleForm

def manage_articles(request):
    ArticleFormSet = formset_factory(ArticleForm)
    if request.method == 'POST':
        formset = ArticleFormSet(request.POST, request.FILES)
        if formset.is_valid():
            # do something with the formset.cleaned_data
            pass
    else:
        formset = ArticleFormSet()
    return render_to_response('manage_articles.html', {'formset': formset})
The manage_articles.html template might look like this:

<form method="post" action="">
    {{ formset.management_form }}
    <table>
        {% for form in formset %}
        {{ form }}
        {% endfor %}
    </table>
</form>
However there’s a slight shortcut for the above by letting the formset itself deal with the management form:

<form method="post" action="">
    <table>
        {{ formset }}
    </table>
</form>
The above ends up calling the as_table method on the formset class.




Manually rendered can_delete and can_order
If you manually render fields in the template, you can render can_delete parameter with {{ form.DELETE }}:

<form method="post" action="">
    {{ formset.management_form }}
    {% for form in formset %}
        <ul>
            <li>{{ form.title }}</li>
            <li>{{ form.pub_date }}</li>
            {% if formset.can_delete %}
                <li>{{ form.DELETE }}</li>
            {% endif %}
        </ul>
    {% endfor %}
</form>
Similarly, if the formset has the ability to order (can_order=True), it is possible to render it with {{ form.ORDER }}.



Using more than one formset in a view
You are able to use more than one formset in a view if you like. Formsets borrow much of its behavior from forms. With that said you are able to use prefix to prefix formset form field names with a given value to allow more than one formset to be sent to a view without name clashing. Lets take a look at how this might be accomplished:

from django.forms.formsets import formset_factory
from django.shortcuts import render_to_response
from myapp.forms import ArticleForm, BookForm

def manage_articles(request):
    ArticleFormSet = formset_factory(ArticleForm)
    BookFormSet = formset_factory(BookForm)
    if request.method == 'POST':
        article_formset = ArticleFormSet(request.POST, request.FILES, prefix='articles')
        book_formset = BookFormSet(request.POST, request.FILES, prefix='books')
        if article_formset.is_valid() and book_formset.is_valid():
            # do something with the cleaned_data on the formsets.
            pass
    else:
        article_formset = ArticleFormSet(prefix='articles')
        book_formset = BookFormSet(prefix='books')
    return render_to_response('manage_articles.html', {
        'article_formset': article_formset,
        'book_formset': book_formset,
    })
You would then render the formsets as normal. It is important to point out that you need to pass prefix on both the POST and non-POST cases so that it is rendered and processed correctly.






















Form and field validation
Form validation happens when the data is cleaned. If you want to customize this process, there are various places you can change, each one serving a different purpose. Three types of cleaning methods are run during form processing. These are normally executed when you call the is_valid() method on a form. There are other things that can trigger cleaning and validation (accessing the errors attribute or calling full_clean() directly), but normally they won’t be needed.

In general, any cleaning method can raise ValidationError if there is a problem with the data it is processing, passing the relevant information to the ValidationError constructor. See below for the best practice in raising ValidationError. If no ValidationError is raised, the method should return the cleaned (normalized) data as a Python object.

Most validation can be done using validators - simple helpers that can be reused easily. Validators are simple functions (or callables) that take a single argument and raise ValidationError on invalid input. Validators are run after the field’s to_python and validate methods have been called.

Validation of a Form is split into several steps, which can be customized or overridden:

•The to_python() method on a Field is the first step in every validation. It coerces the value to correct datatype and raises ValidationError if that is not possible. This method accepts the raw value from the widget and returns the converted value. For example, a FloatField will turn the data into a Python float or raise a ValidationError.

•The validate() method on a Field handles field-specific validation that is not suitable for a validator. It takes a value that has been coerced to correct datatype and raises ValidationError on any error. This method does not return anything and shouldn’t alter the value. You should override it to handle validation logic that you can’t or don’t want to put in a validator.

•The run_validators() method on a Field runs all of the field’s validators and aggregates all the errors into a single ValidationError. You shouldn’t need to override this method.

•The clean() method on a Field subclass. This is responsible for running to_python, validate and run_validators in the correct order and propagating their errors. If, at any time, any of the methods raise ValidationError, the validation stops and that error is raised. This method returns the clean data, which is then inserted into the cleaned_data dictionary of the form.

•The clean_<fieldname>() method in a form subclass – where <fieldname> is replaced with the name of the form field attribute. This method does any cleaning that is specific to that particular attribute, unrelated to the type of field that it is. This method is not passed any parameters. You will need to look up the value of the field in self.cleaned_data and remember that it will be a Python object at this point, not the original string submitted in the form (it will be in cleaned_data because the general field clean() method, above, has already cleaned the data once).



For example, if you wanted to validate that the contents of a CharField called serialnumber was unique, clean_serialnumber() would be the right place to do this. You don’t need a specific field (it’s just a CharField), but you want a formfield-specific piece of validation and, possibly, cleaning/normalizing the data.

This method should return the cleaned value obtained from cleaned_data, regardless of whether it changed anything or not.

•The Form subclass’s clean() method. This method can perform any validation that requires access to multiple fields from the form at once. This is where you might put in things to check that if field A is supplied, field B must contain a valid email address and the like. This method can return a completely different dictionary if it wishes, which will be used as the cleaned_data.

Since the field validation methods have been run by the time clean() is called, you also have access to the form’s errors attribute which contains all the errors raised by cleaning of individual fields.

Note that any errors raised by your Form.clean() override will not be associated with any field in particular. They go into a special “field” (called __all__), which you can access via the non_field_errors() method if you need to. If you want to attach errors to a specific field in the form, you need to call add_error().

Also note that there are special considerations when overriding the clean() method of a ModelForm subclass. (see the ModelForm documentation for more information)

These methods are run in the order given above, one field at a time. That is, for each field in the form (in the order they are declared in the form definition), the Field.clean() method (or its override) is run, then clean_<fieldname>(). Finally, once those two methods are run for every field, the Form.clean() method, or its override, is executed whether or not the previous methods have raised errors.



Raising ValidationError
In order to make error messages flexible and easy to override, consider the following guidelines:

•Provide a descriptive error code to the constructor:

# Good
ValidationError(_('Invalid value'), code='invalid')

# Bad
ValidationError(_('Invalid value'))
•Don’t coerce variables into the message; use placeholders and the params argument of the constructor:

# Good
ValidationError(
    _('Invalid value: %(value)s'),
    params={'value': '42'},
)

# Bad
ValidationError(_('Invalid value: %s') % value)
•Use mapping keys instead of positional formatting. This enables putting the variables in any order or omitting them altogether when rewriting the message:

# Good
ValidationError(
    _('Invalid value: %(value)s'),
    params={'value': '42'},
)

# Bad
ValidationError(
    _('Invalid value: %s'),
    params=('42',),
)
•Wrap the message with gettext to enable translation:

# Good
ValidationError(_('Invalid value'))

# Bad
ValidationError('Invalid value')
Putting it all together:

raise ValidationError(
    _('Invalid value: %(value)s'),
    code='invalid',
    params={'value': '42'},
)
Following these guidelines is particularly necessary if you write reusable forms, form fields, and model fields.

While not recommended, if you are at the end of the validation chain (i.e. your form clean() method) and you know you will never need to override your error message you can still opt for the less verbose:

ValidationError(_('Invalid value: %s') % value)

The Form.errors.as_data() and Form.errors.as_json() methods greatly benefit from fully featured ValidationErrors (with a code name and a params dictionary).




Raising multiple errors
If you detect multiple errors during a cleaning method and wish to signal all of them to the form submitter, it is possible to pass a list of errors to the ValidationError constructor.

As above, it is recommended to pass a list of ValidationError instances with codes and params but a list of strings will also work:

# Good
raise ValidationError([
    ValidationError(_('Error 1'), code='error1'),
    ValidationError(_('Error 2'), code='error2'),
])

# Bad
raise ValidationError([
    _('Error 1'),
    _('Error 2'),
])





Using validators
Django’s form (and model) fields support use of simple utility functions and classes known as validators. A validator is merely a callable object or function that takes a value and simply returns nothing if the value is valid or raises a ValidationError if not. These can be passed to a field’s constructor, via the field’s validators argument, or defined on the Field class itself with the default_validators attribute.

Simple validators can be used to validate values inside the field, let’s have a look at Django’s SlugField:

from django.forms import CharField
from django.core import validators

class SlugField(CharField):
    default_validators = [validators.validate_slug]
As you can see, SlugField is just a CharField with a customized validator that validates that submitted text obeys to some character rules. This can also be done on field definition so:

slug = forms.SlugField()
is equivalent to:

slug = forms.CharField(validators=[validators.validate_slug])
Common cases such as validating against an email or a regular expression can be handled using existing validator classes available in Django. For example, validators.validate_slug is an instance of a RegexValidator constructed with the first argument being the pattern: ^[-a-zA-Z0-9_]+$. See the section on writing validators to see a list of what is already available and for an example of how to write a validator.



Form field default cleaning
Let’s first create a custom form field that validates its input is a string containing comma-separated email addresses. The full class looks like this:

from django import forms
from django.core.validators import validate_email

class MultiEmailField(forms.Field):
    def to_python(self, value):
        "Normalize data to a list of strings."

        # Return an empty list if no input was given.
        if not value:
            return []
        return value.split(',')

    def validate(self, value):
        "Check if value consists only of valid emails."

        # Use the parent's handling of required fields, etc.
        super(MultiEmailField, self).validate(value)

        for email in value:
            validate_email(email)
Every form that uses this field will have these methods run before anything else can be done with the field’s data. This is cleaning that is specific to this type of field, regardless of how it is subsequently used.

Let’s create a simple ContactForm to demonstrate how you’d use this field:

class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField()
    sender = forms.EmailField()
    recipients = MultiEmailField()
    cc_myself = forms.BooleanField(required=False)
Simply use MultiEmailField like any other form field. When the is_valid() method is called on the form, the MultiEmailField.clean() method will be run as part of the cleaning process and it will, in turn, call the custom to_python() and validate() methods.




Cleaning a specific field attribute
Continuing on from the previous example, suppose that in our ContactForm, we want to make sure that the recipients field always contains the address "fred@example.com". This is validation that is specific to our form, so we don’t want to put it into the general MultiEmailField class. Instead, we write a cleaning method that operates on the recipients field, like so:

from django import forms

class ContactForm(forms.Form):
    # Everything as before.
    ...

    def clean_recipients(self):
        data = self.cleaned_data['recipients']
        if "fred@example.com" not in data:
            raise forms.ValidationError("You have forgotten about Fred!")

        # Always return the cleaned data, whether you have changed it or
        # not.
        return data



Cleaning and validating fields that depend on each other
Suppose we add another requirement to our contact form: if the cc_myself field is True, the subject must contain the word "help". We are performing validation on more than one field at a time, so the form’s clean() method is a good spot to do this. Notice that we are talking about the clean() method on the form here, whereas earlier we were writing a clean() method on a field. It’s important to keep the field and form difference clear when working out where to validate things. Fields are single data points, forms are a collection of fields.

By the time the form’s clean() method is called, all the individual field clean methods will have been run (the previous two sections), so self.cleaned_data will be populated with any data that has survived so far. So you also need to remember to allow for the fact that the fields you are wanting to validate might not have survived the initial individual field checks.

There are two ways to report any errors from this step. Probably the most common method is to display the error at the top of the form. To create such an error, you can raise a ValidationError from the clean() method. For example:

from django import forms

class ContactForm(forms.Form):
    # Everything as before.
    ...

    def clean(self):
        cleaned_data = super(ContactForm, self).clean()
        cc_myself = cleaned_data.get("cc_myself")
        subject = cleaned_data.get("subject")

        if cc_myself and subject:
            # Only do something if both fields are valid so far.
            if "help" not in subject:
                raise forms.ValidationError("Did not send for 'help' in "
                        "the subject despite CC'ing yourself.")

In previous versions of Django, form.clean() was required to return a dictionary of cleaned_data. This method may still return a dictionary of data to be used, but it’s no longer required.

In this code, if the validation error is raised, the form will display an error message at the top of the form (normally) describing the problem.

Note that the call to super(ContactForm, self).clean() in the example code ensures that any validation logic in parent classes is maintained.

The second approach might involve assigning the error message to one of the fields. In this case, let’s assign an error message to both the “subject” and “cc_myself” rows in the form display. Be careful when doing this in practice, since it can lead to confusing form output. We’re showing what is possible here and leaving it up to you and your designers to work out what works effectively in your particular situation. Our new code (replacing the previous sample) looks like this:

from django import forms

class ContactForm(forms.Form):
    # Everything as before.
    ...

    def clean(self):
        cleaned_data = super(ContactForm, self).clean()
        cc_myself = cleaned_data.get("cc_myself")
        subject = cleaned_data.get("subject")

        if cc_myself and subject and "help" not in subject:
            msg = u"Must put 'help' in subject when cc'ing yourself."
            self.add_error('cc_myself', msg)
            self.add_error('subject', msg)
The second argument of add_error() can be a simple string, or preferably an instance of ValidationError. See Raising ValidationError for more details. Note that add_error() automatically removes the field from cleaned_data.






















Form preview
Django comes with an optional “form preview” application that helps automate the following workflow:



1.Displays the form as HTML on a Web page.
2.Validates the form data when it’s submitted via POST. a. If it’s valid, displays a preview page. b. If it’s not valid, redisplays the form with error messages.
3.When the “confirmation” form is submitted from the preview page, calls a hook that you define – a done() method that gets passed the valid data.
The framework enforces the required preview by passing a shared-secret hash to the preview page via hidden form fields. If somebody tweaks the form parameters on the preview page, the form submission will fail the hash-comparison test.



How to use FormPreview
1.Point Django at the default FormPreview templates. There are two ways to do this:

Add 'django.contrib.formtools' to your INSTALLED_APPS setting. This will work if your TEMPLATE_LOADERS setting includes the app_directories template loader (which is the case by default). See the template loader docs for more.
Otherwise, determine the full filesystem path to the django/contrib/formtools/templates directory, and add that directory to your TEMPLATE_DIRS setting.

2.Create a FormPreview subclass that overrides the done() method:

from django.contrib.formtools.preview import FormPreview
from django.http import HttpResponseRedirect
from myapp.models import SomeModel

class SomeModelFormPreview(FormPreview):

    def done(self, request, cleaned_data):
        # Do something with the cleaned_data, then redirect
        # to a "success" page.
        return HttpResponseRedirect('/form/success')
This method takes an HttpRequest object and a dictionary of the form data after it has been validated and cleaned. It should return an HttpResponseRedirect that is the end result of the form being submitted.

3.Change your URLconf to point to an instance of your FormPreview subclass:

from myapp.preview import SomeModelFormPreview
from myapp.forms import SomeModelForm
from django import forms
...and add the following line to the appropriate model in your URLconf:

(r'^post/$', SomeModelFormPreview(SomeModelForm)),
where SomeModelForm is a Form or ModelForm class for the model.

4.Run the Django server and visit /post/ in your browser.




















Form wizard
Django comes with an optional “form wizard” application that splits forms across multiple Web pages. It maintains state in one of the backends so that the full server-side processing can be delayed until the submission of the final form.

Here’s the basic workflow for how a user would use a wizard:

1.The user visits the first page of the wizard, fills in the form and submits it.
2.The server validates the data. If it’s invalid, the form is displayed again, with error messages. If it’s valid, the server saves the current state of the wizard in the backend and redirects to the next step.
3.Step 1 and 2 repeat, for every subsequent form in the wizard.
4.Once the user has submitted all the forms and all the data has been validated, the wizard processes the data – saving it to the database, sending an email, or whatever the application needs to do.


Usage
1.Define a number of Form classes – one per wizard page.
2.Create a WizardView subclass that specifies what to do once all of your forms have been submitted and validated. This also lets you override some of the wizard’s behavior.
3.Create some templates that render the forms. You can define a single, generic template to handle every one of the forms, or you can define a specific template for each form.
4.Add django.contrib.formtools to your INSTALLED_APPS list in your settings file.
5.Point your URLconf at your WizardView as_view() method.


Defining Form classes

For example, let’s write a “contact form” wizard, where the first page’s form collects the sender’s email address and subject, and the second page collects the message itself. Here’s what the forms.py might look like:

from django import forms

class ContactForm1(forms.Form):
    subject = forms.CharField(max_length=100)
    sender = forms.EmailField()

class ContactForm2(forms.Form):
    message = forms.CharField(widget=forms.Textarea)
Note


Creating a WizardView subclass
class SessionWizardView
class CookieWizardView

To use the SessionWizardView follow the instructions in the sessions documentation on how to enable sessions.

We will use the SessionWizardView in all examples but is completely fine to use the CookieWizardView instead. As with your Form classes, this WizardView class can live anywhere in your codebase, but convention is to put it in views.py.

The only requirement on this subclass is that it implement a done() method.

WizardView.done(form_list, form_dict, **kwargs)
This method specifies what should happen when the data for every form is submitted and validated. This method is passed a list and dictionary of validated Form instances.

In this simplistic example, rather than performing any database operation, the method simply renders a template of the validated data:

from django.shortcuts import render_to_response
from django.contrib.formtools.wizard.views import SessionWizardView

class ContactWizard(SessionWizardView):
    def done(self, form_list, **kwargs):
        return render_to_response('done.html', {
            'form_data': [form.cleaned_data for form in form_list],
        })
Note that this method will be called via POST, so it really ought to be a good Web citizen and redirect after processing the data. Here’s another example:

from django.http import HttpResponseRedirect
from django.contrib.formtools.wizard.views import SessionWizardView

class ContactWizard(SessionWizardView):
    def done(self, form_list, **kwargs):
        do_something_with_the_form_data(form_list)
        return HttpResponseRedirect('/page-to-redirect-to-when-done/')
In addition to form_list, the done() method is passed a form_dict, which allows you to access the wizard’s forms based on their step names. This is especially useful when using NamedUrlWizardView, for example:

def done(self, form_list, form_dict, **kwargs):
    user = form_dict['user'].save()
    credit_card = form_dict['credit_card'].save()
    # ...



Creating templates for the forms
Next, you’ll need to create a template that renders the wizard’s forms. By default, every form uses a template called formtools/wizard/wizard_form.html. You can change this template name by overriding either the template_name attribute or the get_template_names() method, which are documented in the TemplateResponseMixin documentation. The latter one allows you to use a different template for each form (see the example below).

This template expects a wizard object that has various items attached to it:

•form – The Form or BaseFormSet instance for the current step (either empty or with errors).
•steps – A helper object to access the various steps related data:
step0 – The current step (zero-based).
step1 – The current step (one-based).
count – The total number of steps.
first – The first step.
last – The last step.
current – The current (or first) step.
next – The next step.
prev – The previous step.
index – The index of the current step.
all – A list of all steps of the wizard.
You can supply additional context variables by using the get_context_data() method of your WizardView subclass.

Here’s a full example template:

{% extends "base.html" %}
{% load i18n %}

{% block head %}
{{ wizard.form.media }}
{% endblock %}

{% block content %}
<p>Step {{ wizard.steps.step1 }} of {{ wizard.steps.count }}</p>
<form action="" method="post">{% csrf_token %}
<table>
{{ wizard.management_form }}
{% if wizard.form.forms %}
    {{ wizard.form.management_form }}
    {% for form in wizard.form.forms %}
        {{ form }}
    {% endfor %}
{% else %}
    {{ wizard.form }}
{% endif %}
</table>
{% if wizard.steps.prev %}
<button name="wizard_goto_step" type="submit" value="{{ wizard.steps.first }}">{% trans "first step" %}</button>
<button name="wizard_goto_step" type="submit" value="{{ wizard.steps.prev }}">{% trans "prev step" %}</button>
{% endif %}
<input type="submit" value="{% trans "submit" %}"/>
</form>
{% endblock %}
Note

Note that {{ wizard.management_form }} must be used for the wizard to work properly.



Hooking the wizard into a URLconf
WizardView.as_view()
Finally, we need to specify which forms to use in the wizard, and then deploy the new WizardView object at a URL in the urls.py. The wizard’s as_view() method takes a list of your Form classes as an argument during instantiation:

from django.conf.urls import patterns

from myapp.forms import ContactForm1, ContactForm2
from myapp.views import ContactWizard

urlpatterns = patterns('',
    (r'^contact/$', ContactWizard.as_view([ContactForm1, ContactForm2])),
)

You can also pass the form list as a class attribute named form_list:

class ContactWizard(WizardView):
    form_list = [ContactForm1, ContactForm2]



Using a different template for each form
As mentioned above, you may specify a different template for each form. Consider an example using a form wizard to implement a multi-step checkout process for an online store. In the first step, the user specifies a billing and shipping address. In the second step, the user chooses payment type. If they chose to pay by credit card, they will enter credit card information in the next step. In the final step, they will confirm the purchase.

Here’s what the view code might look like:

from django.http import HttpResponseRedirect
from django.contrib.formtools.wizard.views import SessionWizardView

FORMS = [("address", myapp.forms.AddressForm),
         ("paytype", myapp.forms.PaymentChoiceForm),
         ("cc", myapp.forms.CreditCardForm),
         ("confirmation", myapp.forms.OrderForm)]

TEMPLATES = {"address": "checkout/billingaddress.html",
             "paytype": "checkout/paymentmethod.html",
             "cc": "checkout/creditcard.html",
             "confirmation": "checkout/confirmation.html"}

def pay_by_credit_card(wizard):
    """Return true if user opts to pay by credit card"""
    # Get cleaned data from payment step
    cleaned_data = wizard.get_cleaned_data_for_step('paytype') or {'method': 'none'}
    # Return true if the user selected credit card
    return cleaned_data['method'] == 'cc'


class OrderWizard(SessionWizardView):
    def get_template_names(self):
        return [TEMPLATES[self.steps.current]]

    def done(self, form_list, **kwargs):
        do_something_with_the_form_data(form_list)
        return HttpResponseRedirect('/page-to-redirect-to-when-done/')
        ...
The urls.py file would contain something like:

urlpatterns = patterns('',
    (r'^checkout/$', OrderWizard.as_view(FORMS, condition_dict={'cc': pay_by_credit_card})),
)
 
The condition_dict can be passed as attribute for the as_view() method or as a class attribute named condition_dict:

class OrderWizard(WizardView):
    condition_dict = {'cc': pay_by_credit_card}
Note that the OrderWizard object is initialized with a list of pairs. The first element in the pair is a string that corresponds to the name of the step and the second is the form class.

In this example, the get_template_names() method returns a list containing a single template, which is selected based on the name of the current step.


















SETTING


A settings file is just a Python module with module-level variables.

Here are a couple of example settings:

DEBUG = False
DEFAULT_FROM_EMAIL = 'webmaster@example.com'
TEMPLATE_DIRS = ('/home/templates/mike', '/home/templates/john')
Note

If you set DEBUG to False, you also need to properly set the ALLOWED_HOSTS setting.

Because a settings file is a Python module, the following apply:

•It doesn’t allow for Python syntax errors.

•It can assign settings dynamically using normal Python syntax. For example:

MY_SETTING = [str(i) for i in range(30)]
•It can import values from other settings files.




DJANGO_SETTINGS_MODULE
When you use Django, you have to tell it which settings you’re using. Do this by using an environment variable, DJANGO_SETTINGS_MODULE.

The value of DJANGO_SETTINGS_MODULE should be in Python path syntax, e.g. mysite.settings. Note that the settings module should be on the Python import search path.



The django-admin.py utility
When using django-admin.py, you can either set the environment variable once, or explicitly pass in the settings module each time you run the utility.

Example (Unix Bash shell):

export DJANGO_SETTINGS_MODULE=mysite.settings
django-admin.py runserver


Example (Windows shell):

set DJANGO_SETTINGS_MODULE=mysite.settings
django-admin.py runserver
Use the --settings command-line argument to specify the settings manually:

django-admin.py runserver --settings=mysite.settings

On the server (mod_wsgi)
In your live server environment, you’ll need to tell your WSGI application what settings file to use. Do that with os.environ:

import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'



Default settings

Here’s the algorithm Django uses in compiling settings:

•Load settings from global_settings.py.
•Load settings from the specified settings file, overriding the global settings as necessary.
Note that a settings file should not import from global_settings, because that’s redundant.



Seeing which settings you’ve changed
The command python manage.py diffsettings displays differences between the current settings file and Django’s default settings.



Using settings in Python code
In your Django apps, use settings by importing the object django.conf.settings. Example:

from django.conf import settings

if settings.DEBUG:
    # Do something
Note that django.conf.settings isn’t a module – it’s an object. So importing individual settings is not possible:

from django.conf.settings import DEBUG  # This won't work.

Also note that your code should not import from either global_settings or your own settings file. django.conf.settings abstracts the concepts of default settings and site-specific settings; it presents a single interface. It also decouples the code that uses settings from the location of your settings.



Altering settings at runtime
You shouldn’t alter settings in your applications at runtime. For example, don’t do this in a view:

from django.conf import settings

settings.DEBUG = True   # Don't do this!
The only place you should assign to settings is in a settings file.



Security
Because a settings file contains sensitive information, such as the database password, you should make every attempt to limit access to it. For example, change its file permissions so that only you and your Web server’s user can read it. This is especially important in a shared-hosting environment.




Creating your own settings
There’s nothing stopping you from creating your own settings, for your own Django apps. Just follow these conventions:

•Setting names are in all uppercase.
•Don’t reinvent an already-existing setting.
For settings that are sequences, Django itself uses tuples, rather than lists, but this is only a convention.



Using settings without setting DJANGO_SETTINGS_MODULE
In some cases, you might want to bypass the DJANGO_SETTINGS_MODULE environment variable. For example, if you’re using the template system by itself, you likely don’t want to have to set up an environment variable pointing to a settings module.

In these cases, you can configure Django’s settings manually. Do this by calling:

django.conf.settings.configure(default_settings, **settings)
Example:

from django.conf import settings

settings.configure(DEBUG=True, TEMPLATE_DEBUG=True,
    TEMPLATE_DIRS=('/home/web-apps/myapp', '/home/web-apps/base'))
Pass configure() as many keyword arguments as you’d like, with each keyword argument representing a setting and its value. Each argument name should be all uppercase, with the same name as the settings described above. If a particular setting is not passed to configure() and is needed at some later point, Django will use the default setting value.

Configuring Django in this fashion is mostly necessary – and, indeed, recommended – when you’re using a piece of the framework inside a larger application.

Consequently, when configured via settings.configure(), Django will not make any modifications to the process environment variables (see the documentation of TIME_ZONE for why this would normally occur). It’s assumed that you’re already in full control of your environment in these cases.
















Applications
Django contains a registry of installed applications that stores configuration and provides introspection. It also maintains a list of available models.

This registry is simply called apps and it’s available in django.apps:

from django.apps import apps
apps.get_app_config('admin').verbose_name
'Admin'




Initialization process
How applications are loaded
When Django starts, django.setup() is responsible for populating the application registry.

setup()
Configures Django by:

•Loading the settings.
•Setting up logging.
•Initializing the application registry.

This function is called automatically:

•When running an HTTP server via Django’s WSGI support.
•When invoking a management command.

It must be called explicitly in other cases, for instance in plain Python scripts.

The application registry is initialized in three stages. At each stage, Django processes all applications in the order of INSTALLED_APPS.

1.First Django imports each item in INSTALLED_APPS.

If it’s an application configuration class, Django imports the root package of the application, defined by its name attribute. If it’s a Python package, Django creates a default application configuration.

At this stage, your code shouldn’t import any models!

In other words, your applications’ root packages and the modules that define your application configuration classes shouldn’t import any models, even indirectly.

Strictly speaking, Django allows importing models once their application configuration is loaded. However, in order to avoid needless constraints on the order of INSTALLED_APPS, it’s strongly recommended not import any models at this stage.

Once this stage completes, APIs that operate on application configurations such as get_app_config() become usable.

2.Then Django attempts to import the models submodule of each application, if there is one.

You must define or import all models in your application’s models.py or models/__init__.py. Otherwise, the application registry may not be fully populated at this point, which could cause the ORM to malfunction.

Once this stage completes, APIs that operate on models such as get_model() become usable.

3.Finally Django runs the ready() method of each application configuration.



Troubleshooting
Here are some common problems that you may encounter during initialization:

•AppRegistryNotReady This happens when importing an application configuration or a models module triggers code that depends on the app registry.

For example, ugettext() uses the app registry to look up translation catalogs in applications. To translate at import time, you need ugettext_lazy() instead. (Using ugettext() would be a bug, because the translation would happen at import time, rather than at each request depending on the active language.)

Executing database queries with the ORM at import time in models modules will also trigger this exception. The ORM cannot function properly until all models are available.

Another common culprit is django.contrib.auth.get_user_model(). Use the AUTH_USER_MODEL setting to reference the User model at import time.

This exception also happens if you forget to call django.setup() in a standalone Python script.

•ImportError: cannot import name ... This happens if the import sequence ends up in a loop.

To eliminate such problems, you should minimize dependencies between your models modules and do as little work as possible at import time. To avoid executing code at import time, you can move it into a function and cache its results. The code will be executed when you first need its results. This concept is known as “lazy evaluation”.

•django.contrib.admin automatically performs autodiscovery of admin modules in installed applications. To prevent it, change your INSTALLED_APPS to contain 'django.contrib.admin.apps.SimpleAdminConfig' instead of 'django.contrib.admin'.

























Django Exceptions
Django raises some Django specific exceptions as well as many standard Python exceptions.

Django Core Exceptions
Django core exception classes are defined in django.core.exceptions.

ObjectDoesNotExist and DoesNotExist
exception DoesNotExist
The DoesNotExist exception is raised when an object is not found for the given parameters of a query. Django provides a DoesNotExist exception as an attribute of each model class to identify the class of object that could not be found and to allow you to catch a particular model class with try/except.

exception ObjectDoesNotExist[source]
The base class for DoesNotExist exceptions; a try/except for ObjectDoesNotExist will catch DoesNotExist exceptions for all models.

See get() for further information on ObjectDoesNotExist and DoesNotExist.

MultipleObjectsReturned
exception MultipleObjectsReturned[source]
The MultipleObjectsReturned exception is raised by a query if only one object is expected, but multiple objects are returned. A base version of this exception is provided in django.core.exceptions; each model class contains a subclassed version that can be used to identify the specific object type that has returned multiple objects.

See get() for further information.

SuspiciousOperation
exception SuspiciousOperation[source]
The SuspiciousOperation exception is raised when a user has performed an operation that should be considered suspicious from a security perspective, such as tampering with a session cookie. Subclasses of SuspiciousOperation include:

•DisallowedHost
•DisallowedModelAdminLookup
•DisallowedModelAdminToField
•DisallowedRedirect
•InvalidSessionKey
•SuspiciousFileOperation
•SuspiciousMultipartForm
•SuspiciousSession
•WizardViewCookieModified
If a SuspiciousOperation exception reaches the WSGI handler level it is logged at the Error level and results in a HttpResponseBadRequest. See the logging documentation for more information.

PermissionDenied
exception PermissionDenied[source]
The PermissionDenied exception is raised when a user does not have permission to perform the action requested.

ViewDoesNotExist
exception ViewDoesNotExist[source]
The ViewDoesNotExist exception is raised by django.core.urlresolvers when a requested view does not exist.

MiddlewareNotUsed
exception MiddlewareNotUsed[source]
The MiddlewareNotUsed exception is raised when a middleware is not used in the server configuration.

ImproperlyConfigured
exception ImproperlyConfigured[source]
The ImproperlyConfigured exception is raised when Django is somehow improperly configured – for example, if a value in settings.py is incorrect or unparseable.

FieldError
exception FieldError[source]
The FieldError exception is raised when there is a problem with a model field. This can happen for several reasons:

•A field in a model clashes with a field of the same name from an abstract base class
•An infinite loop is caused by ordering
•A keyword cannot be parsed from the filter parameters
•A field cannot be determined from a keyword in the query parameters
•A join is not permitted on the specified field
•A field name is invalid
•A query contains invalid order_by arguments
ValidationError
exception ValidationError[source]
The ValidationError exception is raised when data fails form or model field validation. For more information about validation, see Form and Field Validation, Model Field Validation and the Validator Reference.

NON_FIELD_ERRORS
NON_FIELD_ERRORS
ValidationErrors that don’t belong to a particular field in a form or model are classified as NON_FIELD_ERRORS. This constant is used as a key in dictionaries that otherwise map fields to their respective list of errors.

URL Resolver exceptions
URL Resolver exceptions are defined in django.core.urlresolvers.

Resolver404
exception Resolver404[source]
The Resolver404 exception is raised by django.core.urlresolvers.resolve() if the path passed to resolve() doesn’t map to a view. It’s a subclass of django.http.Http404

NoReverseMatch
exception NoReverseMatch[source]
The NoReverseMatch exception is raised by django.core.urlresolvers when a matching URL in your URLconf cannot be identified based on the parameters supplied.

Database Exceptions
Database exceptions are provided in django.db.

Django wraps the standard database exceptions so that your Django code has a guaranteed common implementation of these classes.

exception Error
exception InterfaceError
exception DatabaseError
exception DataError
exception OperationalError
exception IntegrityError
exception InternalError
exception ProgrammingError
exception NotSupportedError
The Django wrappers for database exceptions behave exactly the same as the underlying database exceptions. See PEP 249, the Python Database API Specification v2.0, for further information.

As per PEP 3134, a __cause__ attribute is set with the original (underlying) database exception, allowing access to any additional information provided. (Note that this attribute is available under both Python 2 and Python 3, although PEP 3134 normally only applies to Python 3.)

Changed in Django 1.6: 
Previous versions of Django only wrapped DatabaseError and IntegrityError, and did not provide __cause__.

exception models.ProtectedError
Raised to prevent deletion of referenced objects when using django.db.models.PROTECT. models.ProtectedError is a subclass of IntegrityError.

Http Exceptions
Http exceptions are provided in django.http.

exception UnreadablePostError
The UnreadablePostError is raised when a user cancels an upload.

Transaction Exceptions
Transaction exceptions are defined in django.db.transaction.

exception TransactionManagementError[source]
The TransactionManagementError is raised for any and all problems related to database transactions.

Python Exceptions
Django raises built-in Python exceptions when appropriate as well. See the Python documentation for further information on the Built-in Exceptions.























django-admin.py and manage.py
django-admin.py is Django’s command-line utility for administrative tasks. T

In addition, manage.py is automatically created in each Django project. manage.py is a thin wrapper around django-admin.py that takes care of several things for you before delegating to django-admin.py:

•It puts your project’s package on sys.path.
•It sets the DJANGO_SETTINGS_MODULE environment variable so that it points to your project’s settings.py file.
•It calls django.setup() to initialize various internals of Django.


Generally, when working on a single Django project, it’s easier to use manage.py than django-admin.py. If you need to switch between multiple Django settings files, use django-admin.py with DJANGO_SETTINGS_MODULE or the --settings command line option.

The command-line examples throughout this document use django-admin.py to be consistent, but any example can use manage.py just as well.

Usage
$ django-admin.py <command> [options]
$ manage.py <command> [options]
command should be one of the commands listed in this document. options, which is optional, should be zero or more of the options available for the given command.



Getting runtime help
django-admin.py help
Run django-admin.py help to display usage information and a list of the commands provided by each application.

Run django-admin.py help --commands to display a list of all available commands.

Run django-admin.py help <command> to display a description of the given command and a list of its available options.



App names
Many commands take a list of “app names.” An “app name” is the basename of the package containing your models. For example, if your INSTALLED_APPS contains the string 'mysite.blog', the app name is blog.

Determining the version
django-admin.py version
Run django-admin.py version to display the current Django version.

The output follows the schema described in PEP 386:

1.4.dev17026
1.4a1
1.4


Displaying debug output
Use --verbosity to specify the amount of notification and debug information that django-admin.py should print to the console. For more details, see the documentation for the --verbosity option.

Available commands
check <appname appname ...>
django-admin.py check

Uses the system check framework to inspect the entire Django project for common problems.

The system check framework will confirm that there aren’t any problems with your installed models or your admin registrations. It will also provide warnings of common compatibility problems introduced by upgrading Django to a new version. Custom checks may be introduced by other libraries and applications.

By default, all apps will be checked. You can check a subset of apps by providing a list of app labels as arguments:

python manage.py check auth admin myapp
If you do not specify any app, all apps will be checked.

--tag <tagname>
The system check framework performs many different types of checks. These check types are categorized with tags. You can use these tags to restrict the checks performed to just those in a particular category. For example, to perform only security and compatibility checks, you would run:

python manage.py check --tag security --tag compatibility
--list-tags
List all available tags.




createcachetable
django-admin.py createcachetable
Creates the cache tables for use with the database cache backend. 

The --database option can be used to specify the database onto which the cachetable will be installed.



dbshell
django-admin.py dbshell
Runs the command-line client for the database engine specified in your ENGINE setting, with the connection parameters specified in your USER, PASSWORD, etc., settings.

•For PostgreSQL, this runs the psql command-line client.
•For MySQL, this runs the mysql command-line client.
•For SQLite, this runs the sqlite3 command-line client.
This command assumes the programs are on your PATH so that a simple call to the program name (psql, mysql, sqlite3) will find the program in the right place. There’s no way to specify the location of the program manually.

The --database option can be used to specify the database onto which to open a shell.



diffsettings
django-admin.py diffsettings
Displays differences between the current settings file and Django’s default settings.

Settings that don’t appear in the defaults are followed by "###". For example, the default settings don’t define ROOT_URLCONF, so ROOT_URLCONF is followed by "###" in the output of diffsettings.

The --all option may be provided to display all settings, even if they have Django’s default value. Such settings are prefixed by "###".


The --all option was added.



dumpdata <app_label app_label app_label.Model ...>
django-admin.py dumpdata
Outputs to standard output all data in the database associated with the named application(s).

If no application name is provided, all installed applications will be dumped.

The output of dumpdata can be used as input for loaddata.

Note that dumpdata uses the default manager on the model for selecting the records to dump. If you’re using a custom manager as the default manager and it filters some of the available records, not all of the objects will be dumped.

The --all option may be provided to specify that dumpdata should use Django’s base manager, dumping records which might otherwise be filtered or modified by a custom manager.

--format <fmt>
By default, dumpdata will format its output in JSON, but you can use the --format option to specify another format. Currently supported formats are listed in Serialization formats.

--indent <num>
By default, dumpdata will output all data on a single line. This isn’t easy for humans to read, so you can use the --indent option to pretty-print the output with a number of indentation spaces.

The --exclude option may be provided to prevent specific applications or models (specified as in the form of app_label.ModelName) from being dumped. If you specify a model name to dumpdata, the dumped output will be restricted to that model, rather than the entire application. You can also mix application names and model names.

The --database option can be used to specify the database from which data will be dumped.

--natural-foreign
New in Django 1.7. 
When this option is specified, Django will use the natural_key() model method to serialize any foreign key and many-to-many relationship to objects of the type that defines the method. If you are dumping contrib.auth Permission objects or contrib.contenttypes ContentType objects, you should probably be using this flag. See the natural keys documentation for more details on this and the next option.

--natural-primary
New in Django 1.7. 
When this option is specified, Django will not provide the primary key in the serialized data of this object since it can be calculated during deserialization.

--natural
Deprecated since version 1.7: 
Equivalent to the --natural-foreign option; use that instead.

Use natural keys to represent any foreign key and many-to-many relationship with a model that provides a natural key definition.

New in Django 1.6. 
--pks
By default, dumpdata will output all the records of the model, but you can use the --pks option to specify a comma separated list of primary keys on which to filter. This is only available when dumping one model.



flush
django-admin.py flush
Removes all data from the database, re-executes any post-synchronization handlers, and reinstalls any initial data fixtures.

The --noinput option may be provided to suppress all user prompts.

The --database option may be used to specify the database to flush.

--no-initial-data
Use --no-initial-data to avoid loading the initial_data fixture.



inspectdb
django-admin.py inspectdb
Introspects the database tables in the database pointed-to by the NAME setting and outputs a Django model module (a models.py file) to standard output.

Use this if you have a legacy database with which you’d like to use Django. The script will inspect the database and create a model for each table within it.

As you might expect, the created models will have an attribute for every field in the table. Note that inspectdb has a few special cases in its field-name output:

•If inspectdb cannot map a column’s type to a model field type, it’ll use TextField and will insert the Python comment 'This field type is a guess.' next to the field in the generated model.
•If the database column name is a Python reserved word (such as 'pass', 'class' or 'for'), inspectdb will append '_field' to the attribute name. For example, if a table has a column 'for', the generated model will have a field 'for_field', with the db_column attribute set to 'for'. inspectdb will insert the Python comment 'Field renamed because it was a Python reserved word.' next to the field.
This feature is meant as a shortcut, not as definitive model generation. After you run it, you’ll want to look over the generated models yourself to make customizations. In particular, you’ll need to rearrange models’ order, so that models that refer to other models are ordered properly.

Primary keys are automatically introspected for PostgreSQL, MySQL and SQLite, in which case Django puts in the primary_key=True where needed.

inspectdb works with PostgreSQL, MySQL and SQLite. Foreign-key detection only works in PostgreSQL and with certain types of MySQL tables.

Django doesn’t create database defaults when a default is specified on a model field. Similarly, database defaults aren’t translated to model field defaults or detected in any fashion by inspectdb.

By default, inspectdb creates unmanaged models. That is, managed = False in the model’s Meta class tells Django not to manage each table’s creation, modification, and deletion. If you do want to allow Django to manage the table’s lifecycle, you’ll need to change the managed option to True (or simply remove it because True is its default value).

The --database option may be used to specify the database to introspect.




loaddata <fixture fixture ...>
django-admin.py loaddata
Searches for and loads the contents of the named fixture into the database.

The --database option can be used to specify the database onto which the data will be loaded.

--ignorenonexistent
The --ignorenonexistent option can be used to ignore fields that may have been removed from models since the fixture was originally generated.

--app
The --app option can be used to specify a single app to look for fixtures in rather than looking through all apps.


--app was added.

What’s a “fixture”?
A fixture is a collection of files that contain the serialized contents of the database. Each fixture has a unique name, and the files that comprise the fixture can be distributed over multiple directories, in multiple applications.

Django will search in three locations for fixtures:

1.In the fixtures directory of every installed application
2.In any directory named in the FIXTURE_DIRS setting
3.In the literal path named by the fixture
Django will load any and all fixtures it finds in these locations that match the provided fixture names.

If the named fixture has a file extension, only fixtures of that type will be loaded. For example:

django-admin.py loaddata mydata.json
would only load JSON fixtures called mydata. The fixture extension must correspond to the registered name of a serializer (e.g., json or xml).

If you omit the extensions, Django will search all available fixture types for a matching fixture. For example:

django-admin.py loaddata mydata
would look for any fixture of any fixture type called mydata. If a fixture directory contained mydata.json, that fixture would be loaded as a JSON fixture.

The fixtures that are named can include directory components. These directories will be included in the search path. For example:

django-admin.py loaddata foo/bar/mydata.json
would search <app_label>/fixtures/foo/bar/mydata.json for each installed application, <dirname>/foo/bar/mydata.json for each directory in FIXTURE_DIRS, and the literal path foo/bar/mydata.json.

When fixture files are processed, the data is saved to the database as is. Model defined save() methods are not called, and any pre_save or post_save signals will be called with raw=True since the instance only contains attributes that are local to the model. You may, for example, want to disable handlers that access related fields that aren’t present during fixture loading and would otherwise raise an exception:

from django.db.models.signals import post_save
from .models import MyModel

def my_handler(**kwargs):
    # disable the handler during fixture loading
    if kwargs['raw']:
        return
    ...

post_save.connect(my_handler, sender=MyModel)
You could also write a simple decorator to encapsulate this logic:

from functools import wraps

def disable_for_loaddata(signal_handler):
    """
    Decorator that turns off signal handlers when loading fixture data.
    """
    @wraps(signal_handler)
    def wrapper(*args, **kwargs):
        if kwargs['raw']:
            return
        signal_handler(*args, **kwargs)
    return wrapper

@disable_for_loaddata
def my_handler(**kwargs):
    ...
Just be aware that this logic will disable the signals whenever fixtures are deserialized, not just during loaddata.

Note that the order in which fixture files are processed is undefined. However, all fixture data is installed as a single transaction, so data in one fixture can reference data in another fixture. If the database backend supports row-level constraints, these constraints will be checked at the end of the transaction.

The dumpdata command can be used to generate input for loaddata.

Compressed fixtures
Fixtures may be compressed in zip, gz, or bz2 format. For example:

django-admin.py loaddata mydata.json
would look for any of mydata.json, mydata.json.zip, mydata.json.gz, or mydata.json.bz2. The first file contained within a zip-compressed archive is used.

Note that if two fixtures with the same name but different fixture type are discovered (for example, if mydata.json and mydata.xml.gz were found in the same fixture directory), fixture installation will be aborted, and any data installed in the call to loaddata will be removed from the database.

MySQL with MyISAM and fixtures

The MyISAM storage engine of MySQL doesn’t support transactions or constraints, so if you use MyISAM, you won’t get validation of fixture data, or a rollback if multiple transaction files are found.

Database-specific fixtures
If you’re in a multi-database setup, you might have fixture data that you want to load onto one database, but not onto another. In this situation, you can add database identifier into the names of your fixtures.

For example, if your DATABASES setting has a ‘master’ database defined, name the fixture mydata.master.json or mydata.master.json.gz and the fixture will only be loaded when you specify you want to load data into the master database.



makemessages
django-admin.py makemessages
Runs over the entire source tree of the current directory and pulls out all strings marked for translation. It creates (or updates) a message file in the conf/locale (in the Django tree) or locale (for project and application) directory. After making changes to the messages files you need to compile them with compilemessages for use with the builtin gettext support. See the i18n documentation for details.

--all
Use the --all or -a option to update the message files for all available languages.

Example usage:

django-admin.py makemessages --all
--extension
Use the --extension or -e option to specify a list of file extensions to examine (default: ”.html”, ”.txt”).

Example usage:

django-admin.py makemessages --locale=de --extension xhtml
Separate multiple extensions with commas or use -e or –extension multiple times:

django-admin.py makemessages --locale=de --extension=html,txt --extension xml
Use the --locale option (or its shorter version -l) to specify the locale(s) to process.

Example usage:

django-admin.py makemessages --locale=pt_BR
django-admin.py makemessages --locale=pt_BR --locale=fr
django-admin.py makemessages -l pt_BR
django-admin.py makemessages -l pt_BR -l fr
Changed in Django 1.6: 
Added the ability to specify multiple locales.


Added the --previous option to the msgmerge command when merging with existing po files.

--domain
Use the --domain or -d option to change the domain of the messages files. Currently supported:

•django for all *.py, *.html and *.txt files (default)
•djangojs for *.js files
--symlinks
Use the --symlinks or -s option to follow symlinks to directories when looking for new translation strings.

Example usage:

django-admin.py makemessages --locale=de --symlinks
--ignore
Use the --ignore or -i option to ignore files or directories matching the given glob-style pattern. Use multiple times to ignore more.

These patterns are used by default: 'CVS', '.*', '*~', '*.pyc'

Example usage:

django-admin.py makemessages --locale=en_US --ignore=apps/* --ignore=secret/*.html
--no-default-ignore
Use the --no-default-ignore option to disable the default values of --ignore.

--no-wrap
Use the --no-wrap option to disable breaking long message lines into several lines in language files.

--no-location
Use the --no-location option to not write ‘#: filename:line’ comment lines in language files. Note that using this option makes it harder for technically skilled translators to understand each message’s context.

--keep-pot
New in Django 1.6. 
Use the --keep-pot option to prevent Django from deleting the temporary .pot files it generates before creating the .po file. This is useful for debugging errors which may prevent the final language files from being created.



makemigrations [<app_label>]
django-admin.py makemigrations

Creates new migrations based on the changes detected to your models. Migrations, their relationship with apps and more are covered in depth in the migrations documentation.

Providing one or more app names as arguments will limit the migrations created to the app(s) specified and any dependencies needed (the table at the other end of a ForeignKey, for example).

--empty
The --empty option will cause makemigrations to output an empty migration for the specified apps, for manual editing. This option is only for advanced users and should not be used unless you are familiar with the migration format, migration operations, and the dependencies between your migrations.

--dry-run
The --dry-run option shows what migrations would be made without actually writing any migrations files to disk. Using this option along with --verbosity 3 will also show the complete migrations files that would be written.

--merge
The --merge option enables fixing of migration conflicts. The --noinput option may be provided to suppress user prompts during a merge.



migrate [<app_label> [<migrationname>]]
django-admin.py migrate

Synchronizes the database state with the current set of models and migrations. Migrations, their relationship with apps and more are covered in depth in the migrations documentation.

The behavior of this command changes depending on the arguments provided:

•No arguments: All migrated apps have all of their migrations run, and all unmigrated apps are synchronized with the database,
•<app_label>: The specified app has its migrations run, up to the most recent migration. This may involve running other apps’ migrations too, due to dependencies.
•<app_label> <migrationname>: Brings the database schema to a state where the named migration is applied, but no later migrations in the same app are applied. This may involve unapplying migrations if you have previously migrated past the named migration. Use the name zero to unapply all migrations for an app.
Unlike syncdb, this command does not prompt you to create a superuser if one doesn’t exist (assuming you are using django.contrib.auth). Use createsuperuser to do that if you wish.

The --database option can be used to specify the database to migrate.

--fake
The --fake option tells Django to mark the migrations as having been applied or unapplied, but without actually running the SQL to change your database schema.

This is intended for advanced users to manipulate the current migration state directly if they’re manually applying changes; be warned that using --fake runs the risk of putting the migration state table into a state where manual recovery will be needed to make migrations run correctly.

--list, -l
The --list option will list all of the apps Django knows about, the migrations available for each app and if they are applied or not (marked by an [X] next to the migration name).

Apps without migrations are also included in the list, but will have (no migrations) printed under them.




runserver [port or address:port]
django-admin.py runserver
Starts a lightweight development Web server on the local machine. By default, the server runs on port 8000 on the IP address 127.0.0.1. You can pass in an IP address and port number explicitly.

If you run this script as a user with normal privileges (recommended), you might not have access to start a port on a low port number. Low port numbers are reserved for the superuser (root).

This server uses the WSGI application object specified by the WSGI_APPLICATION setting.

DO NOT USE THIS SERVER IN A PRODUCTION SETTING. It has not gone through security audits or performance tests. (And that’s how it’s gonna stay. We’re in the business of making Web frameworks, not Web servers, so improving this server to be able to handle a production environment is outside the scope of Django.)

The development server automatically reloads Python code for each request, as needed. You don’t need to restart the server for code changes to take effect. However, some actions like adding files don’t trigger a restart, so you’ll have to restart the server in these cases.

Changed in Django 1.7: 
Compiling translation files now also restarts the development server.

If you are using Linux and install pyinotify, kernel signals will be used to autoreload the server (rather than polling file modification timestamps each second). This offers better scaling to large projects, reduction in response time to code modification, more robust change detection, and battery usage reduction.

New in Django 1.7: 
pyinotify support was added.

When you start the server, and each time you change Python code while the server is running, the system check framework will check your entire Django project for some common errors (see the check command). If any errors are found, they will be printed to standard output.

You can run as many servers as you want, as long as they’re on separate ports. Just execute django-admin.py runserver more than once.

Note that the default IP address, 127.0.0.1, is not accessible from other machines on your network. To make your development server viewable to other machines on the network, use its own IP address (e.g. 192.168.2.1) or 0.0.0.0 or :: (with IPv6 enabled).

You can provide an IPv6 address surrounded by brackets (e.g. [200a::1]:8000). This will automatically enable IPv6 support.

A hostname containing ASCII-only characters can also be used.

If the staticfiles contrib app is enabled (default in new projects) the runserver command will be overridden with its own runserver command.

If migrate was not previously executed, the table that stores the history of migrations is created at first run of runserver.

--noreload
Use the --noreload option to disable the use of the auto-reloader. This means any Python code changes you make while the server is running will not take effect if the particular Python modules have already been loaded into memory.

Example usage:

django-admin.py runserver --noreload
--nothreading
The development server is multithreaded by default. Use the --nothreading option to disable the use of threading in the development server.

--ipv6, -6
Use the --ipv6 (or shorter -6) option to tell Django to use IPv6 for the development server. This changes the default IP address from 127.0.0.1 to ::1.

Example usage:

django-admin.py runserver --ipv6
Examples of using different ports and addresses
Port 8000 on IP address 127.0.0.1:

django-admin.py runserver
Port 8000 on IP address 1.2.3.4:

django-admin.py runserver 1.2.3.4:8000
Port 7000 on IP address 127.0.0.1:

django-admin.py runserver 7000
Port 7000 on IP address 1.2.3.4:

django-admin.py runserver 1.2.3.4:7000
Port 8000 on IPv6 address ::1:

django-admin.py runserver -6
Port 7000 on IPv6 address ::1:

django-admin.py runserver -6 7000
Port 7000 on IPv6 address 2001:0db8:1234:5678::9:

django-admin.py runserver [2001:0db8:1234:5678::9]:7000
Port 8000 on IPv4 address of host localhost:

django-admin.py runserver localhost:8000
Port 8000 on IPv6 address of host localhost:

django-admin.py runserver -6 localhost:8000


Serving static files with the development server
By default, the development server doesn’t serve any static files for your site (such as CSS files, images, things under MEDIA_URL and so forth). If you want to configure Django to serve static media, read Managing static files (CSS, images).



shell
django-admin.py shell
Starts the Python interactive interpreter.

Django will use IPython or bpython if either is installed. If you have a rich shell installed but want to force use of the “plain” Python interpreter, use the --plain option, like so:

django-admin.py shell --plain
If you would like to specify either IPython or bpython as your interpreter if you have both installed you can specify an alternative interpreter interface with the -i or --interface options like so:

IPython:

django-admin.py shell -i ipython
django-admin.py shell --interface ipython
bpython:

django-admin.py shell -i bpython
django-admin.py shell --interface bpython
When the “plain” Python interactive interpreter starts (be it because --plain was specified or because no other interactive interface is available) it reads the script pointed to by the PYTHONSTARTUP environment variable and the ~/.pythonrc.py script. If you don’t wish this behavior you can use the --no-startup option. e.g.:

django-admin.py shell --plain --no-startup
New in Django 1.6: 
The --no-startup option was added in Django 1.6.



sql <app_label app_label ...>
django-admin.py sql
Prints the CREATE TABLE SQL statements for the given app name(s).

The --database option can be used to specify the database for which to print the SQL.



sqlall <app_label app_label ...>
django-admin.py sqlall
Prints the CREATE TABLE and initial-data SQL statements for the given app name(s).

Refer to the description of sqlcustom for an explanation of how to specify initial data.

The --database option can be used to specify the database for which to print the SQL.

Changed in Django 1.7: 
The sql* management commands now respect the allow_migrate() method of DATABASE_ROUTERS. If you have models synced to non-default databases, use the --database flag to get SQL for those models (previously they would always be included in the output).



sqlclear <app_label app_label ...>
django-admin.py sqlclear
Prints the DROP TABLE SQL statements for the given app name(s).

The --database option can be used to specify the database for which to print the SQL.



sqlcustom <app_label app_label ...>
django-admin.py sqlcustom
Prints the custom SQL statements for the given app name(s).

For each model in each specified app, this command looks for the file <app_label>/sql/<modelname>.sql, where <app_label> is the given app name and <modelname> is the model’s name in lowercase. For example, if you have an app news that includes a Story model, sqlcustom will attempt to read a file news/sql/story.sql and append it to the output of this command.

Each of the SQL files, if given, is expected to contain valid SQL. The SQL files are piped directly into the database after all of the models’ table-creation statements have been executed. Use this SQL hook to make any table modifications, or insert any SQL functions into the database.

Note that the order in which the SQL files are processed is undefined.

The --database option can be used to specify the database for which to print the SQL.



sqldropindexes <app_label app_label ...>
django-admin.py sqldropindexes
New in Django 1.6. 
Prints the DROP INDEX SQL statements for the given app name(s).

The --database option can be used to specify the database for which to print the SQL.



sqlflush
django-admin.py sqlflush
Prints the SQL statements that would be executed for the flush command.

The --database option can be used to specify the database for which to print the SQL.



sqlindexes <app_label app_label ...>
django-admin.py sqlindexes
Prints the CREATE INDEX SQL statements for the given app name(s).

The --database option can be used to specify the database for which to print the SQL.



sqlmigrate <app_label> <migrationname>
django-admin.py sqlmigrate
Prints the SQL for the named migration. This requires an active database connection, which it will use to resolve constraint names; this means you must generate the SQL against a copy of the database you wish to later apply it on.

Note that sqlmigrate doesn’t colorize its output.

The --database option can be used to specify the database for which to generate the SQL.

--backwards
By default, the SQL created is for running the migration in the forwards direction. Pass --backwards to generate the SQL for unapplying the migration instead.


sqlsequencereset <app_label app_label ...>
django-admin.py sqlsequencereset
Prints the SQL statements for resetting sequences for the given app name(s).

Sequences are indexes used by some database engines to track the next available number for automatically incremented fields.

Use this command to generate SQL which will fix cases where a sequence is out of sync with its automatically incremented field data.

The --database option can be used to specify the database for which to print the SQL.



squashmigrations <app_label> <migration_name>
django-admin.py squashmigrations
Squashes the migrations for app_label up to and including migration_name down into fewer migrations, if possible. The resulting squashed migrations can live alongside the unsquashed ones safely. For more information, please read Squashing migrations.

--no-optimize
By default, Django will try to optimize the operations in your migrations to reduce the size of the resulting file. Pass --no-optimize if this process is failing for you or creating incorrect migrations, though please also file a Django bug report about the behavior, as optimization is meant to be safe.



startapp <app_label> [destination]
django-admin.py startapp
Creates a Django app directory structure for the given app name in the current directory or the given destination.

By default the directory created contains a models.py file and other app template files. (See the source for more details.) If only the app name is given, the app directory will be created in the current working directory.

If the optional destination is provided, Django will use that existing directory rather than creating a new one. You can use ‘.’ to denote the current working directory.

For example:

django-admin.py startapp myapp /Users/jezdez/Code/myapp
--template
With the --template option, you can use a custom app template by providing either the path to a directory with the app template file, or a path to a compressed file (.tar.gz, .tar.bz2, .tgz, .tbz, .zip) containing the app template files.

For example, this would look for an app template in the given directory when creating the myapp app:

django-admin.py startapp --template=/Users/jezdez/Code/my_app_template myapp
Django will also accept URLs (http, https, ftp) to compressed archives with the app template files, downloading and extracting them on the fly.

For example, taking advantage of Github’s feature to expose repositories as zip files, you can use a URL like:

django-admin.py startapp --template=https://github.com/githubuser/django-app-template/archive/master.zip myapp
When Django copies the app template files, it also renders certain files through the template engine: the files whose extensions match the --extension option (py by default) and the files whose names are passed with the --name option. The template context used is:

•Any option passed to the startapp command (among the command’s supported options)
•app_name – the app name as passed to the command
•app_directory – the full path of the newly created app
•docs_version – the version of the documentation: 'dev' or '1.x'
Warning

When the app template files are rendered with the Django template engine (by default all *.py files), Django will also replace all stray template variables contained. For example, if one of the Python files contains a docstring explaining a particular feature related to template rendering, it might result in an incorrect example.

To work around this problem, you can use the templatetag templatetag to “escape” the various parts of the template syntax.



startproject <projectname> [destination]
django-admin.py startproject
Creates a Django project directory structure for the given project name in the current directory or the given destination.

By default, the new directory contains manage.py and a project package (containing a settings.py and other files). See the template source for details.

If only the project name is given, both the project directory and project package will be named <projectname> and the project directory will be created in the current working directory.

If the optional destination is provided, Django will use that existing directory as the project directory, and create manage.py and the project package within it. Use ‘.’ to denote the current working directory.

For example:

django-admin.py startproject myproject /Users/jezdez/Code/myproject_repo
As with the startapp command, the --template option lets you specify a directory, file path or URL of a custom project template. See the startapp documentation for details of supported project template formats.

For example, this would look for a project template in the given directory when creating the myproject project:

django-admin.py startproject --template=/Users/jezdez/Code/my_project_template myproject
Django will also accept URLs (http, https, ftp) to compressed archives with the project template files, downloading and extracting them on the fly.

For example, taking advantage of Github’s feature to expose repositories as zip files, you can use a URL like:

django-admin.py startproject --template=https://github.com/githubuser/django-project-template/archive/master.zip myproject
When Django copies the project template files, it also renders certain files through the template engine: the files whose extensions match the --extension option (py by default) and the files whose names are passed with the --name option. The template context used is:

•Any option passed to the startproject command (among the command’s supported options)
•project_name – the project name as passed to the command
•project_directory – the full path of the newly created project
•secret_key – a random key for the SECRET_KEY setting
•docs_version – the version of the documentation: 'dev' or '1.x'
Please also see the rendering warning as mentioned for startapp.





test <app or test identifier>
django-admin.py test
Runs tests for all installed models. See Testing in Django for more information.

--failfast
The --failfast option can be used to stop running tests and report the failure immediately after a test fails.

--testrunner
The --testrunner option can be used to control the test runner class that is used to execute tests. If this value is provided, it overrides the value provided by the TEST_RUNNER setting.

--liveserver
The --liveserver option can be used to override the default address where the live server (used with LiveServerTestCase) is expected to run from. The default value is localhost:8081.




testserver <fixture fixture ...>
django-admin.py testserver
Runs a Django development server (as in runserver) using data from the given fixture(s).

For example, this command:

django-admin.py testserver mydata.json
...would perform the following steps:

1.Create a test database, as described in The test database.
2.Populate the test database with fixture data from the given fixtures. (For more on fixtures, see the documentation for loaddata above.)
3.Runs the Django development server (as in runserver), pointed at this newly created test database instead of your production database.
This is useful in a number of ways:

•When you’re writing unit tests of how your views act with certain fixture data, you can use testserver to interact with the views in a Web browser, manually.
•Let’s say you’re developing your Django application and have a “pristine” copy of a database that you’d like to interact with. You can dump your database to a fixture (using the dumpdata command, explained above), then use testserver to run your Web application with that data. With this arrangement, you have the flexibility of messing up your data in any way, knowing that whatever data changes you’re making are only being made to a test database.
Note that this server does not automatically detect changes to your Python source code (as runserver does). It does, however, detect changes to templates.

--addrport [port number or ipaddr:port]
Use --addrport to specify a different port, or IP address and port, from the default of 127.0.0.1:8000. This value follows exactly the same format and serves exactly the same function as the argument to the runserver command.

Examples:

To run the test server on port 7000 with fixture1 and fixture2:

django-admin.py testserver --addrport 7000 fixture1 fixture2
django-admin.py testserver fixture1 fixture2 --addrport 7000
(The above statements are equivalent. We include both of them to demonstrate that it doesn’t matter whether the options come before or after the fixture arguments.)

To run on 1.2.3.4:7000 with a test fixture:

django-admin.py testserver --addrport 1.2.3.4:7000 test
The --noinput option may be provided to suppress all user prompts.

validate
django-admin.py validate
Deprecated since version 1.7: 
Replaced by the check command.

Validates all installed models (according to the INSTALLED_APPS setting) and prints validation errors to standard output.






django.contrib.auth
changepassword
django-admin.py changepassword
This command is only available if Django’s authentication system (django.contrib.auth) is installed.

Allows changing a user’s password. It prompts you to enter twice the password of the user given as parameter. If they both match, the new password will be changed immediately. If you do not supply a user, the command will attempt to change the password whose username matches the current user.

Use the --database option to specify the database to query for the user. If it’s not supplied, Django will use the default database.

Example usage:

django-admin.py changepassword ringo
createsuperuser
django-admin.py createsuperuser
This command is only available if Django’s authentication system (django.contrib.auth) is installed.

Creates a superuser account (a user who has all permissions). This is useful if you need to create an initial superuser account or if you need to programmatically generate superuser accounts for your site(s).

When run interactively, this command will prompt for a password for the new superuser account. When run non-interactively, no password will be set, and the superuser account will not be able to log in until a password has been manually set for it.

--username
--email
The username and email address for the new account can be supplied by using the --username and --email arguments on the command line. If either of those is not supplied, createsuperuser will prompt for it when running interactively.

Use the --database option to specify the database into which the superuser object will be saved.



django.contrib.gis
ogrinspect
This command is only available if GeoDjango (django.contrib.gis) is installed.

Please refer to its description in the GeoDjango documentation.



django.contrib.sessions
clearsessions
django-admin.py clearsessions
Can be run as a cron job or directly to clean out expired sessions.

django.contrib.sitemaps
ping_google
This command is only available if the Sitemaps framework (django.contrib.sitemaps) is installed.

Please refer to its description in the Sitemaps documentation.

django.contrib.staticfiles
collectstatic
This command is only available if the static files application (django.contrib.staticfiles) is installed.

Please refer to its description in the staticfiles documentation.

findstatic
This command is only available if the static files application (django.contrib.staticfiles) is installed.

Please refer to its description in the staticfiles documentation.

Default options
Although some commands may allow their own custom options, every command allows for the following options:

--pythonpath
Example usage:

django-admin.py migrate --pythonpath='/home/djangoprojects/myproject'
Adds the given filesystem path to the Python import search path. If this isn’t provided, django-admin.py will use the PYTHONPATH environment variable.

Note that this option is unnecessary in manage.py, because it takes care of setting the Python path for you.

--settings
Example usage:

django-admin.py migrate --settings=mysite.settings
Explicitly specifies the settings module to use. The settings module should be in Python package syntax, e.g. mysite.settings. If this isn’t provided, django-admin.py will use the DJANGO_SETTINGS_MODULE environment variable.

Note that this option is unnecessary in manage.py, because it uses settings.py from the current project by default.

--traceback
Example usage:

django-admin.py migrate --traceback
By default, django-admin.py will show a simple error message whenever an CommandError occurs, but a full stack trace for any other exception. If you specify --traceback, django-admin.py will also output a full stack trace when a CommandError is raised.

Changed in Django 1.6: 
Previously, Django didn’t show a full stack trace by default for exceptions other than CommandError.

--verbosity
Example usage:

django-admin.py migrate --verbosity 2
Use --verbosity to specify the amount of notification and debug information that django-admin.py should print to the console.

•0 means no output.
•1 means normal output (default).
•2 means verbose output.
•3 means very verbose output.
--no-color
New in Django 1.7. 
Example usage:

django-admin.py sqlall --no-color
By default, django-admin.py will format the output to be colorized. For example, errors will be printed to the console in red and SQL statements will be syntax highlighted. To prevent this and have a plain text output, pass the --no-color option when running your command.

Common options
The following options are not available on every command, but they are common to a number of commands.

--database
Used to specify the database on which a command will operate. If not specified, this option will default to an alias of default.

For example, to dump data from the database with the alias master:

django-admin.py dumpdata --database=master
--exclude
Exclude a specific application from the applications whose contents is output. For example, to specifically exclude the auth application from the output of dumpdata, you would call:

django-admin.py dumpdata --exclude=auth
If you want to exclude multiple applications, use multiple --exclude directives:

django-admin.py dumpdata --exclude=auth --exclude=contenttypes
--locale
Use the --locale or -l option to specify the locale to process. If not provided all locales are processed.

--noinput
Use the --noinput option to suppress all user prompting, such as “Are you sure?” confirmation messages. This is useful if django-admin.py is being executed as an unattended, automated script.

E



















Writing tests
Django’s unit tests use a Python standard library module: unittest. This module defines tests using a class-based approach.


from django.test import TestCase
from myapp.models import Animal

class AnimalTestCase(TestCase):
    def setUp(self):
        Animal.objects.create(name="lion", sound="roar")
        Animal.objects.create(name="cat", sound="meow")

    def test_animals_can_speak(self):
        """Animals that can speak are correctly identified"""
        lion = Animal.objects.get(name="lion")
        cat = Animal.objects.get(name="cat")
        self.assertEqual(lion.speak(), 'The lion says "roar"')
        self.assertEqual(cat.speak(), 'The cat says "meow"')
When you run your tests, the default behavior of the test utility is to find all the test cases (that is, subclasses of unittest.TestCase) in any file whose name begins with test, automatically build a test suite out of those test cases, and run that suite.




Running tests
Once you’ve written tests, run them using the test command of your project’s manage.py utility:

$ ./manage.py test
Test discovery is based on the unittest module’s built-in test discovery. By default, this will discover tests in any file named “test*.py” under the current working directory.

You can specify particular tests to run by supplying any number of “test labels” to ./manage.py test. Each test label can be a full Python dotted path to a package, module, TestCase subclass, or test method. For instance:

# Run all the tests in the animals.tests module
$ ./manage.py test animals.tests

# Run all the tests found within the 'animals' package
$ ./manage.py test animals

# Run just one test case
$ ./manage.py test animals.tests.AnimalTestCase

# Run just one test method
$ ./manage.py test animals.tests.AnimalTestCase.test_animals_can_speak
You can also provide a path to a directory to discover tests below that directory:

$ ./manage.py test animals/
You can specify a custom filename pattern match using the -p (or --pattern) option, if your test files are named differently from the test*.py pattern:

$ ./manage.py test --pattern="tests_*.py"
Previously, test labels were in the form applabel, applabel.TestCase, or applabel.TestCase.test_method, rather than being true Python dotted paths, and tests could only be found within tests.py or models.py files within a Python package listed in INSTALLED_APPS. The --pattern option and file paths as test labels are new in 1.6.

If you press Ctrl-C while the tests are running, the test runner will wait for the currently running test to complete and then exit gracefully. During a graceful exit the test runner will output details of any test failures, report on how many tests were run and how many errors and failures were encountered, and destroy any test databases as usual. Thus pressing Ctrl-C can be very useful if you forget to pass the --failfast option, notice that some tests are unexpectedly failing, and want to get details on the failures without waiting for the full test run to complete.

If you do not want to wait for the currently running test to finish, you can press Ctrl-C a second time and the test run will halt immediately, but not gracefully. No details of the tests run before the interruption will be reported, and any test databases created by the run will not be destroyed.




















Testing tools

The test client
The test client is a Python class that acts as a dummy Web browser, allowing you to test your views and interact with your Django-powered application programmatically.

Some of the things you can do with the test client are:

•Simulate GET and POST requests on a URL and observe the response – everything from low-level HTTP (result headers and status codes) to page content.
•See the chain of redirects (if any) and check the URL and status code at each step.
•Test that a given request is rendered by a given Django template, with a template context that contains certain values.

Note that the test client is not intended to be a replacement for Selenium or other “in-browser” frameworks. Django’s test client has a different focus. In short:

•Use Django’s test client to establish that the correct template is being rendered and that the template is passed the correct context data.
•Use in-browser frameworks like Selenium to test rendered HTML and the behavior of Web pages, namely JavaScript functionality. Django also provides special support for those frameworks; see the section on LiveServerTestCase for more details.
A comprehensive test suite should use a combination of both test types.

To use the test client, instantiate django.test.Client and retrieve Web pages:

from django.test import Client
c = Client()
response = c.post('/login/', {'username': 'john', 'password': 'smith'})
response.status_code
200
response = c.get('/customer/details/')
response.content
'<!DOCTYPE html...'


As this example suggests, you can instantiate Client from within a session of the Python interactive interpreter.

Note a few important things about how the test client works:

•The test client does not require the Web server to be running. In fact, it will run just fine with no Web server running at all! That’s because it avoids the overhead of HTTP and deals directly with the Django framework. This helps make the unit tests run quickly.

•When retrieving pages, remember to specify the path of the URL, not the whole domain. For example, this is correct:

c.get('/login/')
This is incorrect:

c.get('http://www.example.com/login/')
The test client is not capable of retrieving Web pages that are not powered by your Django project. If you need to retrieve other Web pages, use a Python standard library module such as urllib.

•To resolve URLs, the test client uses whatever URLconf is pointed-to by your ROOT_URLCONF setting.

•Although the above example would work in the Python interactive interpreter, some of the test client’s functionality, notably the template-related functionality, is only available while tests are running.

The reason for this is that Django’s test runner performs a bit of black magic in order to determine which template was loaded by a given view. This black magic (essentially a patching of Django’s template system in memory) only happens during test running.

•By default, the test client will disable any CSRF checks performed by your site.

If, for some reason, you want the test client to perform CSRF checks, you can create an instance of the test client that enforces CSRF checks. To do this, pass in the enforce_csrf_checks argument when you construct your client:

from django.test import Client
csrf_client = Client(enforce_csrf_checks=True)
Making requests
Use the django.test.Client class to make requests.

class Client(enforce_csrf_checks=False, **defaults)
It requires no arguments at time of construction. However, you can use keywords arguments to specify some default headers. For example, this will send a User-Agent HTTP header in each request:

c = Client(HTTP_USER_AGENT='Mozilla/5.0')
The values from the extra keywords arguments passed to get(), post(), etc. have precedence over the defaults passed to the class constructor.

The enforce_csrf_checks argument can be used to test CSRF protection (see above).

Once you have a Client instance, you can call any of the following methods:

get(path, data=None, follow=False, secure=False, **extra)
New in Django 1.7: 
The secure argument was added.

Makes a GET request on the provided path and returns a Response object, which is documented below.

The key-value pairs in the data dictionary are used to create a GET data payload. For example:

c = Client()
c.get('/customers/details/', {'name': 'fred', 'age': 7})
...will result in the evaluation of a GET request equivalent to:

/customers/details/?name=fred&age=7
The extra keyword arguments parameter can be used to specify headers to be sent in the request. For example:

c = Client()
c.get('/customers/details/', {'name': 'fred', 'age': 7},
...       HTTP_X_REQUESTED_WITH='XMLHttpRequest')
...will send the HTTP header HTTP_X_REQUESTED_WITH to the details view, which is a good way to test code paths that use the django.http.HttpRequest.is_ajax() method.



CGI specification

The headers sent via **extra should follow CGI specification. For example, emulating a different “Host” header as sent in the HTTP request from the browser to the server should be passed as HTTP_HOST.

If you already have the GET arguments in URL-encoded form, you can use that encoding instead of using the data argument. For example, the previous GET request could also be posed as:

c = Client()
c.get('/customers/details/?name=fred&age=7')
If you provide a URL with both an encoded GET data and a data argument, the data argument will take precedence.

If you set follow to True the client will follow any redirects and a redirect_chain attribute will be set in the response object containing tuples of the intermediate urls and status codes.

If you had a URL /redirect_me/ that redirected to /next/, that redirected to /final/, this is what you’d see:

response = c.get('/redirect_me/', follow=True)
response.redirect_chain
[(u'http://testserver/next/', 302), (u'http://testserver/final/', 302)]
If you set secure to True the client will emulate an HTTPS request.

post(path, data=None, content_type=MULTIPART_CONTENT, follow=False, secure=False, **extra)
Makes a POST request on the provided path and returns a Response object, which is documented below.

The key-value pairs in the data dictionary are used to submit POST data. For example:

c = Client()
c.post('/login/', {'name': 'fred', 'passwd': 'secret'})
...will result in the evaluation of a POST request to this URL:

/login/
...with this POST data:

name=fred&passwd=secret
If you provide content_type (e.g. text/xml for an XML payload), the contents of data will be sent as-is in the POST request, using content_type in the HTTP Content-Type header.

If you don’t provide a value for content_type, the values in data will be transmitted with a content type of multipart/form-data. In this case, the key-value pairs in data will be encoded as a multipart message and used to create the POST data payload.

To submit multiple values for a given key – for example, to specify the selections for a <select multiple> – provide the values as a list or tuple for the required key. For example, this value of data would submit three selected values for the field named choices:

{'choices': ('a', 'b', 'd')}
Submitting files is a special case. To POST a file, you need only provide the file field name as a key, and a file handle to the file you wish to upload as a value. For example:

c = Client()
with open('wishlist.doc') as fp:
...     c.post('/customers/wishes/', {'name': 'fred', 'attachment': fp})
(The name attachment here is not relevant; use whatever name your file-processing code expects.)

Note that if you wish to use the same file handle for multiple post() calls then you will need to manually reset the file pointer between posts. The easiest way to do this is to manually close the file after it has been provided to post(), as demonstrated above.

You should also ensure that the file is opened in a way that allows the data to be read. If your file contains binary data such as an image, this means you will need to open the file in rb (read binary) mode.

The extra argument acts the same as for Client.get().

If the URL you request with a POST contains encoded parameters, these parameters will be made available in the request.GET data. For example, if you were to make the request:

c.post('/login/?visitor=true', {'name': 'fred', 'passwd': 'secret'})
... the view handling this request could interrogate request.POST to retrieve the username and password, and could interrogate request.GET to determine if the user was a visitor.

If you set follow to True the client will follow any redirects and a redirect_chain attribute will be set in the response object containing tuples of the intermediate urls and status codes.

If you set secure to True the client will emulate an HTTPS request.

head(path, data=None, follow=False, secure=False, **extra)
Makes a HEAD request on the provided path and returns a Response object. This method works just like Client.get(), including the follow, secure and extra arguments, except it does not return a message body.

options(path, data='', content_type='application/octet-stream', follow=False, secure=False, **extra)
Makes an OPTIONS request on the provided path and returns a Response object. Useful for testing RESTful interfaces.

When data is provided, it is used as the request body, and a Content-Type header is set to content_type.

The follow, secure and extra arguments act the same as for Client.get().

put(path, data='', content_type='application/octet-stream', follow=False, secure=False, **extra)
Makes a PUT request on the provided path and returns a Response object. Useful for testing RESTful interfaces.

When data is provided, it is used as the request body, and a Content-Type header is set to content_type.

The follow, secure and extra arguments act the same as for Client.get().

patch(path, data='', content_type='application/octet-stream', follow=False, secure=False, **extra)
Makes a PATCH request on the provided path and returns a Response object. Useful for testing RESTful interfaces.

The follow, secure and extra arguments act the same as for Client.get().

delete(path, data='', content_type='application/octet-stream', follow=False, secure=False, **extra)
Makes an DELETE request on the provided path and returns a Response object. Useful for testing RESTful interfaces.

When data is provided, it is used as the request body, and a Content-Type header is set to content_type.

The follow, secure and extra arguments act the same as for Client.get().

login(**credentials)
If your site uses Django’s authentication system and you deal with logging in users, you can use the test client’s login() method to simulate the effect of a user logging into the site.

After you call this method, the test client will have all the cookies and session data required to pass any login-based tests that may form part of a view.

The format of the credentials argument depends on which authentication backend you’re using (which is configured by your AUTHENTICATION_BACKENDS setting). If you’re using the standard authentication backend provided by Django (ModelBackend), credentials should be the user’s username and password, provided as keyword arguments:

c = Client()
c.login(username='fred', password='secret')

# Now you can access a view that's only available to logged-in users.
If you’re using a different authentication backend, this method may require different credentials. It requires whichever credentials are required by your backend’s authenticate() method.

login() returns True if it the credentials were accepted and login was successful.

Finally, you’ll need to remember to create user accounts before you can use this method. As we explained above, the test runner is executed using a test database, which contains no users by default. As a result, user accounts that are valid on your production site will not work under test conditions. You’ll need to create users as part of the test suite – either manually (using the Django model API) or with a test fixture. Remember that if you want your test user to have a password, you can’t set the user’s password by setting the password attribute directly – you must use the set_password() function to store a correctly hashed password. Alternatively, you can use the create_user() helper method to create a new user with a correctly hashed password.


logout()
If your site uses Django’s authentication system, the logout() method can be used to simulate the effect of a user logging out of your site.

After you call this method, the test client will have all the cookies and session data cleared to defaults. Subsequent requests will appear to come from an AnonymousUser.




Testing responses
The get() and post() methods both return a Response object. This Response object is not the same as the HttpResponse object returned by Django views; the test response object has some additional data useful for test code to verify.

Specifically, a Response object has the following attributes:

class Response
client
The test client that was used to make the request that resulted in the response.

content
The body of the response, as a string. This is the final page content as rendered by the view, or any error message.

context
The template Context instance that was used to render the template that produced the response content.

If the rendered page used multiple templates, then context will be a list of Context objects, in the order in which they were rendered.

Regardless of the number of templates used during rendering, you can retrieve context values using the [] operator. For example, the context variable name could be retrieved using:

response = client.get('/foo/')
response.context['name']
'Arthur'
request
The request data that stimulated the response.

wsgi_request
The WSGIRequest instance generated by the test handler that generated the response.

status_code
The HTTP status of the response, as an integer. See RFC 2616 for a full list of HTTP status codes.

templates
A list of Template instances used to render the final content, in the order they were rendered. For each template in the list, use template.name to get the template’s file name, if the template was loaded from a file. (The name is a string such as 'admin/index.html'.)

You can also use dictionary syntax on the response object to query the value of any settings in the HTTP headers. For example, you could determine the content type of a response using response['Content-Type'].

Exceptions
If you point the test client at a view that raises an exception, that exception will be visible in the test case. You can then use a standard try ... except block or assertRaises() to test for exceptions.

The only exceptions that are not visible to the test client are Http404, PermissionDenied, SystemExit, and SuspiciousOperation. Django catches these exceptions internally and converts them into the appropriate HTTP response codes. In these cases, you can check response.status_code in your test.

Persistent state
The test client is stateful. If a response returns a cookie, then that cookie will be stored in the test client and sent with all subsequent get() and post() requests.

Expiration policies for these cookies are not followed. If you want a cookie to expire, either delete it manually or create a new Client instance (which will effectively delete all cookies).

A test client has two attributes that store persistent state information. You can access these properties as part of a test condition.

Client.cookies
A Python SimpleCookie object, containing the current values of all the client cookies. See the documentation of the http.cookies module for more.

Client.session
A dictionary-like object containing session information. See the session documentation for full details.

In Django 1.7, client.session returns a plain dictionary if the session is empty. The following code creates a test client with a fully working session engine:

from importlib import import_module

from django.conf import settings
from django.test import Client

def get_client_with_session(self):
    client = Client()
    engine = import_module(settings.SESSION_ENGINE)
    s = engine.SessionStore()
    s.save()
    client.cookies[settings.SESSION_COOKIE_NAME] = s.session_key
    return client
Example
The following is a simple unit test using the test client:

import unittest
from django.test import Client

class SimpleTest(unittest.TestCase):
    def setUp(self):
        # Every test needs a client.
        self.client = Client()

    def test_details(self):
        # Issue a GET request.
        response = self.client.get('/customer/details/')

        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200)

        # Check that the rendered context contains 5 customers.
        self.assertEqual(len(response.context['customers']), 5)






SimpleTestCase
class SimpleTestCase
A thin subclass of unittest.TestCase, it extends it with some basic functionality like:

•Saving and restoring the Python warning machinery state.
•Some useful assertions like:
?Checking that a callable raises a certain exception.
?Testing form field rendering and error treatment.
?Testing HTML responses for the presence/lack of a given fragment.
?Verifying that a template has/hasn't been used to generate a given response content.
?Verifying a HTTP redirect is performed by the app.
?Robustly testing two HTML fragments for equality/inequality or containment.
?Robustly testing two XML fragments for equality/inequality.
?Robustly testing two JSON fragments for equality.
•The ability to run tests with modified settings.
•Using the client Client.
•Custom test-time URL maps.
Changed in Django 1.6: 
The latter two features were moved from TransactionTestCase to SimpleTestCase in Django 1.6.

If you need any of the other more complex and heavyweight Django-specific features like:

•Testing or using the ORM.
•Database fixtures.
•Test skipping based on database backend features.
•The remaining specialized assert* methods.
then you should use TransactionTestCase or TestCase instead.

SimpleTestCase inherits from unittest.TestCase.



TransactionTestCase
class TransactionTestCase
Django’s TestCase class (described below) makes use of database transaction facilities to speed up the process of resetting the database to a known state at the beginning of each test. A consequence of this, however, is that the effects of transaction commit and rollback cannot be tested by a Django TestCase class. If your test requires testing of such transactional behavior, you should use a Django TransactionTestCase.

TransactionTestCase and TestCase are identical except for the manner in which the database is reset to a known state and the ability for test code to test the effects of commit and rollback:

•A TransactionTestCase resets the database after the test runs by truncating all tables. A TransactionTestCase may call commit and rollback and observe the effects of these calls on the database.
•A TestCase, on the other hand, does not truncate tables after a test. Instead, it encloses the test code in a database transaction that is rolled back at the end of the test. Both explicit commits like transaction.commit() and implicit ones that may be caused by transaction.atomic() are replaced with a nop operation. This guarantees that the rollback at the end of the test restores the database to its initial state.
Warning

TestCase running on a database that does not support rollback (e.g. MySQL with the MyISAM storage engine), and all instances of TransactionTestCase, will roll back at the end of the test by deleting all data from the test database and reloading initial data for apps without migrations.

Apps with migrations will not see their data reloaded; if you need this functionality (for example, third-party apps should enable this) you can set serialized_rollback = True inside the TestCase body.

Warning

While commit and rollback operations still appear to work when used in TestCase, no actual commit or rollback will be performed by the database. This can cause your tests to pass or fail unexpectedly. Always use TransactionTestCase when testing transactional behavior or any code that can’t normally be executed in autocommit mode (select_for_update() is an example).

TransactionTestCase inherits from SimpleTestCase.



TestCase
class TestCase
This class provides some additional capabilities that can be useful for testing Web sites.

Converting a normal unittest.TestCase to a Django TestCase is easy: Just change the base class of your test from 'unittest.TestCase' to 'django.test.TestCase'. All of the standard Python unit test functionality will continue to be available, but it will be augmented with some useful additions, including:

•Automatic loading of fixtures.
•Wraps each test in a transaction.
•Creates a TestClient instance.
•Django-specific assertions for testing for things like redirection and form errors.
TestCase inherits from TransactionTestCase.




LiveServerTestCase
class LiveServerTestCase
LiveServerTestCase does basically the same as TransactionTestCase with one extra feature: it launches a live Django server in the background on setup, and shuts it down on teardown. This allows the use of automated test clients other than the Django dummy client such as, for example, the Selenium client, to execute a series of functional tests inside a browser and simulate a real user’s actions.

By default the live server’s address is 'localhost:8081' and the full URL can be accessed during the tests with self.live_server_url. If you’d like to change the default address (in the case, for example, where the 8081 port is already taken) then you may pass a different one to the test command via the --liveserver option, for example:

$ ./manage.py test --liveserver=localhost:8082
Another way of changing the default server address is by setting the DJANGO_LIVE_TEST_SERVER_ADDRESS environment variable somewhere in your code (for example, in a custom test runner):

import os
os.environ['DJANGO_LIVE_TEST_SERVER_ADDRESS'] = 'localhost:8082'
In the case where the tests are run by multiple processes in parallel (for example, in the context of several simultaneous continuous integration builds), the processes will compete for the same address, and therefore your tests might randomly fail with an “Address already in use” error. To avoid this problem, you can pass a comma-separated list of ports or ranges of ports (at least as many as the number of potential parallel processes). For example:

$ ./manage.py test --liveserver=localhost:8082,8090-8100,9000-9200,7041
Then, during test execution, each new live test server will try every specified port until it finds one that is free and takes it.

To demonstrate how to use LiveServerTestCase, let’s write a simple Selenium test. First of all, you need to install the selenium package into your Python path:

$ pip install selenium
Then, add a LiveServerTestCase-based test to your app’s tests module (for example: myapp/tests.py). The code for this test may look as follows:

from django.test import LiveServerTestCase
from selenium.webdriver.firefox.webdriver import WebDriver

class MySeleniumTests(LiveServerTestCase):
    fixtures = ['user-data.json']

    @classmethod
    def setUpClass(cls):
        cls.selenium = WebDriver()
        super(MySeleniumTests, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super(MySeleniumTests, cls).tearDownClass()

    def test_login(self):
        self.selenium.get('%s%s' % (self.live_server_url, '/login/'))
        username_input = self.selenium.find_element_by_name("username")
        username_input.send_keys('myuser')
        password_input = self.selenium.find_element_by_name("password")
        password_input.send_keys('secret')
        self.selenium.find_element_by_xpath('//input[@value="Log in"]').click()
Finally, you may run the test as follows:

$ ./manage.py test myapp.tests.MySeleniumTests.test_login
This example will automatically open Firefox then go to the login page, enter the credentials and press the “Log in” button. Selenium offers other drivers in case you do not have Firefox installed or wish to use another browser. The example above is just a tiny fraction of what the Selenium client can do; check out the full reference for more details.

Changed in Django 1.7: 
In older versions, LiveServerTestCase relied on the staticfiles contrib app to transparently serve static files during the execution of tests. This functionality has been moved to the StaticLiveServerTestCase subclass, so use that subclass if you need the original behavior.

LiveServerTestCase now simply publishes the contents of the file system under STATIC_ROOT at the STATIC_URL.

Note

When using an in-memory SQLite database to run the tests, the same database connection will be shared by two threads in parallel: the thread in which the live server is run and the thread in which the test case is run. It’s important to prevent simultaneous database queries via this shared connection by the two threads, as that may sometimes randomly cause the tests to fail. So you need to ensure that the two threads don’t access the database at the same time. In particular, this means that in some cases (for example, just after clicking a link or submitting a form), you might need to check that a response is received by Selenium and that the next page is loaded before proceeding with further test execution. Do this, for example, by making Selenium wait until the <body> HTML tag is found in the response (requires Selenium > 2.13):

def test_login(self):
    from selenium.webdriver.support.wait import WebDriverWait
    timeout = 2
    ...
    self.selenium.find_element_by_xpath('//input[@value="Log in"]').click()
    # Wait until the response is received
    WebDriverWait(self.selenium, timeout).until(
        lambda driver: driver.find_element_by_tag_name('body'))
The tricky thing here is that there’s really no such thing as a “page load,” especially in modern Web apps that generate HTML dynamically after the server generates the initial document. So, simply checking for the presence of <body> in the response might not necessarily be appropriate for all use cases. Please refer to the Selenium FAQ and Selenium documentation for more information.




Default test client
SimpleTestCase.client
Every test case in a django.test.*TestCase instance has access to an instance of a Django test client. This client can be accessed as self.client. This client is recreated for each test, so you don’t have to worry about state (such as cookies) carrying over from one test to another.

This means, instead of instantiating a Client in each test:

import unittest
from django.test import Client

class SimpleTest(unittest.TestCase):
    def test_details(self):
        client = Client()
        response = client.get('/customer/details/')
        self.assertEqual(response.status_code, 200)

    def test_index(self):
        client = Client()
        response = client.get('/customer/index/')
        self.assertEqual(response.status_code, 200)
...you can just refer to self.client, like so:

from django.test import TestCase

class SimpleTest(TestCase):
    def test_details(self):
        response = self.client.get('/customer/details/')
        self.assertEqual(response.status_code, 200)

    def test_index(self):
        response = self.client.get('/customer/index/')
        self.assertEqual(response.status_code, 200)

Customizing the test client
SimpleTestCase.client_class
If you want to use a different Client class (for example, a subclass with customized behavior), use the client_class class attribute:

from django.test import TestCase, Client

class MyTestClient(Client):
    # Specialized methods for your environment
    ...

class MyTest(TestCase):
    client_class = MyTestClient

    def test_my_stuff(self):
        # Here self.client is an instance of MyTestClient...
        call_some_test_code()



TransactionTestCase.fixtures
A test case for a database-backed Web site isn’t much use if there isn’t any data in the database. To make it easy to put test data into the database, Django’s custom TransactionTestCase class provides a way of loading fixtures.

A fixture is a collection of data that Django knows how to import into a database. For example, if your site has user accounts, you might set up a fixture of fake user accounts in order to populate your database during tests.

The most straightforward way of creating a fixture is to use the manage.py dumpdata command. This assumes you already have some data in your database. See the dumpdata documentation for more details.

Note

If you’ve ever run manage.py migrate, you’ve already used a fixture without even knowing it! When you call migrate in the database for the first time, Django installs a fixture called initial_data. This gives you a way of populating a new database with any initial data, such as a default set of categories.

Fixtures with other names can always be installed manually using the manage.py loaddata command.

Initial SQL data and testing

Django provides a second way to insert initial data into models – the custom SQL hook. However, this technique cannot be used to provide initial data for testing purposes. Django’s test framework flushes the contents of the test database after each test; as a result, any data added using the custom SQL hook will be lost.

Once you’ve created a fixture and placed it in a fixtures directory in one of your INSTALLED_APPS, you can use it in your unit tests by specifying a fixtures class attribute on your django.test.TestCase subclass:

from django.test import TestCase
from myapp.models import Animal

class AnimalTestCase(TestCase):
    fixtures = ['mammals.json', 'birds']

    def setUp(self):
        # Test definitions as before.
        call_setup_methods()

    def testFluffyAnimals(self):
        # A test that uses the fixtures.
        call_some_test_code()
Here’s specifically what will happen:

•At the start of each test case, before setUp() is run, Django will flush the database, returning the database to the state it was in directly after migrate was called.
•Then, all the named fixtures are installed. In this example, Django will install any JSON fixture named mammals, followed by any fixture named birds. See the loaddata documentation for more details on defining and installing fixtures.
This flush/load procedure is repeated for each test in the test case, so you can be certain that the outcome of a test will not be affected by another test, or by the order of test execution.

By default, fixtures are only loaded into the default database. If you are using multiple databases and set multi_db=True, fixtures will be loaded into all databases.

URLconf configuration
SimpleTestCase.urls
If your application provides views, you may want to include tests that use the test client to exercise those views. However, an end user is free to deploy the views in your application at any URL of their choosing. This means that your tests can’t rely upon the fact that your views will be available at a particular URL.

In order to provide a reliable URL space for your test, django.test.*TestCase classes provide the ability to customize the URLconf configuration for the duration of the execution of a test suite. If your *TestCase instance defines an urls attribute, the *TestCase will use the value of that attribute as the ROOT_URLCONF for the duration of that test.

For example:

from django.test import TestCase

class TestMyViews(TestCase):
    urls = 'myapp.test_urls'

    def testIndexPageView(self):
        # Here you'd test your view using Client.
        call_some_test_code()
This test case will use the contents of myapp.test_urls as the URLconf for the duration of the test case.




Assertions
As Python’s normal unittest.TestCase class implements assertion methods such as assertTrue() and assertEqual(), Django’s custom TestCase class provides a number of custom assertion methods that are useful for testing Web applications:

The failure messages given by most of these assertion methods can be customized with the msg_prefix argument. This string will be prefixed to any failure message generated by the assertion. This allows you to provide additional details that may help you to identify the location and cause of an failure in your test suite.

SimpleTestCase.assertRaisesMessage(expected_exception, expected_message, callable_obj=None, *args, **kwargs)
Asserts that execution of callable callable_obj raised the expected_exception exception and that such exception has an expected_message representation. Any other outcome is reported as a failure. Similar to unittest’s assertRaisesRegex() with the difference that expected_message isn’t a regular expression.

SimpleTestCase.assertFieldOutput(fieldclass, valid, invalid, field_args=None, field_kwargs=None, empty_value=u'')
Asserts that a form field behaves correctly with various inputs.

Parameters: •fieldclass – the class of the field to be tested.
•valid – a dictionary mapping valid inputs to their expected cleaned values.
•invalid – a dictionary mapping invalid inputs to one or more raised error messages.
•field_args – the args passed to instantiate the field.
•field_kwargs – the kwargs passed to instantiate the field.
•empty_value – the expected clean output for inputs in empty_values.
 

For example, the following code tests that an EmailField accepts a@a.com as a valid email address, but rejects aaa with a reasonable error message:

self.assertFieldOutput(EmailField, {'a@a.com': 'a@a.com'}, {'aaa': [u'Enter a valid email address.']})
SimpleTestCase.assertFormError(response, form, field, errors, msg_prefix='')
Asserts that a field on a form raises the provided list of errors when rendered on the form.

form is the name the Form instance was given in the template context.

field is the name of the field on the form to check. If field has a value of None, non-field errors (errors you can access via form.non_field_errors()) will be checked.

errors is an error string, or a list of error strings, that are expected as a result of form validation.

SimpleTestCase.assertFormsetError(response, formset, form_index, field, errors, msg_prefix='')
New in Django 1.6. 
Asserts that the formset raises the provided list of errors when rendered.

formset is the name the Formset instance was given in the template context.

form_index is the number of the form within the Formset. If form_index has a value of None, non-form errors (errors you can access via formset.non_form_errors()) will be checked.

field is the name of the field on the form to check. If field has a value of None, non-field errors (errors you can access via form.non_field_errors()) will be checked.

errors is an error string, or a list of error strings, that are expected as a result of form validation.

SimpleTestCase.assertContains(response, text, count=None, status_code=200, msg_prefix='', html=False)
Asserts that a Response instance produced the given status_code and that text appears in the content of the response. If count is provided, text must occur exactly count times in the response.

Set html to True to handle text as HTML. The comparison with the response content will be based on HTML semantics instead of character-by-character equality. Whitespace is ignored in most cases, attribute ordering is not significant. See assertHTMLEqual() for more details.

SimpleTestCase.assertNotContains(response, text, status_code=200, msg_prefix='', html=False)
Asserts that a Response instance produced the given status_code and that text does not appears in the content of the response.

Set html to True to handle text as HTML. The comparison with the response content will be based on HTML semantics instead of character-by-character equality. Whitespace is ignored in most cases, attribute ordering is not significant. See assertHTMLEqual() for more details.

SimpleTestCase.assertTemplateUsed(response, template_name, msg_prefix='')
Asserts that the template with the given name was used in rendering the response.

The name is a string such as 'admin/index.html'.

You can use this as a context manager, like this:

with self.assertTemplateUsed('index.html'):
    render_to_string('index.html')
with self.assertTemplateUsed(template_name='index.html'):
    render_to_string('index.html')
SimpleTestCase.assertTemplateNotUsed(response, template_name, msg_prefix='')
Asserts that the template with the given name was not used in rendering the response.

You can use this as a context manager in the same way as assertTemplateUsed().

SimpleTestCase.assertRedirects(response, expected_url, status_code=302, target_status_code=200, host=None, msg_prefix='', fetch_redirect_response=True)
Asserts that the response returned a status_code redirect status, redirected to expected_url (including any GET data), and that the final page was received with target_status_code.

If your request used the follow argument, the expected_url and target_status_code will be the url and status code for the final point of the redirect chain.

The host argument sets a default host if expected_url doesn’t include one (e.g. "/bar/"). If expected_url is an absolute URL that includes a host (e.g. "http://testhost/bar/"), the host parameter will be ignored. Note that the test client doesn’t support fetching external URLs, but the parameter may be useful if you are testing with a custom HTTP host (for example, initializing the test client with Client(HTTP_HOST="testhost").

New in Django 1.7. 
If fetch_redirect_response is False, the final page won’t be loaded. Since the test client can’t fetch externals URLs, this is particularly useful if expected_url isn’t part of your Django app.

New in Django 1.7. 
Scheme is handled correctly when making comparisons between two URLs. If there isn’t any scheme specified in the location where we are redirected to, the original request’s scheme is used. If present, the scheme in expected_url is the one used to make the comparisons to.

SimpleTestCase.assertHTMLEqual(html1, html2, msg=None)
Asserts that the strings html1 and html2 are equal. The comparison is based on HTML semantics. The comparison takes following things into account:

•Whitespace before and after HTML tags is ignored.
•All types of whitespace are considered equivalent.
•All open tags are closed implicitly, e.g. when a surrounding tag is closed or the HTML document ends.
•Empty tags are equivalent to their self-closing version.
•The ordering of attributes of an HTML element is not significant.
•Attributes without an argument are equal to attributes that equal in name and value (see the examples).
The following examples are valid tests and don’t raise any AssertionError:

self.assertHTMLEqual('<p>Hello <b>world!</p>',
    '''<p>
        Hello   <b>world! <b/>
    </p>''')
self.assertHTMLEqual(
    '<input type="checkbox" checked="checked" id="id_accept_terms" />',
    '<input id="id_accept_terms" type='checkbox' checked>')
html1 and html2 must be valid HTML. An AssertionError will be raised if one of them cannot be parsed.

Output in case of error can be customized with the msg argument.

SimpleTestCase.assertHTMLNotEqual(html1, html2, msg=None)
Asserts that the strings html1 and html2 are not equal. The comparison is based on HTML semantics. See assertHTMLEqual() for details.

html1 and html2 must be valid HTML. An AssertionError will be raised if one of them cannot be parsed.

Output in case of error can be customized with the msg argument.

SimpleTestCase.assertXMLEqual(xml1, xml2, msg=None)
Asserts that the strings xml1 and xml2 are equal. The comparison is based on XML semantics. Similarly to assertHTMLEqual(), the comparison is made on parsed content, hence only semantic differences are considered, not syntax differences. When invalid XML is passed in any parameter, an AssertionError is always raised, even if both string are identical.

Output in case of error can be customized with the msg argument.

SimpleTestCase.assertXMLNotEqual(xml1, xml2, msg=None)
Asserts that the strings xml1 and xml2 are not equal. The comparison is based on XML semantics. See assertXMLEqual() for details.

Output in case of error can be customized with the msg argument.

SimpleTestCase.assertInHTML(needle, haystack, count=None, msg_prefix='')
Asserts that the HTML fragment needle is contained in the haystack one.

If the count integer argument is specified, then additionally the number of needle occurrences will be strictly verified.

Whitespace in most cases is ignored, and attribute ordering is not significant. The passed-in arguments must be valid HTML.

SimpleTestCase.assertJSONEqual(raw, expected_data, msg=None)
Asserts that the JSON fragments raw and expected_data are equal. Usual JSON non-significant whitespace rules apply as the heavyweight is delegated to the json library.

Output in case of error can be customized with the msg argument.

TransactionTestCase.assertQuerysetEqual(qs, values, transform=repr, ordered=True, msg=None)
Asserts that a queryset qs returns a particular list of values values.

The comparison of the contents of qs and values is performed using the function transform; by default, this means that the repr() of each value is compared. Any other callable can be used if repr() doesn’t provide a unique or helpful comparison.

By default, the comparison is also ordering dependent. If qs doesn’t provide an implicit ordering, you can set the ordered parameter to False, which turns the comparison into a Python set comparison.

Output in case of error can be customized with the msg argument.

Changed in Django 1.6: 
The method now checks for undefined order and raises ValueError if undefined order is spotted. The ordering is seen as undefined if the given qs isn’t ordered and the comparison is against more than one ordered values.

Changed in Django 1.7: 
The method now accepts a msg parameter to allow customization of error message

TransactionTestCase.assertNumQueries(num, func, *args, **kwargs)
Asserts that when func is called with *args and **kwargs that num database queries are executed.

If a "using" key is present in kwargs it is used as the database alias for which to check the number of queries. If you wish to call a function with a using parameter you can do it by wrapping the call with a lambda to add an extra parameter:

self.assertNumQueries(7, lambda: my_function(using=7))
You can also use this as a context manager:

with self.assertNumQueries(2):
    Person.objects.create(name="Aaron")
    Person.objects.create(name="Daniel")
Email services
If any of your Django views send email using Django’s email functionality, you probably don’t want to send email each time you run a test using that view. For this reason, Django’s test runner automatically redirects all Django-sent email to a dummy outbox. This lets you test every aspect of sending email – from the number of messages sent to the contents of each message – without actually sending the messages.

The test runner accomplishes this by transparently replacing the normal email backend with a testing backend. (Don’t worry – this has no effect on any other email senders outside of Django, such as your machine’s mail server, if you’re running one.)

django.core.mail.outbox
During test running, each outgoing email is saved in django.core.mail.outbox. This is a simple list of all EmailMessage instances that have been sent. The outbox attribute is a special attribute that is created only when the locmem email backend is used. It doesn’t normally exist as part of the django.core.mail module and you can’t import it directly. The code below shows how to access this attribute correctly.

Here’s an example test that examines django.core.mail.outbox for length and contents:

from django.core import mail
from django.test import TestCase

class EmailTest(TestCase):
    def test_send_email(self):
        # Send message.
        mail.send_mail('Subject here', 'Here is the message.',
            'from@example.com', ['to@example.com'],
            fail_silently=False)

        # Test that one message has been sent.
        self.assertEqual(len(mail.outbox), 1)

        # Verify that the subject of the first message is correct.
        self.assertEqual(mail.outbox[0].subject, 'Subject here')
As noted previously, the test outbox is emptied at the start of every test in a Django *TestCase. To empty the outbox manually, assign the empty list to mail.outbox:

from django.core import mail

# Empty the test outbox
mail.outbox = []
Management Commands
Management commands can be tested with the call_command() function. The output can be redirected into a StringIO instance:

from django.core.management import call_command
from django.test import TestCase
from django.utils.six import StringIO

class ClosepollTest(TestCase):
    def test_command_output(self):
        out = StringIO()
        call_command('closepoll', stdout=out)
        self.assertIn('Expected output', out.getvalue())
Skipping tests
The unittest library provides the @skipIf and @skipUnless decorators to allow you to skip tests if you know ahead of time that those tests are going to fail under certain conditions.

For example, if your test requires a particular optional library in order to succeed, you could decorate the test case with @skipIf. Then, the test runner will report that the test wasn’t executed and why, instead of failing the test or omitting the test altogether.

To supplement these test skipping behaviors, Django provides two additional skip decorators. Instead of testing a generic boolean, these decorators check the capabilities of the database, and skip the test if the database doesn’t support a specific named feature.

The decorators use a string identifier to describe database features. This string corresponds to attributes of the database connection features class. See django.db.backends.BaseDatabaseFeatures class for a full list of database features that can be used as a basis for skipping tests.

skipIfDBFeature(feature_name_string)
Skip the decorated test or TestCase if the named database feature is supported.

For example, the following test will not be executed if the database supports transactions (e.g., it would not run under PostgreSQL, but it would under MySQL with MyISAM tables):

class MyTests(TestCase):
    @skipIfDBFeature('supports_transactions')
    def test_transaction_behavior(self):
        # ... conditional test code
Changed in Django 1.7: 
skipIfDBFeature can now be used to decorate a TestCase class.

skipUnlessDBFeature(feature_name_string)
Skip the decorated test or TestCase if the named database feature is not supported.

For example, the following test will only be executed if the database supports transactions (e.g., it would run under PostgreSQL, but not under MySQL with MyISAM tables):

class MyTests(TestCase):
    @skipUnlessDBFeature('supports_transactions')
    def test_transaction_behavior(self):
        # ... conditional test code
Changed in Django 1.7: 
skipUnlessDBFeature can now be used to decorate a TestCase class.





@@@
Sessions, Users, and Registration





Cookies

When you open your browser and type in google.com, your browser sends an HTTP request to Google that starts something like this:

           GET / HTTP/1.1
           Host: google.com
           ...

When Google replies, the HTTP response looks something like the following:

           HTTP/1.1 200 OK
           Content-Type: text/html
           Set-Cookie: PREF=ID=5b14f22bdaf1e81c:TM=1167000671:LM=1167000671;
                       expires=Sun, 17-Jan-2038 19:14:07 GMT;
                       path=/; domain=.google.com
           Server: GWS/2.1
           ...

Notice       the      Set-Cookie          header.     Your     browser        will     store      that  cookie value (PREF=ID=5b14f22bdaf1e81c:TM=1167000671:LM=1167000671) and serve it back to Google every time you
access the site. So the next time you access Google, your browser is going to send a request like this:

           GET / HTTP/1.1
           Host: google.com
           Cookie: PREF=ID=5b14f22bdaf1e81c:TM=1167000671:LM=1167000671
           ...




Getting Cookies


Every request object has a COOKIES object that acts like a dictionary; you can use it to read any cookies that the browser has sent to the view:

          def show_color(request):
              if "favorite_color" in request.COOKIES:
                  return HttpResponse("Your favorite color is %s" % \
                      request.COOKIES["favorite_color"])
              else:
                  return HttpResponse("You don't have a favorite color.")

Setting Cookies

          def set_color(request):
              if "favorite_color" in request.GET:

                      # Create an HttpResponse object...
                      response = HttpResponse("Your favorite color is now %s" % \
                          request.GET["favorite_color"])

                      # ... and set a cookie on the response
                      response.set_cookie("favorite_color",
                                          request.GET["favorite_color"])

                      return response

                else:
                    return HttpResponse("You didn't give a favorite color.")

optional arguments to response.set_cookie()

max_age ,expires, path, domain, secure



DJANGO'S SESSION FRAMEWORK _ use Session framework because Plain cookie is not secure or persistant



Enabling Sessions (default is enabled)
To enable sessions, you'll needto follow these steps:
       1. Edit your MIDDLEWARE_CLASSES setting and make sure MIDDLEWARE_CLASSES contains     'django.contrib.sessions.middleware.SessionMiddleware'.
       2. Make sure 'django.contrib.sessions' is in your INSTALLED_APPS setting (and run manage.py   syncdb if you have to add it).


Using Sessions in Views

When SessionMiddleware is activated, each HttpRequest object--the first argument to any Django view function--will have a session attribute, which is a dictionary-like object

 Set a session value:
          request.session["fav_color"] = "blue"

          # Get a session value -- this could be called in a different view,
          # or many requests later (or both):
          fav_color = request.session["fav_color"]

          # Clear an item from the session:
          del request.session["fav_color"]

          # Check if the session has a given key:
          if "fav_color" in request.session:

Example

User comment example:

          def post_comment(request, new_comment):
              if request.session.get('has_commented', False):
                  return HttpResponse("You've already commented.")
              c = comments.Comment(comment=new_comment)
              c.save()
              request.session['has_commented'] = True
              return HttpResponse('Thanks for your comment!')


Login example:

           def login(request):
               try:
                    m = Member.objects.get(username__exact=request.POST['username'])
                    if m.password == request.POST['password']:
                        request.session['member_id'] = m.id
                        return HttpResponse("You're logged in.")
               except Member.DoesNotExist:
                    return HttpResponse("Your username and password didn't match.")

Logout Example:

           def logout(request):
               try:
                    del request.session['member_id']
               except KeyError:
                    pass
               return HttpResponse("You're logged out.")



Another login version with test cookies: every browser does not accept cookies. To check whether browser accepts cookies,
call request.session.set_test_cookie() in a view, and check request.session.test_cookie_worked() in a subsequent view

def login(request):

                 # If we submitted the form...
                 if request.method == 'POST':

                       # Check that the test cookie worked (we set it below):
                       if request.session.test_cookie_worked():

                             # The test cookie worked, so delete it.
                             request.session.delete_test_cookie()

                             # In practice, we'd need some logic to check username/password
                             # here, but since this is an example...
                             return HttpResponse("You're logged in.")

                      # The test cookie failed, so display an error message. If this
                      # was a real site we'd want to display a friendlier message.
                      else:
                          return HttpResponse("Please enable cookies and try again.")

                # If we didn't post, send the test cookie along with the login form.
                request.session.set_test_cookie()
                return render_to_response('foo/login_form.html')




Using Sessions Outside of Views
each session is just a normal Django model defined in django.contrib.sessions.models. 

          from django.contrib.sessions.models import Session
          s = Session.objects.get(pk='2b1189a188b44ad18c35e113ac6ceead')
          s.expire_date
          datetime.datetime(2005, 8, 20, 13, 35, 12)

call get_decoded() to get the actual session data. 

          s.session_data
          'KGRwMQpTJ19hdXRoX3VzZXJfaWQnCnAyCkkxCnMuMTExY2ZjODI2Yj...'
          s.get_decoded()
          {'user_id': 42}


When Sessions Are Saved
By default, Django only saves to the database if the session has been modified --that is, if any of its dictionary values have been assigned or deleted:

          # Session is modified.
          request.session['foo'] = 'bar'

          # Session is modified.
          del request.session['foo']

          # Session is modified.
          request.session['foo'] = {}

          # Gotcha: Session is NOT modified, because this alters
          # request.session['foo'] instead of request.session.
          request.session['foo']['bar'] = 'baz'



To      change      this     default    behavior,     set    SESSION_SAVE_EVERY_REQUEST     to      True in settings.py

 Note that the session cookie is sent only when a session has been created or modified. If SESSION_SAVE_EVERY_REQUEST is True, the session cookie will be sent on every request. Similarly, the expires
part of a session cookie is updated each time the session cookie is sent.


Browser-Length Sessions vs. Persistent Sessions

By default, SESSION_EXPIRE_AT_BROWSER_CLOSE is set to False, which means session cookies will be stored in users' browsers for SESSION_COOKIE_AGE seconds (which defaults to two weeks, or 1,209,600 seconds). 
Use this if you don't want people to have to log in every time they open a browser.

If SESSION_EXPIRE_AT_BROWSER_CLOSE is set to True, Django will use browser-length cookies.


Other Session Settings

Setting                                     Description                                          Default
SESSION_COOKIE_DOMAIN                       The domain to use for session cookies. Set this      None
                                            to a string such as ".lawrence.com" for
                                            cross-domain cookies, or use None for a
                                            standard cookie.
SESSION_COOKIE_NAME                         The name of the cookie to use for sessions.          "sessionid"
                                            This can be any string.
SESSION_COOKIE_SECURE                       Whether to use a "secure" cookie for the             False
                                            session cookie. If this is set to True, the cookie
                                            will be marked as "secure," which means that
                                            browsers will ensure that the cookie is only
                                            sent via HTTPS.


USERS AND AUTHENTICATION

Django's auth/auth system consists of a number of parts:
        · Users: People registered with your site
        · Permissions: Binary (yes/no) flags designating whether a user may perform a certain task
        · Groups: A generic way of applying labels and permissions to more than one user
        · Messages: A simple way to queue and display system messages to users
        · Profiles: A mechanism to extend the user object with custom fields

Enabling Authentication Support

 1. Make sure the session framework is installed 
 2. Put 'django.contrib.auth' in your INSTALLED_APPS setting and run manage.py syncdb.
 3. Make sure that 'django.contrib.auth.middleware.AuthenticationMiddleware' is in your    MIDDLEWARE_CLASSES setting--after SessionMiddleware.

Main function for authentication

The main interface to use is  within a view is request.user; this is an object that represents the currently logged-in user. If the user isn'tlogged in, this will instead be an AnonymousUser object 

           if request.user.is_authenticated():
               # Do something for authenticated users.
           else:
               # Do something for anonymous users.


fields and methods of User class


Field                       Description
username                    Required; 30 characters or fewer. Alphanumeric characters only (letters, digits, and
                            underscores).
first_name                  Optional; 30 characters or fewer.
last_name                   Optional; 30 characters or fewer.
email                       Optional. Email address.
password                    Required. A hash of, and metadata about, the password (Django doesn't store the raw
                            password). See the "Passwords" section for more about this value.
is_staff                    Boolean. Designates whether this user can access the admin site.
is_active                   Boolean. Designates whether this account can be used to log in. Set this flag to False
                            instead of deleting accounts.
is_superuser                Boolean. Designates that this user has all permissions without explicitly assigning them.
last_login                  A datetime of the user's last login. This is set to the current date/time by default.
date_joined                 A datetime designating when the account was created. This is set to the current date/time
                            by default when the account is created.



Method                                              Description
is_authenticated()                                  Always returns True for "real" User objects. This is a way to tell
                                                    if the user has been authenticated. This does not imply any
                                                    permissions, and it doesn't check if the user is active. It only
                                                    indicates that the user has sucessfully authenticated.
is_anonymous()                                      Returns True only for AnonymousUser objects (and False for
                                                    "real" User objects). Generally, you should prefer using
                                                    is_authenticated() to this method.
get_full_name()                                     Returns the first_name plus the last_name, with a space in
                                                    between.
set_password(passwd)                                Sets the user's password to the given raw string, taking care of the
                                                    password hashing. This doesn't actually save the User object.
check_password(passwd)                              Returns True if the given raw string is the correct password for
                                                    the user. This takes care of the password hashing in making the
                                                    comparison.
get_group_permissions()                             Returns a list of permission strings that the user has through the
                                                    groups he or she belongs to.
get_all_permissions()                               Returns a list of permission strings that the user has, both through
                                                    group and user permissions.

has_perm(perm)                                       Returns True if the user has the specified permission, where
                                                     perm is in the format "package.codename". If the user is
                                                     inactive, this method will always return False.
has_perms(perm_list)                                 Returns True if the user has all of the specified permissions. If the
                                                     user is inactive, this method will always return False.
has_module_perms(app_label)                          Returns True if the user has any permissions in the given
                                                     app_label. If the user is inactive, this method will always return
                                                     False.
get_and_delete_messages()                            Returns a list of Message objects in the user's queue and deletes
                                                     the messages from the queue.
email_user(subj, msg)                                Sends an email to the user. This email is sent from the
                                                     DEFAULT_FROM_EMAIL setting. You can also pass a third
                                                     argument, from_email, to override the From address on the
                                                     email.
get_profile()                                        Returns a site-specific profile for this user. See the "Profiles"
                                                     section for more on this method.


User objects have two many-to-many fields: groups and permissions. User objects can access their related objects in the same way as any other many-to-many field:

          # Set a user's groups:
          myuser.groups = group_list

          # Add a user to some groups:
          myuser.groups.add(group1, group2,...)

          # Remove a user from some groups:
          myuser.groups.remove(group1, group2,...)

          # Remove a user from all groups:
          myuser.groups.clear()

          # Permissions work the same way
          myuser.permissions = permission_list
          myuser.permissions.add(permission1, permission2, ...)
          myuser.permissions.remove(permission1, permission2, ...)
          myuser.permissions.clear()



Authenticating and Login/logout


Option -1:


          from django.contrib import auth

          def login(request):
              username = request.POST['username']
              password = request.POST['password']
              user = auth.authenticate(username=username, password=password)		#Athenticate
              if user is not None and user.is_active:
                  # Correct password, and the user is marked "active"
                  auth.login(request, user)						#Login
                  # Redirect to a success page.
                  return HttpResponseRedirect("/account/loggedin/")
              else:
                  # Show an error page
                  return HttpResponseRedirect("/account/invalid/")



          def logout(request):
              auth.logout(request)
              # Redirect to a success page.
              return HttpResponseRedirect("/account/loggedout/")


Option-2

 from django.contrib.auth.views import login, logout

          urlpatterns = patterns('',
              # existing patterns here...
              (r'^accounts/login/$', login),
              (r'^accounts/logout/$', logout),
          )

/accounts/login/ and /accounts/logout/ are the default URLs that Django uses for these views.

By default, the login view renders a template at registration/login.html (you can change this template name by passing an extra view argument ,template_name)

If the user successfully logs in, he or she will be redirected to /accounts/profile/ by default. You can override this by providing a hidden field called next with the URL to redirect to after logging in. You can also pass this value as a GET
parameter to the login view and it will be automatically added to the context as a variable called next that you can insertinto that hidden field.

Default template would work but can be changed like below


          {% extends "base.html" %}

          {% block content %}

             {% if form.errors %}
               <p class="error">Sorry, that's not a valid username or password</p>
             {% endif %}

             <form action='.' method='post'>
               <label for="username">User name:</label>
               <input type="text" name="username" value="" id="username">
               <label for="password">Password:</label>
               <input type="password" name="password" value="" id="password">

               <input type="submit" value="login" />
               <input type="hidden" name="next" value="{{ next|escape }}" />
             <form action='.' method='post'>

          {% endblock %}


By default logout renders a template at registration/logged_out.html (which usually contains a "You've successfully logged out" message)



Limiting Access to Logged-in Users

Option-1

from django.http import HttpResponseRedirect

          def my_view(request):
              if not request.user.is_authenticated():
                  return HttpResponseRedirect('/login/?next=%s' % request.path)   # note next field
              # ...

or perhaps display an error message:

          def my_view(request):
              if not request.user.is_authenticated():
                  return render_to_response('myapp/login_error.html')
              # ...

Option-2


          from django.contrib.auth.decorators import login_required

          @login_required
          def my_view(request):
              # ...

login_required does the following:
     · If the user isn't logged in, redirect to /accounts/login/, passing the current absolute URL in the query string        as next, for example: /accounts/login/?next=/polls/3/.
     · If the user is logged in, execute the view normally. The view code can then assume that the user is logged in.


Limiting Access to Users based on Permission

Option-1

        def vote(request):
              if request.user.is_authenticated() and   request.user.has_perm('polls.can_vote')):
                  # vote here
              else:
                  return HttpResponse("You can't vote in this poll.")

Option-2

          def user_can_vote(user):
              return user.is_authenticated() and user.has_perm("polls.can_vote")

          @user_passes_text(user_can_vote, login_url="/login/")
          def vote(request):
              # Code here can assume a logged-in user with the correct permission.
              ...

login_url, which lets you specify the URL for your login page (/accounts/login/ by default).

Option-3
          from django.contrib.auth.decorators import permission_required

          @permission_required('polls.can_vote', login_url="/login/")
          def vote(request):
              # ...





Limiting Access to Generic Views
write a thin wrapper around the view and point your URLconf to your wrapper instead of the generic view itself:

                        from dango.contrib.auth.decorators import login_required
                        from django.views.generic.date_based import object_detail

                        @login_required
                        def limited_object_detail(*args, **kwargs):
                            return object_detail(*args, **kwargs)



Managing Users, Permissions, and Groups
The easiest way by far to manage the auth system is through the admin interface. Use that

Alternate is low level API


Creating Users

          from django.contrib.auth.models import User
          user = User.objects.create_user(username='john',
          ...                                 email='jlennon@beatles.com',
          ...                                 password='glass onion')

          user.is_staff = True
          user.save()


Changing Passwords


          user = User.objects.get(username='john')
          user.set_password('goo goo goo joob')
          user.save()


Handling Registration

          from   django import oldforms as forms
          from   django.http import HttpResponseRedirect
          from   django.shortcuts import render_to_response
          from   django.contrib.auth.forms import UserCreationForm

          def register(request):
              form = UserCreationForm()					# use this for User creation form
                 if request.method == 'POST':
                     data = request.POST.copy()
                     errors = form.get_validation_errors(data)
                     if not errors:
                         new_user = form.save(data)
                         return HttpResponseRedirect("/books/")
                 else:
                     data, errors = {}, {}

                 return render_to_response("registration/register.html", {
                     'form' : forms.FormWrapper(form, data, errors)
                 })



This form assumes a template named registration/register.html. Here's an example of what that template might look like:

         {% extends "base.html" %}

         {% block title %}Create an account{% endblock %}

         {% block content %}
           <h1>Create an account</h1>
           <form action="." method="post">
             {% if form.error_dict %}
               <p class="error">Please correct the errors below.</p>
             {% endif %}

              {% if form.username.errors %}
                {{ form.username.html_error_list }}
              {% endif %}
              <label for="id_username">Username:</label> {{ form.username }}

              {% if form.password1.errors %}
                {{ form.password1.html_error_list }}
              {% endif %}
              <label for="id_password1">Password: {{ form.password1 }}

              {% if form.password2.errors %}
                {{ form.password2.html_error_list }}
              {% endif %}
              <label for="id_password2">Password (again): {{ form.password2 }}

             <input type="submit" value="Create the account" />
           </label>
         {% endblock %}


Built-in forms  in django.contrib.auth.forms:

AdminPasswordChangeForm
A form used in the admin interface to change a user’s password.


class AuthenticationForm
A form for logging a user in.

class PasswordChangeForm
A form for allowing a user to change their password.

class PasswordResetForm
A form for generating and emailing a one-time use link to reset a user’s password.

class SetPasswordForm
A form that lets a user change their password without entering the old password.

class UserChangeForm
A form used in the admin interface to change a user’s information and permissions.

class UserCreationForm
A form for creating a new user.



Using the builtin

There are different methods to implement these views in your project. 
The easiest way is to include the provided URLconf in django.contrib.auth.urls in  URLconf, for example:


urlpatterns = [
    url('^', include('django.contrib.auth.urls'))
]


This will include the following URL patterns:


^login/$ [name='login']
^logout/$ [name='logout']
^password_change/$ [name='password_change']
^password_change/done/$ [name='password_change_done']
^password_reset/$ [name='password_reset']
^password_reset/done/$ [name='password_reset_done']
^reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$ [name='password_reset_confirm']
^reset/done/$ [name='password_reset_complete']


The views provide a URL name for easier reference. 

If you want more control over your URLs, you can reference a specific view in your URLconf:


urlpatterns = [
    url('^change-password/', 'django.contrib.auth.views.password_change')
]


The views have optional arguments you can use to alter the behavior of the view. 
For example, if you want to change the template name a view uses, you can provide the template_name argument. 


urlpatterns = [
    url(
        '^change-password/',
        'django.contrib.auth.views.password_change',
        {'template_name': 'change-password.html'}
    )
]


All views return a TemplateResponse instance, which allows you to easily customize the response data before rendering.
A way to do this is to wrap a view in your own view:


from django.contrib.auth import views

def change_password(request):
    template_response = views.password_change(request)
    # Do something with `template_response`
    return template_response




Using Authentication Data in Templates

When using RequestContext, the current user (either a User instance or an AnonymousUser instance) is stored in the template variable {{ user }}:

          {% if user.is_authenticated %}
            <p>Welcome, {{ user.username }}. Thanks for logging in.</p>
          {% else %}
            <p>Welcome, new user. Please log in.</p>
          {% endif %}


Usage:
This user's permissions are stored in the template variable {{ perms }}. 


You can use something like {{ perms.polls }} to check if theuser has any permissions for some given application, or you can use something like {{ perms.polls.can_vote }}
to check if the user has a specific permission.

          {% if perms.polls %}
            <p>You have permission to do something in the polls app.</p>
            {% if perms.polls.can_vote %}
              <p>You can vote!</p>
            {% endif %}
          {% else %}
            <p>You don't have permission to do anything in the polls app.</p>
          {% endif %}



Permissions


The Django admin site uses permissions as follows:
       · Access to view the "add" form, and add an object is limited to users with the add permission for that type of           object.
       · Access to view the change list, view the "change" form, and change an object is limited to users with the change            permission for that type of object.
       · Access to delete an object is limited to users with the delete permission for that type of object.


Permissions are set globally per type of object, not per specific object instance. 

These three basic permissions--add, change, and delete--are automatically created for each Django model that has a class Admin. 

These permissions will be of the form "<app>.<action>_<object_name>". That is, if you have a polls application with a Choice model, you'll get permissions named "polls.add_choice", "polls.change_choice",
and "polls.delete_choice".

Note that if your model doesn't have class Admin set when you run migrate., the permissions won't be created. If you initialize your database and add class Admin to models after the fact, you'll need to run migrate again to create
any missing permissions for your installed applications.


You can also create custom permissions for a given model object using the permissions attribute on Meta.

          class USCitizen(models.Model):
              # ...
              class Meta:
                  permissions = (
                      # Permission identifier                        human-readable permission name
                      ("can_drive",                                  "Can drive"),
                      ("can_vote",                                   "Can vote in elections"),
                      ("can_drink",                                  "Can drink alcohol"),
                  )



Assuming you have an application with an app_label foo and a model named Bar, to test for basic permissions you should use:
•add: user.has_perm('foo.add_bar')
•change: user.has_perm('foo.change_bar')
•delete: user.has_perm('foo.delete_bar')



Programmatically creating permissions

While custom permissions can be defined within a model’s Meta class, you can also create permissions directly. 
For example, you can create the can_publish permission for a BlogPost model in myapp:


from myapp.models import BlogPost
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType

content_type = ContentType.objects.get_for_model(BlogPost)
permission = Permission.objects.create(codename='can_publish',
                                       name='Can Publish Posts',
                                       content_type=content_type)


The permission can then be assigned to a User via its user_permissions attribute or to a Group via its permissions attribute.



Groups
Groups are a generic way of categorizing users so you can apply permissions, or some other label, to those users. A user can belong to any number of groups.


Messages
The message system is a lightweight way to queue messages for given users. A message is associated with a User. There's no concept of expiration or timestamps.

	def create_playlist(request, songs):
              # Create the playlist with the given songs.
              # ...
              request.user.message_set.create(    message="Your playlist was added successfully."              )
              return render_to_response("playlists/create.html",    context_instance=RequestContext(request))



When you use RequestContext, the current logged-in user and his or her messages are made available in the template context as the template variable {{ messages }}. Here's an example of template code that displays messages:

           {% if messages %}
           <ul>
                {% for message in messages %}
                <li>{{ message }}</li>
                {% endfor %}
           </ul>
           {% endif %}

Note that RequestContext calls get_and_delete_messages behind the scenes, so any messages will be deleted even if you don't display them.


User objects have two many-to-many fields: groups and user_permissions. User objects can access their related objects in the same way as any other Django model:


myuser.groups = [group_list]
myuser.groups.add(group, group, ...)
myuser.groups.remove(group, group, ...)
myuser.groups.clear()
myuser.user_permissions = [permission_list]
myuser.user_permissions.add(permission, permission, ...)
myuser.user_permissions.remove(permission, permission, ...)
myuser.user_permissions.clear()





User Profiles

The first step in creating a profile is to define a model that holds the profile information. The only requirement Django places on this model is that it have a unique ForeignKey to the User model; this field must be named user. 



           from django.db import models
           from django.contrib.auth.models import User

           class MySiteProfile(models.Model):
               # This is the only required field
               user = models.ForeignKey(User, unique=True)

                 # The rest is completely up to you...
                 favorite_band = models.CharField(maxlength=100, blank=True)
                 favorite_cheese = models.CharField(maxlength=100, blank=True)
                 lucky_number = models.IntegerField()

Next, you'll need to tell Django where to look for this profile object. 

AUTH_PROFILE_MODULE = "myapp.mysiteprofile"

Once that's done, you can access a user's profile by calling user.get_profile(). This function could raise a SiteProfileNotAvailable exception if AUTH_PROFILE_MODULE isn't defined, or it could raise a
DoesNotExist exception if the user doesn't have a profile already (you'll usually catch that exception and create a new profile at that time).




ACTIVATING THE ADMIN INTERFACE

     1. Add admin metadata to your models.
  
                     class Book(models.Model):
                         title = models.CharField(maxlength=100)
                         authors = models.ManyToManyField(Author)
                         publisher = models.ForeignKey(Publisher)
                         publication_date = models.DateField()
                         num_pages = models.IntegerField(blank=True, null=True)

                           def __str__(self):
                               return self.title

                           class Admin:
                               pass

 

     2. Install the admin application. Do this by adding "django.contrib.admin" to your INSTALLED_APPS        setting.
     3.   make   sure      that    "django.contrib.sessions",        "django.contrib.auth", and "django.contrib.contenttypes" are uncommented, since the
        admin application depends on them. Also uncomment all the lines in the MIDDLEWARE_CLASSES setting tuple
        and delete the TEMPLATE_CONTEXT_PROCESSOR setting to allow it to take the default values again.
     4. Run python manage.py migrate. This step will install the extra database tables the admin interface uses.


     5. Add the URL pattern to your urls.py. 
                   from django.conf.urls.defaults import *

                   urlpatterns = patterns('',
                       (r'^admin/', include('django.contrib.admin.urls')),
                   )

Now run python manage.py runserver to start the development server


Users, Groups, and Permissions
Since you're logged in as a superuser, you have access to create, edit, and delete any object. 

    You edit these users and permissions through the admin interface just like any other object. The link to the User and Group models is there on the admin index along with all the objects you've defined yourself.

        · The "is active" flag controls whether the user is active at all. If this flag is off, the user has no access to any URLs            that require login.
        · The "is staff" flag controls whether the user is allowed to log in to the admin interface (i.e., whether that user is           considered a "staff member" in your organization). Since this same user system can be used to control access to
           public (i.e., non-admin) sites (see Chapter 12), this flag differentiates between public users and administrators.
        · The "is superuser" flag gives the user full, unfettered access to every item in the admin interface; regular            permissions are ignored.


CUSTOMIZING THE ADMIN INTERFACE
To  add some display, searching, and filtering functions to this interface. Change the Admin declaration as follows:

           class Book(models.Model):
               title = models.CharField(maxlength=100)
               authors = models.ManyToManyField(Author)
               publisher = models.ForeignKey(Publisher)
               publication_date = models.DateField()

                 class Admin:
                     list_display = ('title', 'publisher', 'publication_date')
                     list_filter = ('publisher', 'publication_date')
                     ordering = ('-publication_date',)
                     search_fields = ('title',)



Activating admin interface


The admin is enabled in the default project template used by startproject.

For reference, here are the requirements:
1.Add 'django.contrib.admin' to your INSTALLED_APPS setting.
2.The admin has four dependencies - django.contrib.auth, django.contrib.contenttypes, django.contrib.messages and django.contrib.sessions. If these applications are not in your INSTALLED_APPS list, add them.
3.Add django.contrib.messages.context_processors.messages to TEMPLATE_CONTEXT_PROCESSORS as well as django.contrib.auth.middleware.AuthenticationMiddleware and django.contrib.messages.middleware.MessageMiddleware to MIDDLEWARE_CLASSES. (These are all active by default, so you only need to do this if you’ve manually tweaked the settings.)
4.Determine which of your application’s models should be editable in the admin interface.
5.For each of those models, optionally create a ModelAdmin class in admin.py that encapsulates the customized admin functionality and options for that particular model.
6.Instantiate an AdminSite and tell it about each of your models and ModelAdmin classes.
7.Hook the AdminSite instance into your URLconf.

Add the URL pattern to your urls.py. 
                   from django.conf.urls.defaults import *

                   urlpatterns = patterns('',
                       (r'^admin/', include('django.contrib.admin.urls')),
                   )

Example:

from django.contrib import admin
from myproject.myapp.models import Author

class AuthorAdmin(admin.ModelAdmin):
    pass
admin.site.register(Author, AuthorAdmin)

OR usig a decorator

from django.contrib import admin
from .models import Author

@admin.register(Author)
class AuthorAdmin(admin.ModelAdmin):
    pass



ModelAdmin options

ModelAdmin.date_hierarchy
Set date_hierarchy to the name of a DateField or DateTimeField in your model, and the change list page will include a date-based drilldown navigation by that field.

Example:


date_hierarchy = 'pub_date'


ModelAdmin.exclude
This attribute, if given, should be a list of field names to exclude from the form.

Example:

from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
    title = models.CharField(max_length=3)
    birth_date = models.DateField(blank=True, null=True)


If you want a form for the Author model that includes only the name and title fields, you would specify fields or exclude like this:


from django.contrib import admin

class AuthorAdmin(admin.ModelAdmin):
    fields = ('name', 'title')

class AuthorAdmin(admin.ModelAdmin):
    exclude = ('birth_date',)




Middleware

On occasion, you'll need to run a piece of code on each and every request that Django handles. This code might need to modify the request before the view handles it, it might need to log information about the request for debugging purposes,
and so forth.
You can do this with Django's middleware framework, which is a set of hooks into Django's request/response processing. 

Example

 middleware that lets sites running behind a proxy still see the correct IP address in request.META["REMOTE_ADDR"]:

          class SetRemoteAddrFromForwardedFor(object):
              def process_request(self, request):
                  try:
                       real_ip = request.META['HTTP_X_FORWARDED_FOR']
                  except KeyError:
                       pass
                  else:
                       # HTTP_X_FORWARDED_FOR can be a comma-separated list of IPs.
                       # Take just the first one.
                       real_ip = real_ip.split(",")[0]
                       request.META['REMOTE_ADDR'] = real_ip

every request's X-Forwarded-For value will be automatically inserted into request.META['REMOTE_ADDR']. 

Activatation:

To activate a middleware component, add it to the MIDDLEWARE_CLASSES tuple in your settings module.

Default list is

          MIDDLEWARE_CLASSES = (
              'django.middleware.common.CommonMiddleware',
              'django.contrib.sessions.middleware.SessionMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.middleware.doc.XViewMiddleware'
          )


The order is significant. On the request and view phases, Django applies middleware in the order given in MIDDLEWARE_CLASSES, and on the response and exception phases, Django applies middleware in reverse order. That is,
Django treats MIDDLEWARE_CLASSES as a sort of "wrapper" around the view function: on the request it walks down the list to the view, and on the response it walks back up. 


MIDDLEWARE METHODS


Initializer: __init__(self)
Use __init__() to perform systemwide setup for a given middleware class.
   For performance reasons, each activated middleware class is instantiated only once per server process. This means that
__init__() is called only once -- at server startup -- not for individual requests.


Request Preprocessor: process_request(self, request)
This method gets called as soon as the request has been received -- before Django has parsed the URL to determine which
view to run. It gets passed the HttpRequest object, which you may modify at will.
   process_request() should return either None or an HttpResponse object.


       · If it returns None, Django will continue processing this request, executing any other middleware and then the         appropriate view.
       · If it returns an HttpResponse object, Django won't bother calling any other middleware (of any type) or the          appropriate view. Django will immediately return that HttpResponse.


View Preprocessor: process_view(self, request, view, args, kwargs)
This method gets called after the request preprocessor is called and Django has determined which view to execute, but before that view has actually been executed.
   The arguments passed to this view are shown in Table 15-1.
                                      Table 15-1. Arguments Passed to process_view()
Argument               Explanation
request                The HttpRequest object.
view                   The Python function that Django will call to handle this request. This is the actual function object
                       itself, not the name of the function as a string.
args                   The list of positional arguments that will be passed to the view, not including the request
                       argument (which is always the first argument to a view).
kwargs                 The dictionary of keyword arguments that will be passed to the view.
Just like process_request(), process_view() should return either None or an HttpResponse object.
        · If it returns None, Django will continue processing this request, executing any other middleware and then the
           appropriate view.
        · If it returns an HttpResponse object, Django won't bother calling any other middleware (of any type) or the
           appropriate view. Django will immediately return that HttpResponse.


Response Postprocessor: process_response(self, request, response)
This method gets called after the view function is called and the response is generated. Here, the processor can modify the
content of a response; one obvious use case is content compression, such as gzipping of the request's HTML.

process_response() must return an HttpResponse object. That response could be the original one passed into the function (possibly modified) or a
brand-new one.


Exception Postprocessor: process_exception(self, request, exception)
This method gets called only if something goes wrong and a view raises an uncaught exception. 

   process_exception() should return a either None or an HttpResponse object.
       · If it returns None, Django will continue processing this request with the framework's built-in exception handling.
       · If it returns an HttpResponse object, Django will use that response instead of the framework's built-in
          exception handling.

BUILT-IN MIDDLEWARE

Middleware class: django.contrib.auth.middleware.AuthenticationMiddleware.
   This middleware enables authentication support. It adds the request.user attribute, representing the currently
logged-in user, to every incoming HttpRequest object.



"Common" Middleware
Middleware class: django.middleware.common.CommonMiddleware.
   This middleware adds a few conveniences 

Compression Middleware
Middleware class: django.middleware.gzip.GZipMiddleware.
   This middleware automatically compresses content for browsers that understand gzip compression 


Conditional GET Middleware
Middleware class: django.middleware.http.ConditionalGetMiddleware.
    This middleware provides support for conditional GET operations. If the response has an Last-Modified or ETagor header, and the request has If-None-Match or If-Modified-Since, the response is replaced by an 304 ("Not
modified") response. ETag support depends on on the USE_ETAGS setting and expects the ETag response header toalready be set. the ETag header is set by the Common middleware.
    It also removes the content from any response to a HEAD request and sets the Date and Content-Length response headers for all requests.


Reverse Proxy Support (X-Forwarded-For Middleware)
Middleware class: django.middleware.http.SetRemoteAddrFromForwardedFor.
It sets request.META['REMOTE_ADDR'] based on request.META['HTTP_X_FORWARDED_FOR'], if the latter is set.
This is useful if you're sitting behind a reverse proxy that causes each request's REMOTE_ADDR to be set to 127.0.0.1.


Session Support Middleware
Middleware class: django.contrib.sessions.middleware.SessionMiddleware.
   This middleware enables session support. 


Sitewide Cache Middleware
Middleware class: django.middleware.cache.CacheMiddleware.
   This middleware caches each Django-powered page. 


Transaction Middleware
Middleware class: django.middleware.transaction.TransactionMiddleware.
   This middleware binds a database COMMIT or ROLLBACK to the request/response phase. If a view function runs
successfully, a COMMIT is issued. If the view raises an exception, a ROLLBACK is issued.


"X-View" Middleware
Middleware class: django.middleware.doc.XViewMiddleware.
   This middleware sends custom X-View HTTP headers to HEAD requests that come from IP addresses defined in the INTERNAL_IPS setting. This is used by Django's automatic documentation system.


Example custom Tag

Firstly create the file structure. Go into the app directory where the tag is needed, and add these files:
templatetags
templatetags/__init__.py
templatetags/video_tags.py

The templatetags/video_tags.py file:
from django import template

register = template.Library()

@register.simple_tag
def get_rate(object):
    return object.rate

templates/index.html:

{% load video_tags %}

<html><body>

{% get_rate newobject %}

</body></html>

#views.py

from django.shortcuts import render

class MyClass:
	def __init__(self, arg):
		self.rate = arg


def index(request):
	a = MyClass(20)
	context = {'newobject': a}
	return render(request, 'index.html', context)




#urls.py

from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
]


Example Custom Filter

project/
  my_app/
    templatetags/
      __init__.py    # Important! It makes templatetags a module. You can put your filters here, or in another file.
      apptags.py     # Or just put them in __init__.py


apptags.py:

from django import template
register = template.Library()

@register.filter(name = 'commatodot')
def commatodot(value, arg):
    return str(value).replace(",", '.')
commatodot.isSafe = True

# templates/index.html
{% load apptags %}

<html><body>

{{ string|commatodot }}

</body></html>

#views.py

from django.shortcuts import render


def index(request):
    context = {'string': "I, am, OK"}
    return render(request, 'index.html', context)




#urls.py

from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
]



Detail view example

#models.py
from django.db import models
from django.core.urlresolvers import reverse

class Ancillary(models.Model):
     product_code = models.CharField(max_length=60, null=True)
     type = models.CharField(max_length=120, null=True)
     product = models.CharField(max_length=120, null=True)
     standard = models.CharField(max_length=120,   null=True)
     measurement = models.CharField(max_length=120,  null=True)
     brand = models.CharField(max_length=120,   null=True)

     class Meta:
          verbose_name_plural = "Ancillaries"
     def get_absolute_url(self):
          return reverse('ancillaries')
     def __unicode__(self):
          return u'%s %s %s %s %s %s  %s' % (self.id, self.product_code, self.type, 
                                self.product, self.standard, 
                                self.measurement, self.brand)


#urls.py
url(r'^ancillary/(?P<pk>\d+)/', AncillaryDetail.as_view(template_name='ancillary-detail.html'), name="ancillary_detail")

#view.py
class AncillaryDetail(DetailView):
    model = Ancillary

#templates/ancillary-detail.html
{{ object.product}}
{{ object.type }}
{{ object.brand }}
{{ object.measurement }}

Listview example

#views.py

class AncillaryList(ListView):
    model = Ancillary
    def get_context_data(self, **kwargs):
       context = super(AncillaryDetail, self).get_context_data(**kwargs)
       context['ancillary_list'] = Ancillary.objects.all()
       return context

#urls.py
url(r'^ancillaries/(?P<pk>\d+)/', AncillaryList.as_view(template_name='ancillary-List.html'), name="ancillary_list"),

#templates/ancillary-List.html
{% for ancillary in ancillary_list %}
    {{ ancillary.product}}
    {{ ancillary.type }}
    {{ ancillary.brand }}
    {{ ancillary.measurement }}
{% endfor %}


@@@

Migrations

Migrations are Django’s way of propagating changes you make to your models (adding a field, deleting a model, etc.) into your database schema. They’re designed to be mostly automatic, but you’ll need to know when to make migrations, when to run them, and the common problems you might run into.


The Commands
There are several commands which you will use to interact with migrations and Django’s handling of database schema:

•migrate, which is responsible for applying migrations, as well as unapplying and listing their status.
•makemigrations, which is responsible for creating new migrations based on the changes you have made to your models.
•sqlmigrate, which displays the SQL statements for a migration.


PostgreSQL
PostgreSQL is the most capable of all the databases here in terms of schema support; the only caveat is that adding columns with default values will cause a full rewrite of the table, for a time proportional to its size.

For this reason, it’s recommended you always create new columns with null=True, as this way they will be added immediately.

MySQL
MySQL lacks support for transactions around schema alteration operations, meaning that if a migration fails to apply you will have to manually unpick the changes in order to try again (it’s impossible to roll back to an earlier point).

In addition, MySQL will fully rewrite tables for almost every schema operation and generally takes a time proportional to the number of rows in the table to add or remove columns. On slower hardware this can be worse than a minute per million rows - adding a few columns to a table with just a few million rows could lock your site up for over ten minutes.

Finally, MySQL has reasonably small limits on name lengths for columns, tables and indexes, as well as a limit on the combined size of all columns an index covers. This means that indexes that are possible on other backends will fail to be created under MySQL.

SQLite
SQLite has very little built-in schema alteration support, and so Django attempts to emulate it by:

•Creating a new table with the new schema
•Copying the data across
•Dropping the old table
•Renaming the new table to match the original name
This process generally works well, but it can be slow and occasionally buggy. It is not recommended that you run and migrate SQLite in a production environment unless you are very aware of the risks and its limitations; the support Django ships with is designed to allow developers to use SQLite on their local machines to develop less complex Django projects without the need for a full database.

Workflow
Working with migrations is simple. Make changes to your models - say, add a field and remove a model - and then run makemigrations:

$ python manage.py makemigrations
Migrations for 'books':
  0003_auto.py:
    - Alter field author on book
Your models will be scanned and compared to the versions currently contained in your migration files, and then a new set of migrations will be written out. Make sure to read the output to see what makemigrations thinks you have changed - it’s not perfect, and for complex changes it might not be detecting what you expect.

Once you have your new migration files, you should apply them to your database to make sure they work as expected:

$ python manage.py migrate
Operations to perform:
  Synchronize unmigrated apps: sessions, admin, messages, auth, staticfiles, contenttypes
  Apply all migrations: books
Synchronizing apps without migrations:
  Creating tables...
  Installing custom SQL...
  Installing indexes...
Installed 0 object(s) from 0 fixture(s)
Running migrations:
  Applying books.0003_auto... OK
The command runs in two stages; first, it synchronizes unmigrated apps (performing the same functionality that syncdb used to provide), and then it runs any migrations that have not yet been applied.

Once the migration is applied, commit the migration and the models change to your version control system as a single commit - that way, when other developers (or your production servers) check out the code, they’ll get both the changes to your models and the accompanying migration at the same time.



Migration files
Migrations are stored as an on-disk format, referred to here as “migration files”. These files are actually just normal Python files with an agreed-upon object layout, written in a declarative style.

A basic migration file looks like this:

from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [("migrations", "0001_initial")]

    operations = [
        migrations.DeleteModel("Tribble"),
        migrations.AddField("Author", "rating", models.IntegerField(default=0)),
    ]
What Django looks for when it loads a migration file (as a Python module) is a subclass of django.db.migrations.Migration called Migration. It then inspects this object for four attributes, only two of which are used most of the time:

•dependencies, a list of migrations this one depends on.
•operations, a list of Operation classes that define what this migration does.
The operations are the key; they are a set of declarative instructions which tell Django what schema changes need to be made. Django scans them and builds an in-memory representation of all of the schema changes to all apps, and uses this to generate the SQL which makes the schema changes.


Adding migrations to apps

$ python manage.py makemigrations your_app_label

Note that this only works given two things:

•You have not changed your models since you made their tables. For migrations to work, you must make the initial migration first and then make changes, as Django compares changes against migration files, not the database.
•You have not manually edited your database - Django won’t be able to detect that your database doesn’t match your models, you’ll just get errors when migrations try to modify those tables.


Data Migrations
As well as changing the database schema, you can also use migrations to change the data in the database itself, in conjunction with the schema if you want.

Migrations that alter data are usually called “data migrations”; they’re best written as separate migrations, sitting alongside your schema migrations.

Django can’t automatically generate data migrations for you, as it does with schema migrations, but it’s not very hard to write them. 
Migration files in Django are made up of Operations, and the main operation you use for data migrations is RunPython.

To start, make an empty migration file you can work from (Django will put the file in the right place, suggest a name, and add dependencies for you):

python manage.py makemigrations --empty yourappname
Then, open up the file; it should look something like this:

# -*- coding: utf-8 -*-
from django.db import models, migrations

class Migration(migrations.Migration):

    dependencies = [
        ('yourappname', '0001_initial'),
    ]

    operations = [
    ]
Now, all you need to do is create a new function and have RunPython use it. 
RunPython expects a callable as its argument which takes two arguments - the first is an app registry that has the historical versions of all your models loaded into it to match where in your history the migration sits, and the second is a SchemaEditor, which you can use to manually effect database schema changes (but beware, doing this can confuse the migration autodetector!)

Let’s write a simple migration that populates our new name field with the combined values of first_name and last_name (we’ve come to our senses and realized that not everyone has first and last names). All we need to do is use the historical model and iterate over the rows:

# -*- coding: utf-8 -*-
from django.db import models, migrations

def combine_names(apps, schema_editor):
    # We can't import the Person model directly as it may be a newer
    # version than this migration expects. We use the historical version.
    Person = apps.get_model("yourappname", "Person")
    for person in Person.objects.all():
        person.name = "%s %s" % (person.first_name, person.last_name)
        person.save()

class Migration(migrations.Migration):

    dependencies = [
        ('yourappname', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(combine_names),
    ]
Once that’s done, we can just run python manage.py migrate as normal and the data migration will run in place alongside other migrations.


Be careful when running a migration with DEBUG=True as Django saves all SQL queries that are run which may result in large memory usage. This issue is addressed in Django 1.8 where only 9000 queries are saved.

You can pass a second callable to RunPython to run whatever logic you want executed when migrating backwards. If this callable is omitted, migrating backwards will raise an exception.



Accessing models from other apps

In the following example, we have a migration in app1 which needs to use models in app2. We aren’t concerned with the details of move_m1 other than the fact it will need to access models from both apps. Therefore we’ve added a dependency that specifies the last migration of app2:

class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0001_initial'),
        # added dependency to enable using models from app2 in move_m1
        ('app2', '0004_foobar'),
    ]

    operations = [
        migrations.RunPython(move_m1),
    ]


More advanced migrations
If you’re interested in the more advanced migration operations, or want to be able to write your own, see the migration operations reference.

Squashing migrations

Squashing is the act of reducing an existing set of many migrations down to one (or sometimes a few) migrations which still represent the same changes.

$ ./manage.py squashmigrations myapp 0004
Will squash the following migrations:
 - 0001_initial
 - 0002_some_change
 - 0003_another_change
 - 0004_undo_something
Do you wish to proceed? [yN] y
Optimizing...
  Optimized from 12 operations to 7 operations.
Created new squashed migration /home/andrew/Programs/DjangoTest/test/migrations/0001_squashed_0004_undo_somthing.py
  You should commit this migration but leave the old ones in place;
  the new migration will be used for new installs. Once you are sure
  all instances of the codebase have applied the migrations you squashed,
  you can delete them.
Note that model interdependencies in Django can get very complex, and squashing may result in migrations that do not run; either mis-optimized (in which case you can try again with --no-optimize, though you should also report an issue), or with a CircularDependencyError, in which case you can manually resolve it.


After this has been done, you must then transition the squashed migration to a normal initial migration, by:

•Deleting all the migration files it replaces
•Removing the replaces argument in the Migration class of the squashed migration (this is how Django tells that it is a squashed migration)
Note

Once you’ve squashed a migration, you should not then re-squash that squashed migration until you have fully transitioned it to a normal migration.



Serializing values
Migrations are just Python files containing the old definitions of your models - thus, to write them, Django must take the current state of your models and serialize them out into a file.

While Django can serialize most things, there are some things that we just can’t serialize out into a valid Python representation - there’s no Python standard for how a value can be turned back into code (repr() only works for basic values, and doesn’t specify import paths).

Django can serialize the following:

•int, long, float, bool, str, unicode, bytes, None
•list, set, tuple, dict
•datetime.date, datetime.time, and datetime.datetime instances (include those that are timezone-aware)
•decimal.Decimal instances
•Any Django field
•Any function or method reference (e.g. datetime.datetime.today) (must be in module’s top-level scope)
•Any class reference (must be in module’s top-level scope)
•Anything with a custom deconstruct() method (see below)
Support for serializing timezone-aware datetimes was added.

Django can serialize the following on Python 3 only:

•Unbound methods used from within the class body (see below)
Django cannot serialize:

•Nested classes
•Arbitrary class instances (e.g. MyClass(4.3, 5.7))
•Lambdas
Due to the fact __qualname__ was only introduced in Python 3, Django can only serialize the following pattern (an unbound method used within the class body) on Python 3, and will fail to serialize a reference to it on Python 2:

class MyModel(models.Model):

    def upload_to(self):
        return "something dynamic"

    my_file = models.FileField(upload_to=upload_to)
If you are using Python 2, we recommend you move your methods for upload_to and similar arguments that accept callables (e.g. default) to live in the main module body, rather than the class body.



Adding a deconstruct() method
You can let Django serialize your own custom class instances by giving the class a deconstruct() method. It takes no arguments, and should return a tuple of three things (path, args, kwargs):

•path should be the Python path to the class, with the class name included as the last part (for example, myapp.custom_things.MyClass). If your class is not available at the top level of a module it is not serializable.
•args should be a list of positional arguments to pass to your class’ __init__ method. Everything in this list should itself be serializable.
•kwargs should be a dict of keyword arguments to pass to your class’ __init__ method. Every value should itself be serializable.
Note

This return value is different from the deconstruct() method for custom fields which returns a tuple of four items.

Django will write out the value as an instantiation of your class with the given arguments, similar to the way it writes out references to Django fields.

To prevent a new migration from being created each time makemigrations is run, you should also add a __eq__() method to the decorated class. This function will be called by Django’s migration framework to detect changes between states.

As long as all of the arguments to your class’ constructor are themselves serializable, you can use the @deconstructible class decorator from django.utils.deconstruct to add the deconstruct() method:

from django.utils.deconstruct import deconstructible

@deconstructible
class MyCustomClass(object):

    def __init__(self, foo=1):
        self.foo = foo
        ...

    def __eq__(self, other):
        return self.foo == other.foo
The decorator adds logic to capture and preserve the arguments on their way into your constructor, and then returns those arguments exactly when deconstruct() is called.

Supporting Python 2 and 3
In order to generate migrations that support both Python 2 and 3, all string literals used in your models and fields (e.g. verbose_name, related_name, etc.), must be consistently either bytestrings or text (unicode) strings in both Python 2 and 3 (rather than bytes in Python 2 and text in Python 3, the default situation for unmarked string literals.) Otherwise running makemigrations under Python 3 will generate spurious new migrations to convert all these string attributes to text.

The easiest way to achieve this is to follow the advice in Django’s Python 3 porting guide and make sure that all your modules begin with from __future__ import unicode_literals, so that all unmarked string literals are always unicode, regardless of Python version. When you add this to an app with existing migrations generated on Python 2, your next run of makemigrations on Python 3 will likely generate many changes as it converts all the bytestring attributes to text strings; this is normal and should only happen once.















Managing database transactions
Django’s default transaction behavior
Django’s default behavior is to run in autocommit mode. Each query is immediately committed to the database, unless a transaction is active. 

Django uses transactions or savepoints automatically to guarantee the integrity of ORM operations that require multiple queries, especially delete() and update() queries.

Django’s TestCase class also wraps each test in a transaction for performance reasons.


Tying transactions to HTTP requests
A common way to handle transactions on the web is to wrap each request in a transaction. Set ATOMIC_REQUESTS to True in the configuration of each database for which you want to enable this behavior.

It works like this. Before calling a view function, Django starts a transaction. If the response is produced without problems, Django commits the transaction. If the view produces an exception, Django rolls back the transaction.

You may perform partial commits and rollbacks in your view code, typically with the atomic() context manager. However, at the end of the view, either all the changes will be committed, or none of them.

Warning

While the simplicity of this transaction model is appealing, it also makes it inefficient when traffic increases. Opening a transaction for every view has some overhead. The impact on performance depends on the query patterns of your application and on how well your database handles locking.



Per-request transactions and streaming responses

When a view returns a StreamingHttpResponse, reading the contents of the response will often execute code to generate the content. Since the view has already returned, such code runs outside of the transaction.

Generally speaking, it isn’t advisable to write to the database while generating a streaming response, since there’s no sensible way to handle errors after starting to send the response.

In practice, this feature simply wraps every view function in the atomic() decorator described below.

Note that only the execution of your view is enclosed in the transactions. Middleware runs outside of the transaction, and so does the rendering of template responses.

When ATOMIC_REQUESTS is enabled, it’s still possible to prevent views from running in a transaction.

non_atomic_requests(using=None)[source]
This decorator will negate the effect of ATOMIC_REQUESTS for a given view:

from django.db import transaction

@transaction.non_atomic_requests
def my_view(request):
    do_stuff()

@transaction.non_atomic_requests(using='other')
def my_other_view(request):
    do_stuff_on_the_other_database()
It only works if it’s applied to the view itself.


Controlling transactions explicitly
Django provides a single API to control database transactions.

atomic(using=None, savepoint=True)
Atomicity is the defining property of database transactions. atomic allows us to create a block of code within which the atomicity on the database is guaranteed. If the block of code is successfully completed, the changes are committed to the database. If there is an exception, the changes are rolled back.

atomic blocks can be nested. In this case, when an inner block completes successfully, its effects can still be rolled back if an exception is raised in the outer block at a later point.

atomic is usable both as a decorator:

from django.db import transaction

@transaction.atomic
def viewfunc(request):
    # This code executes inside a transaction.
    do_stuff()
and as a context manager:

from django.db import transaction

def viewfunc(request):
    # This code executes in autocommit mode (Django's default).
    do_stuff()

    with transaction.atomic():
        # This code executes inside a transaction.
        do_more_stuff()
Wrapping atomic in a try/except block allows for natural handling of integrity errors:

from django.db import IntegrityError, transaction

@transaction.atomic
def viewfunc(request):
    create_parent()

    try:
        with transaction.atomic():
            generate_relationships()
    except IntegrityError:
        handle_exception()

    add_children()
In this example, even if generate_relationships() causes a database error by breaking an integrity constraint, you can execute queries in add_children(), and the changes from create_parent() are still there. Note that any operations attempted in generate_relationships() will already have been rolled back safely when handle_exception() is called, so the exception handler can also operate on the database if necessary.





get_autocommit(using=None)
set_autocommit(autocommit, using=None)
These functions take a using argument which should be the name of a database. If it isn’t provided, Django uses the "default" database.

Autocommit is initially turned on. If you turn it off, it’s your responsibility to restore it.

Once you turn autocommit off, you get the default behavior of your database adapter, and Django won’t help you. Although that behavior is specified in PEP 249, implementations of adapters aren’t always consistent with one another. Review the documentation of the adapter you’re using carefully.

You must ensure that no transaction is active, usually by issuing a commit() or a rollback(), before turning autocommit back on.

Django will refuse to turn autocommit off when an atomic() block is active, because that would break atomicity.



commit(using=None)[source]
rollback(using=None)[source]
These functions take a using argument which should be the name of a database. If it isn’t provided, Django uses the "default" database.

Django will refuse to commit or to rollback when an atomic() block is active, because that would break atomicity.


savepoint(using=None)
Creates a new savepoint. This marks a point in the transaction that is known to be in a “good” state. Returns the savepoint ID (sid).
A savepoint is a marker within a transaction that enables you to roll back part of a transaction, rather than the full transaction. Savepoints are available with the SQLite (= 3.6.8), PostgreSQL, Oracle and MySQL (when using the InnoDB storage engine) backends. Other backends provide the savepoint functions, but they’re empty operations – they don’t actually do anything.

Savepoints aren’t especially useful if you are using autocommit, the default behavior of Django. However, once you open a transaction with atomic(), you build up a series of database operations awaiting a commit or rollback. If you issue a rollback, the entire transaction is rolled back. Savepoints provide the ability to perform a fine-grained rollback, rather than the full rollback that would be performed by transaction.rollback().




savepoint_commit(sid, using=None)[source]
Releases savepoint sid. The changes performed since the savepoint was created become part of the transaction.

savepoint_rollback(sid, using=None)[source]
Rolls back the transaction to savepoint sid.

These functions do nothing if savepoints aren’t supported or if the database is in autocommit mode.

In addition, there’s a utility function:

clean_savepoints(using=None)[source]
Resets the counter used to generate unique savepoint IDs.

Example:

from django.db import transaction

# open a transaction
@transaction.atomic
def viewfunc(request):

    a.save()
    # transaction now contains a.save()

    sid = transaction.savepoint()

    b.save()
    # transaction now contains a.save() and b.save()

    if want_to_keep_b:
        transaction.savepoint_commit(sid)
        # open transaction still contains a.save() and b.save()
    else:
        transaction.savepoint_rollback(sid)
        # open transaction now contains only a.save()
Savepoints may be used to recover from a database error by performing a partial rollback. If you’re doing this inside an atomic() block, the entire block will still be rolled back, because it doesn’t know you’ve handled the situation at a lower level! To prevent this, you can control the rollback behavior with the following functions.



get_rollback(using=None)[source]
set_rollback(rollback, using=None)[source]
Setting the rollback flag to True forces a rollback when exiting the innermost atomic block. This may be useful to trigger a rollback without raising an exception.

Setting it to False prevents such a rollback. Before doing that, make sure you’ve rolled back the transaction to a known-good savepoint within the current atomic block! Otherwise you’re breaking atomicity and data corruption may occur.

Savepoints in SQLite
While SQLite = 3.6.8 supports savepoints, a flaw in the design of the sqlite3 module makes them hardly usable.

When autocommit is enabled, savepoints don’t make sense. When it’s disabled, sqlite3 commits implicitly before savepoint statements. (In fact, it commits before any statement other than SELECT, INSERT, UPDATE, DELETE and REPLACE.) This bug has two consequences:

•The low level APIs for savepoints are only usable inside a transaction ie. inside an atomic() block.
•It’s impossible to use atomic() when autocommit is turned off.

Transactions in MySQL
If you’re using MySQL, your tables may or may not support transactions; it depends on your MySQL version and the table types you’re using. (By “table types,” we mean something like “InnoDB” or “MyISAM”.) MySQL transaction peculiarities are outside the scope of this article, but the MySQL site has information on MySQL transactions.

If your MySQL setup does not support transactions, then Django will always function in autocommit mode: statements will be executed and committed as soon as they’re called. If your MySQL setup does support transactions, Django will handle transactions as explained in this document.



Transaction rollback
The first option is to roll back the entire transaction. For example:

a.save() # Succeeds, but may be undone by transaction rollback
try:
    b.save() # Could throw exception
except IntegrityError:
    transaction.rollback()
c.save() # Succeeds, but a.save() may have been undone
Calling transaction.rollback() rolls back the entire transaction. Any uncommitted database operations will be lost. In this example, the changes made by a.save() would be lost, even though that operation raised no error itself.


Savepoint rollback
You can use savepoints to control the extent of a rollback. Before performing a database operation that could fail, you can set or update the savepoint; that way, if the operation fails, you can roll back the single offending operation, rather than the entire transaction. For example:

a.save() # Succeeds, and never undone by savepoint rollback
sid = transaction.savepoint()
try:
    b.save() # Could throw exception
    transaction.savepoint_commit(sid)
except IntegrityError:
    transaction.savepoint_rollback(sid)
c.save() # Succeeds, and a.save() is never undone
In this example, a.save() will not be undone in the case where b.save() raises an exception.









Multiple databases

DATABASES = {
    'default': {
        'NAME': 'app_data',
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'USER': 'postgres_user',
        'PASSWORD': 's3krit'
    },
    'users': {
        'NAME': 'user_data',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'priv4te'
    }
}
If the concept of a default database doesn’t make sense in the context of your project, you need to be careful to always specify the database that you want to use. Django requires that a default database entry be defined, but the parameters dictionary can be left blank if it will not be used. You must setup DATABASE_ROUTERS for all of your apps’ models, including those in any contrib and third-party apps you are using, so that no queries are routed to the default database in order to do this. The following is an example settings.py snippet defining two non-default databases, with the default entry intentionally left empty:

DATABASES = {
    'default': {},
    'users': {
        'NAME': 'user_data',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'superS3cret'
    },
    'customers': {
        'NAME': 'customer_data',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_cust',
        'PASSWORD': 'veryPriv@ate'
    }
}
If you attempt to access a database that you haven’t defined in your DATABASES setting, Django will raise a django.db.utils.ConnectionDoesNotExist exception.

Synchronizing your databases
The migrate management command operates on one database at a time. By default, it operates on the default database, but by providing a --database argument, you can tell migrate to synchronize a different database. So, to synchronize all models onto all databases in our example, you would need to call:

$ ./manage.py migrate
$ ./manage.py migrate --database=users
If you don’t want every application to be synchronized onto a particular database, you can define a database router that implements a policy constraining the availability of particular models.


Using other management commands
The other django-admin.py commands that interact with the database operate in the same way as migrate – they only ever operate on one database at a time, using --database to control the database used.


Automatic database routing
The easiest way to use multiple databases is to set up a database routing scheme. The default routing scheme ensures that objects remain ‘sticky’ to their original database (i.e., an object retrieved from the foo database will be saved on the same database). The default routing scheme ensures that if a database isn’t specified, all queries fall back to the default database.

You don’t have to do anything to activate the default routing scheme – it is provided ‘out of the box’ on every Django project. However, if you want to implement more interesting database allocation behaviors, you can define and install your own database routers.



Database routers
A database Router is a class that provides up to four methods:

db_for_read(model, **hints)
Suggest the database that should be used for read operations for objects of type model.

If a database operation is able to provide any additional information that might assist in selecting a database, it will be provided in the hints dictionary. Details on valid hints are provided below.

Returns None if there is no suggestion.

db_for_write(model, **hints)
Suggest the database that should be used for writes of objects of type Model.

If a database operation is able to provide any additional information that might assist in selecting a database, it will be provided in the hints dictionary. Details on valid hints are provided below.

Returns None if there is no suggestion.

allow_relation(obj1, obj2, **hints)
Return True if a relation between obj1 and obj2 should be allowed, False if the relation should be prevented, or None if the router has no opinion. This is purely a validation operation, used by foreign key and many to many operations to determine if a relation should be allowed between two objects.

allow_migrate(db, model)
Determine if the model should have tables/indexes created in the database with alias db. Return True if the model should be migrated, False if it should not be migrated, or None if the router has no opinion. This method can be used to determine the availability of a model on a given database.

Note that migrations will just silently not perform any operations on a model for which this returns False. This may result in broken ForeignKeys, extra tables or missing tables if you change it once you have applied some migrations.

The value passed for model may be a historical model, and thus not have any custom attributes, methods or managers. You should only rely on _meta.

A router doesn’t have to provide all these methods – it may omit one or more of them. If one of the methods is omitted, Django will skip that router when performing the relevant check.




Using routers
Database routers are installed using the DATABASE_ROUTERS setting. This setting defines a list of class names, each specifying a router that should be used by the master router (django.db.router).

The master router is used by Django’s database operations to allocate database usage. Whenever a query needs to know which database to use, it calls the master router, providing a model and a hint (if available). Django then tries each router in turn until a database suggestion can be found. If no suggestion can be found, it tries the current _state.db of the hint instance. If a hint instance wasn’t provided, or the instance doesn’t currently have database state, the master router will allocate the default database.

An example

DATABASES = {
    'auth_db': {
        'NAME': 'auth_db',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'swordfish',
    },
    'master': {
        'NAME': 'master',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'spam',
    },
    'slave1': {
        'NAME': 'slave1',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'eggs',
    },
    'slave2': {
        'NAME': 'slave2',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'bacon',
    },
}
Now we’ll need to handle routing. First we want a router that knows to send queries for the auth app to auth_db:

class AuthRouter(object):
    """
    A router to control all database operations on models in the
    auth application.
    """
    def db_for_read(self, model, **hints):
        """
        Attempts to read auth models go to auth_db.
        """
        if model._meta.app_label == 'auth':
            return 'auth_db'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write auth models go to auth_db.
        """
        if model._meta.app_label == 'auth':
            return 'auth_db'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the auth app is involved.
        """
        if obj1._meta.app_label == 'auth' or \
           obj2._meta.app_label == 'auth':
           return True
        return None

    def allow_migrate(self, db, model):
        """
        Make sure the auth app only appears in the 'auth_db'
        database.
        """
        if db == 'auth_db':
            return model._meta.app_label == 'auth'
        elif model._meta.app_label == 'auth':
            return False
        return None
And we also want a router that sends all other apps to the master/slave configuration, and randomly chooses a slave to read from:

import random

class MasterSlaveRouter(object):
    def db_for_read(self, model, **hints):
        """
        Reads go to a randomly-chosen slave.
        """
        return random.choice(['slave1', 'slave2'])

    def db_for_write(self, model, **hints):
        """
        Writes always go to master.
        """
        return 'master'

    def allow_relation(self, obj1, obj2, **hints):
        """
        Relations between objects are allowed if both objects are
        in the master/slave pool.
        """
        db_list = ('master', 'slave1', 'slave2')
        if obj1._state.db in db_list and obj2._state.db in db_list:
            return True
        return None

    def allow_migrate(self, db, model):
        """
        All non-auth models end up in this pool.
        """
        return True
Finally, in the settings file, we add the following (substituting path.to. with the actual python path to the module(s) where the routers are defined):

DATABASE_ROUTERS = ['path.to.AuthRouter', 'path.to.MasterSlaveRouter']
The order in which routers are processed is significant. Routers will be queried in the order the are listed in the DATABASE_ROUTERS setting . In this example, the AuthRouter is processed before the MasterSlaveRouter, and as a result, decisions concerning the models in auth are processed before any other decision is made. If the DATABASE_ROUTERS setting listed the two routers in the other order, MasterSlaveRouter.allow_migrate() would be processed first. The catch-all nature of the MasterSlaveRouter implementation would mean that all models would be available on all databases.

With this setup installed, lets run some Django code:

# This retrieval will be performed on the 'auth_db' database
fred = User.objects.get(username='fred')
fred.first_name = 'Frederick'

# This save will also be directed to 'auth_db'
fred.save()

# These retrieval will be randomly allocated to a slave database
dna = Person.objects.get(name='Douglas Adams')

# A new object has no database allocation when created
mh = Book(title='Mostly Harmless')

# This assignment will consult the router, and set mh onto
# the same database as the author object
mh.author = dna

# This save will force the 'mh' instance onto the master database...
mh.save()

# ... but if we re-retrieve the object, it will come back on a slave
mh = Book.objects.get(title='Mostly Harmless')




Manually selecting a database for a QuerySet
You can select the database for a QuerySet at any point in the QuerySet “chain.” Just call using() on the QuerySet to get another QuerySet that uses the specified database.

using() takes a single argument: the alias of the database on which you want to run the query. For example:

# This will run on the 'default' database.
Author.objects.all()

# So will this.
Author.objects.using('default').all()

# This will run on the 'other' database.
Author.objects.using('other').all()


Selecting a database for save()
Use the using keyword to Model.save() to specify to which database the data should be saved.

For example, to save an object to the legacy_users database, you’d use this:

my_object.save(using='legacy_users')
If you don’t specify using, the save() method will save into the default database allocated by the routers.

Moving an object from one database to another
If you’ve saved an instance to one database, it might be tempting to use save(using=...) as a way to migrate the instance to a new database. However, if you don’t take appropriate steps, this could have some unexpected consequences.

Consider the following example:

p = Person(name='Fred')
p.save(using='first')  # (statement 1)
p.save(using='second') # (statement 2)
In statement 1, a new Person object is saved to the first database. At this time, p doesn’t have a primary key, so Django issues an SQL INSERT statement. This creates a primary key, and Django assigns that primary key to p.

When the save occurs in statement 2, p already has a primary key value, and Django will attempt to use that primary key on the new database. If the primary key value isn’t in use in the second database, then you won’t have any problems – the object will be copied to the new database.

However, if the primary key of p is already in use on the second database, the existing object in the second database will be overridden when p is saved.

You can avoid this in two ways. First, you can clear the primary key of the instance. If an object has no primary key, Django will treat it as a new object, avoiding any loss of data on the second database:

p = Person(name='Fred')
p.save(using='first')
p.pk = None # Clear the primary key.
p.save(using='second') # Write a completely new object.
The second option is to use the force_insert option to save() to ensure that Django does an SQL INSERT:

p = Person(name='Fred')
p.save(using='first')
p.save(using='second', force_insert=True)
This will ensure that the person named Fred will have the same primary key on both databases. If that primary key is already in use when you try to save onto the second database, an error will be raised.



Selecting a database to delete from
By default, a call to delete an existing object will be executed on the same database that was used to retrieve the object in the first place:

u = User.objects.using('legacy_users').get(username='fred')
u.delete() # will delete from the `legacy_users` database
To specify the database from which a model will be deleted, pass a using keyword argument to the Model.delete() method. This argument works just like the using keyword argument to save().

For example, if you’re migrating a user from the legacy_users database to the new_users database, you might use these commands:

user_obj.save(using='new_users')
user_obj.delete(using='legacy_users')
Using managers with multiple databases
Use the db_manager() method on managers to give managers access to a non-default database.

For example, say you have a custom manager method that touches the database – User.objects.create_user(). Because create_user() is a manager method, not a QuerySet method, you can’t do User.objects.using('new_users').create_user(). (The create_user() method is only available on User.objects, the manager, not on QuerySet objects derived from the manager.) The solution is to use db_manager(), like this:

User.objects.db_manager('new_users').create_user(...)
db_manager() returns a copy of the manager bound to the database you specify.




Using get_queryset() with multiple databases
If you’re overriding get_queryset() on your manager, be sure to either call the method on the parent (using super()) or do the appropriate handling of the _db attribute on the manager (a string containing the name of the database to use).

For example, if you want to return a custom QuerySet class from the get_queryset method, you could do this:

class MyManager(models.Manager):
    def get_queryset(self):
        qs = CustomQuerySet(self.model)
        if self._db is not None:
            qs = qs.using(self._db)
        return qs
Exposing multiple databases in Django’s admin interface
Django’s admin doesn’t have any explicit support for multiple databases. If you want to provide an admin interface for a model on a database other than that specified by your router chain, you’ll need to write custom ModelAdmin classes that will direct the admin to use a specific database for content.

ModelAdmin objects have five methods that require customization for multiple-database support:

class MultiDBModelAdmin(admin.ModelAdmin):
    # A handy constant for the name of the alternate database.
    using = 'other'

    def save_model(self, request, obj, form, change):
        # Tell Django to save objects to the 'other' database.
        obj.save(using=self.using)

    def delete_model(self, request, obj):
        # Tell Django to delete objects from the 'other' database
        obj.delete(using=self.using)

    def get_queryset(self, request):
        # Tell Django to look for objects on the 'other' database.
        return super(MultiDBModelAdmin, self).get_queryset(request).using(self.using)

    def formfield_for_foreignkey(self, db_field, request=None, **kwargs):
        # Tell Django to populate ForeignKey widgets using a query
        # on the 'other' database.
        return super(MultiDBModelAdmin, self).formfield_for_foreignkey(db_field, request=request, using=self.using, **kwargs)

    def formfield_for_manytomany(self, db_field, request=None, **kwargs):
        # Tell Django to populate ManyToMany widgets using a query
        # on the 'other' database.
        return super(MultiDBModelAdmin, self).formfield_for_manytomany(db_field, request=request, using=self.using, **kwargs)
The implementation provided here implements a multi-database strategy where all objects of a given type are stored on a specific database (e.g., all User objects are in the other database). If your usage of multiple databases is more complex, your ModelAdmin will need to reflect that strategy.

Inlines can be handled in a similar fashion. They require three customized methods:

class MultiDBTabularInline(admin.TabularInline):
    using = 'other'

    def get_queryset(self, request):
        # Tell Django to look for inline objects on the 'other' database.
        return super(MultiDBTabularInline, self).get_queryset(request).using(self.using)

    def formfield_for_foreignkey(self, db_field, request=None, **kwargs):
        # Tell Django to populate ForeignKey widgets using a query
        # on the 'other' database.
        return super(MultiDBTabularInline, self).formfield_for_foreignkey(db_field, request=request, using=self.using, **kwargs)

    def formfield_for_manytomany(self, db_field, request=None, **kwargs):
        # Tell Django to populate ManyToMany widgets using a query
        # on the 'other' database.
        return super(MultiDBTabularInline, self).formfield_for_manytomany(db_field, request=request, using=self.using, **kwargs)
Once you’ve written your model admin definitions, they can be registered with any Admin instance:

from django.contrib import admin

# Specialize the multi-db admin objects for use with specific models.
class BookInline(MultiDBTabularInline):
    model = Book

class PublisherAdmin(MultiDBModelAdmin):
    inlines = [BookInline]

admin.site.register(Author, MultiDBModelAdmin)
admin.site.register(Publisher, PublisherAdmin)

othersite = admin.AdminSite('othersite')
othersite.register(Publisher, MultiDBModelAdmin)
This example sets up two admin sites. On the first site, the Author and Publisher objects are exposed; Publisher objects have an tabular inline showing books published by that publisher. The second site exposes just publishers, without the inlines.



Using raw cursors with multiple databases
If you are using more than one database you can use django.db.connections to obtain the connection (and cursor) for a specific database. django.db.connections is a dictionary-like object that allows you to retrieve a specific connection using its alias:

from django.db import connections
cursor = connections['my_db_alias'].cursor()


Limitations of multiple databases
Cross-database relations
Django doesn’t currently provide any support for foreign key or many-to-many relationships spanning multiple databases. If you have used a router to partition models to different databases, any foreign key and many-to-many relationships defined by those models must be internal to a single database.






Tablespaces
A common paradigm for optimizing performance in database systems is the use of tablespaces to organize disk layout.


Django does not create the tablespaces for you. Please refer to your database engine’s documentation for details on creating and managing tablespaces.



Declaring tablespaces for tables
A tablespace can be specified for the table generated by a model by supplying the db_tablespace option inside the model’s class Meta. This option also affects tables automatically created for ManyToManyFields in the model.

You can use the DEFAULT_TABLESPACE setting to specify a default value for db_tablespace. This is useful for setting a tablespace for the built-in Django apps and other applications whose code you cannot control.



Declaring tablespaces for indexes
You can pass the db_tablespace option to a Field constructor to specify an alternate tablespace for the Field’s column index. If no index would be created for the column, the option is ignored.

You can use the DEFAULT_INDEX_TABLESPACE setting to specify a default value for db_tablespace.

If db_tablespace isn’t specified and you didn’t set DEFAULT_INDEX_TABLESPACE, the index is created in the same tablespace as the tables.

An example
class TablespaceExample(models.Model):
    name = models.CharField(max_length=30, db_index=True, db_tablespace="indexes")
    data = models.CharField(max_length=255, db_index=True)
    edges = models.ManyToManyField(to="self", db_tablespace="indexes")

    class Meta:
        db_tablespace = "tables"
In this example, the tables generated by the TablespaceExample model (i.e. the model table and the many-to-many table) would be stored in the tables tablespace. The index for the name field and the indexes on the many-to-many table would be stored in the indexes tablespace. The data field would also generate an index, but no tablespace for it is specified, so it would be stored in the model tablespace tables by default.

Database support
PostgreSQL and Oracle support tablespaces. SQLite and MySQL don’t.

When you use a backend that lacks support for tablespaces, Django ignores all tablespace-related options.









Database access optimization

Profile first
Use standard DB optimization techniques
...including:

•Indexes. This is a number one priority, after you have determined from profiling what indexes should be added. Use Field.db_index or Meta.index_together to add these from Django. Consider adding indexes to fields that you frequently query using filter(), exclude(), order_by(), etc. as indexes may help to speed up lookups. Note that determining the best indexes is a complex database-dependent topic that will depend on your particular application. The overhead of maintaining an index may outweigh any gains in query speed.
•Appropriate use of field types.


Understand QuerySet evaluation
To avoid performance problems, it is important to understand:

•that QuerySets are lazy.
•when they are evaluated.
•how the data is held in memory.


Understand cached attributes
As well as caching of the whole QuerySet, there is caching of the result of attributes on ORM objects. In general, attributes that are not callable will be cached. For example, assuming the example Weblog models:

entry = Entry.objects.get(id=1)
entry.blog   # Blog object is retrieved at this point
entry.blog   # cached version, no DB access
But in general, callable attributes cause DB lookups every time:

entry = Entry.objects.get(id=1)
entry.authors.all()   # query performed
entry.authors.all()   # query performed again
Be careful when reading template code - the template system does not allow use of parentheses, but will call callables automatically, hiding the above distinction.

Be careful with your own custom properties - it is up to you to implement caching when required, for example using the cached_property decorator.


Use the with template tag
To make use of the caching behavior of QuerySet, you may need to use the with template tag.

Use iterator()
When you have a lot of objects, the caching behavior of the QuerySet can cause a large amount of memory to be used. In this case, iterator() may help.


Do database work in the database rather than in Python
For instance:

•At the most basic level, use filter and exclude to do filtering in the database.
•Use F expressions to filter based on other fields within the same model.
•Use annotate to do aggregation in the database.
If these aren’t enough to generate the SQL you need:


Use QuerySet.extra()
A less portable but more powerful method is extra(), which allows some SQL to be explicitly added to the query. If that still isn’t powerful enough:


Use raw SQL
Write your own custom SQL to retrieve data or populate models. Use django.db.connection.queries to find out what Django is writing for you and start from there.


Retrieve individual objects using a unique, indexed column
There are two reasons to use a column with unique or db_index when using get() to retrieve individual objects. First, the query will be quicker because of the underlying database index. Also, the query could run much slower if multiple objects match the lookup; having a unique constraint on the column guarantees this will never happen.

So using the example Weblog models:

entry = Entry.objects.get(id=10)
will be quicker than:

entry = Entry.object.get(headline="News Item Title")
because id is indexed by the database and is guaranteed to be unique.

Doing the following is potentially quite slow:

entry = Entry.objects.get(headline__startswith="News")
First of all, headline is not indexed, which will make the underlying database fetch slower.

Second, the lookup doesn’t guarantee that only one object will be returned. If the query matches more than one object, it will retrieve and transfer all of them from the database. This penalty could be substantial if hundreds or thousands of records are returned. The penalty will be compounded if the database lives on a separate server, where network overhead and latency also play a factor.

Retrieve everything at once if you know you will need it
Hitting the database multiple times for different parts of a single ‘set’ of data that you will need all parts of is, in general, less efficient than retrieving it all in one query. This is particularly important if you have a query that is executed in a loop, and could therefore end up doing many database queries, when only one was needed. So:

Use QuerySet.select_related() and prefetch_related()
Understand select_related() and prefetch_related() thoroughly, and use them:

•in view code,
•and in managers and default managers where appropriate. Be aware when your manager is and is not used; sometimes this is tricky so don’t make assumptions.


Don’t retrieve things you don’t need
Use QuerySet.values() and values_list()
When you just want a dict or list of values, and don’t need ORM model objects, make appropriate usage of values(). These can be useful for replacing model objects in template code - as long as the dicts you supply have the same attributes as those used in the template, you are fine.

Use QuerySet.defer() and only()
Use defer() and only() if there are database columns you know that you won’t need (or won’t need in most cases) to avoid loading them. Note that if you do use them, the ORM will have to go and get them in a separate query, making this a pessimization if you use it inappropriately.

Also, be aware that there is some (small extra) overhead incurred inside Django when constructing a model with deferred fields. Don’t be too aggressive in deferring fields without profiling as the database has to read most of the non-text, non-VARCHAR data from the disk for a single row in the results, even if it ends up only using a few columns. The defer() and only() methods are most useful when you can avoid loading a lot of text data or for fields that might take a lot of processing to convert back to Python. As always, profile first, then optimize.

Use QuerySet.count()
...if you only want the count, rather than doing len(queryset).

Use QuerySet.exists()
...if you only want to find out if at least one result exists, rather than if queryset.

But:

Don’t overuse count() and exists()
If you are going to need other data from the QuerySet, just evaluate it.

For example, assuming an Email model that has a body attribute and a many-to-many relation to User, the following template code is optimal:

{% if display_inbox %}
  {% with emails=user.emails.all %}
    {% if emails %}
      <p>You have {{ emails|length }} email(s)</p>
      {% for email in emails %}
        <p>{{ email.body }}</p>
      {% endfor %}
    {% else %}
      <p>No messages today.</p>
    {% endif %}
  {% endwith %}
{% endif %}
It is optimal because:

1.Since QuerySets are lazy, this does no database queries if ‘display_inbox’ is False.
2.Use of with means that we store user.emails.all in a variable for later use, allowing its cache to be re-used.
3.The line {% if emails %} causes QuerySet.__bool__() to be called, which causes the user.emails.all() query to be run on the database, and at the least the first line to be turned into an ORM object. If there aren’t any results, it will return False, otherwise True.
4.The use of {{ emails|length }} calls QuerySet.__len__(), filling out the rest of the cache without doing another query.
5.The for loop iterates over the already filled cache.
In total, this code does either one or zero database queries. The only deliberate optimization performed is the use of the with tag. Using QuerySet.exists() or QuerySet.count() at any point would cause additional queries.



Use QuerySet.update() and delete()
Rather than retrieve a load of objects, set some values, and save them individual, use a bulk SQL UPDATE statement, via QuerySet.update(). Similarly, do bulk deletes where possible.

Note, however, that these bulk update methods cannot call the save() or delete() methods of individual instances, which means that any custom behavior you have added for these methods will not be executed, including anything driven from the normal database object signals.



Use foreign key values directly
If you only need a foreign key value, use the foreign key value that is already on the object you’ve got, rather than getting the whole related object and taking its primary key. i.e. do:

entry.blog_id
instead of:

entry.blog.id


Don’t order results if you don’t care
Ordering is not free; each field to order by is an operation the database must perform. If a model has a default ordering (Meta.ordering) and you don’t need it, remove it on a QuerySet by calling order_by() with no parameters.

Adding an index to your database may help to improve ordering performance.



Insert in bulk
When creating objects, where possible, use the bulk_create() method to reduce the number of SQL queries. For example:

Entry.objects.bulk_create([
    Entry(headline="Python 3.0 Released"),
    Entry(headline="Python 3.1 Planned")
])
...is preferable to:

Entry.objects.create(headline="Python 3.0 Released")
Entry.objects.create(headline="Python 3.1 Planned")
Note that there are a number of caveats to this method, so make sure it’s appropriate for your use case.

This also applies to ManyToManyFields, so doing:

my_band.members.add(me, my_friend)
...is preferable to:

my_band.members.add(me)
my_band.members.add(my_friend)
...where Bands and Artists have a many-to-many relationship.








Integrating Django with a legacy database


Auto-generate the models
Django comes with a utility called inspectdb that can create models by introspecting an existing database. You can view the output by running this command:

$ python manage.py inspectdb
Save this as a file by using standard Unix output redirection:

$ python manage.py inspectdb > models.py
This feature is meant as a shortcut, not as definitive model generation. See the documentation of inspectdb for more information.

Once you’ve cleaned up your models, name the file models.py and put it in the Python package that holds your app. Then add the app to your INSTALLED_APPS setting.

By default, inspectdb creates unmanaged models. That is, managed = False in the model’s Meta class tells Django not to manage each table’s creation, modification, and deletion:

class Person(models.Model):
    id = models.IntegerField(primary_key=True)
    first_name = models.CharField(max_length=70)
    class Meta:
       managed = False
       db_table = 'CENSUS_PERSONS'
If you do want to allow Django to manage the table’s lifecycle, you’ll need to change the managed option above to True (or simply remove it because True is its default value).



Install the core Django tables
Next, run the migrate command to install any extra needed database records such as admin permissions and content types:

$ python manage.py migrate






Providing initial data with fixtures
A fixture is a collection of data that Django knows how to import into a database. The most straightforward way of creating a fixture if you’ve already got some data is to use the manage.py dumpdata command. Or, you can write fixtures by hand; fixtures can be written as JSON, XML or YAML (with PyYAML installed) documents. The serialization documentation has more details about each of these supported serialization formats.

As an example, though, here’s what a fixture for a simple Person model might look like in JSON:

[
  {
    "model": "myapp.person",
    "pk": 1,
    "fields": {
      "first_name": "John",
      "last_name": "Lennon"
    }
  },
  {
    "model": "myapp.person",
    "pk": 2,
    "fields": {
      "first_name": "Paul",
      "last_name": "McCartney"
    }
  }
]
And here’s that same fixture as YAML:

- model: myapp.person
  pk: 1
  fields:
    first_name: John
    last_name: Lennon
- model: myapp.person
  pk: 2
  fields:
    first_name: Paul
    last_name: McCartney
You’ll store this data in a fixtures directory inside your app.

Loading data is easy: just call manage.py loaddata <fixturename>, where <fixturename> is the name of the fixture file you’ve created. 
Each time you run loaddata, the data will be read from the fixture and re-loaded into the database. Note this means that if you change one of the rows created by a fixture and then run loaddata again, you’ll wipe out any changes you’ve made.



Providing initial SQL data

Django provides a hook for passing the database arbitrary SQL that’s executed just after the CREATE TABLE statements when you run migrate. You can use this hook to populate default records, or you could also create SQL functions, views, triggers, etc.

The hook is simple: Django just looks for a file called sql/<modelname>.sql, in your app directory, where <modelname> is the model’s name in lowercase.

So, if you had a Person model in an app called myapp, you could add arbitrary SQL to the file sql/person.sql inside your myapp directory. Here’s an example of what the file might contain:

INSERT INTO myapp_person (first_name, last_name) VALUES ('John', 'Lennon');
INSERT INTO myapp_person (first_name, last_name) VALUES ('Paul', 'McCartney');
Each SQL file, if given, is expected to contain valid SQL statements which will insert the desired data (e.g., properly-formatted INSERT statements separated by semicolons).

The SQL files are read by the sqlcustom and sqlall commands in manage.py. Refer to the manage.py documentation for more information.

Note that if you have multiple SQL data files, there’s no guarantee of the order in which they’re executed. The only thing you can assume is that, by the time your custom data files are executed, all the database tables already will have been created.




