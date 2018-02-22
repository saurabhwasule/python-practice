
#-------------------------------------------

#Re Expression 
#Advanced Sorting 
#Web Page & CGI Programming
#Database Access
#SMTP & Emailing
#CSV handling - std module 
#xml.etree.ElementTree — The ElementTree XML API (same in Py3.x and Py2.7)
#Apache requests 
#Module Json 
#Python Standard Library – datetime


#******************************************
###Re Expression 
##match() vs search()
#re.match() checks for a match only at the beginning 
#of the string, 
#while re.search() checks for a match anywhere in the string 


>>> re.match("c", "abcdef")    # No match
>>> re.search("c", "abcdef")   # Match
<_sre.SRE_Match object at ...>


#Regular expressions beginning with '^' can be used 
#with search() to restrict the match at the beginning of the string:


>>> re.match("c", "abcdef")    # No match
>>> re.search("^c", "abcdef")  # No match
>>> re.search("^a", "abcdef")  # Match
<_sre.SRE_Match object at ...>


#in MULTILINE mode match() only matches 
#at the beginning of the string, 
#whereas using search() with a regular expression 
#beginning with '^' will match at the beginning of each line.


>>> re.match('X', 'A\nB\nX', re.MULTILINE)  # No match
>>> re.search('^X', 'A\nB\nX', re.MULTILINE)  # Match
<_sre.SRE_Match object at ...>

##sub with repl as function 
#it is called for every non-overlapping occurrence of pattern. 
#The function takes a single match object argument, 
#and returns the replacement string


def dashrepl(matchobj):
    if matchobj.group(0) == '-': return ' '
    else: return '-'
>>> re.sub('-{1,2}', dashrepl, 'pro----gram-files')
'pro--gram files'


##Flags 
re.DEBUG        Display debug information about compiled expression.
re.I            re.IGNORECASE
re.L            re.LOCALE, Make \w, \W, \b, \B, \s and \S dependent on the current locale.
re.M            re.MULTILINE  ^,$ for each newline 
re.S            re.DOTALL   . matches newline 
re.X            re.VERBOSE

a = re.compile(r"""\d +  # the integral part
                   \.    # the decimal point
                   \d *  # some fractional digits""", re.X)
b = re.compile(r"\d+\.\d*")



##Match Object 
match = re.search(pattern, string)
if match:
    process(match)
    
#meaning of group 
>>> m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
>>> m.group(0)       # The entire match
'Isaac Newton'
>>> m.group(1)       # The first parenthesized subgroup.
'Isaac'
>>> m.group(2)       # The second parenthesized subgroup.
'Newton'
>>> m.group(1, 2)    # Multiple arguments give us a tuple.
('Isaac', 'Newton')





###Advanced Sorting 
from operator import itemgetter, attrgetter

student_tuples = [
        ('john', 'A', 15),
        ('jane', 'B', 12),
        ('dave', 'B', 10),
    ]
>>> sorted(student_tuples, key=lambda student: student[2])   # sort by age
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

#OR 
>>> sorted(student_tuples, key=itemgetter(2))
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

#for object 

>>> class Student:
            def __init__(self, name, grade, age):
                self.name = name
                self.grade = grade
                self.age = age
            def __repr__(self):
                return repr((self.name, self.grade, self.age))



>>> student_objects = [
        Student('john', 'A', 15),
        Student('jane', 'B', 12),
        Student('dave', 'B', 10),
    ]
>>> sorted(student_objects, key=lambda student: student.age)   # sort by age
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

#OR 

>>> sorted(student_objects, key=attrgetter('age'))
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]


#for multiple key , use below 

>>> sorted(student_tuples, key=itemgetter(1,2))
[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]



>>> sorted(student_objects, key=attrgetter('grade', 'age'))
[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]


#The operator.methodcaller() function makes method calls 
#with fixed parameters for each object being sorted. 

>>> from operator import methodcaller
>>> messages = ['critical!!!', 'hurry!', 'standby', 'immediate!!']
>>> sorted(messages, key=methodcaller('count', '!')) #calls str.count('!')
['standby', 'hurry!', 'immediate!!', 'critical!!!']











###Web Page & CGI Programming
#Web programming and automation using Python

#Common Gateway Interface 
#A CGI script is invoked by an HTTP server, eg to process <FORM> or <ISINDEX> element
#cgi script - print text to display in browser
#must be under ['/cgi-bin', '/htbin']

#file :  cgi-bin/cgiEx.py

import cgi
import cgitb
cgitb.enable(display=0, logdir="logs") #display detailed reports in the Web browser if any errors occur, display=0 means don't display, but write to log 
form = cgi.FieldStorage()


#headers section
print("Content-Type: text/html")    # HTML is following
print()                             # blank line, end of headers

#content section
print("<HTML>")
print("<TITLE>CGI script output</TITLE>")
print("<BODY>")
print("<H1>This is my first CGI script</H1>")
print("Hello, world!")
print(form.getfirst("name","").upper(), ",".join(form.getlist("addr")) )
print("</BODY></HTML>")

#run http server from outside of this script 
#python3: python -m http.server --bind 127.0.0.1 --cgi 8080
#py2.7: python -m CGIHTTPServer  8080
#in browser http://localhost:8080/cgi-bin/cgiEx.py
#or http://localhost:8080/cgi-bin/cgiEx.py?name=Joe+Blow&addr=At+Home&addr=At+Office



##for  an uploaded file field 

fileitem = form["userfile"]
if fileitem.file:
    # It's an uploaded file; count lines
    linecount = 0
    while True:
        line = fileitem.file.readline()
        if not line: break
        linecount = linecount + 1

#Other important methods
cgi.print_environ()
Format the shell environment in HTML.

cgi.print_form(form)  #form is dict of { key:value} which will be formatted
Format a form in HTML.

cgi.print_directory()
Format the current directory in HTML.

cgi.print_environ_usage()
Print a list of useful (used by CGI) environment variables in HTML.

cgi.escape(s, quote=False)
Convert the characters '&', '<' and '>'  and quotes (if quote=True) in string s to HTML-safe sequences
Deprecated, Must use quote=True 
or  Use html.escape(s, quote=True) for escaping and html.unescape(s) for unescaping 

cgi.test()
Robust test CGI script, usable as main program. Writes minimal HTTP headers and formats all information provided to the script in HTML form


cgi.parse_qs(qs, keep_blank_values=False, strict_parsing=False)
Use urllib.parse.parse_qs(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace')
Parse a query string given as a string argument (data of type application/x-www-form-urlencoded). 
Data are returned as a dictionary

cgi.parse_qsl(qs, keep_blank_values=False, strict_parsing=False)
Use urllib.parse.parse_qsl(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace')
Parse a query string given as a string argument (data of type application/x-www-form-urlencoded). Data are returned as a list of name, value pairs

		
		
###Web Server Gateway Interface (WSGI)
#Used for writing a server, or a py file used by  web server eg http
#Ref implementation is wsgiref module
#details of protocol between webserver and py file is  http://www.wsgi.org




###Database Access

#Module-sqllit , standard package
from sqlite3 import connect

conn = connect(r'D:/temp.db')
curs = conn.cursor()
curs.execute('create table if not exists emp (who, job, pay)')

prefix = 'insert into emp values '
curs.execute(prefix + "('Bob', 'dev', 100)")
curs.execute(prefix + "('Sue', 'dev', 120)")
conn.commit()
curs.execute("select * from emp where pay > 100")
for (who, job, pay) in curs.fetchall():
		print(who, job, pay)

result = curs.execute("select who, pay from emp")
result.fetchone()

query = "select * from emp where job = ?"
curs.execute(query, ('dev',)).fetchall()
conn.close()

#bulk insert 
import sqlite3
conn = connect(r'D:/temp.db')
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE stocks
             (date text, trans text, symbol text, qty real, price real)''')

# Insert a row of data
c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")



# Larger example that inserts many records at a time
purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
             ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
             ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
            ]
curs.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)

conn.commit()
conn.close()
#Other reference - http://pythoncentral.io/advanced-sqlite-usage-in-python/




$ sqlite3 sample.db
SQLite version 3.19.1 2017-05-24 13:08:33
Enter ".help" for usage hints.
sqlite> .tables
emp
sqlite> .schema emp
sqlite> select * from emp;






###SMTP & Emailing


#Mail handling using smtplib and poplib
#to use google access, make it less secure https://www.google.com/settings/security/lesssecureapps

#Sending a mail
1. Create a message using email.mime.text.MIMEText
2. Create smptp object via SMTP and sendmail()

#example:
import smtplib
import email.utils
from email.mime.text import MIMEText
import getpass

# Create the message
msg = MIMEText('This is the body of the message.')
msg['To'] = email.utils.formataddr(('Recipient', 'ndas1971@gmail.com'))
msg['From'] = email.utils.formataddr(('Author', 'ndas1971@gmail.com'))
msg['Subject'] = 'Simple test message'


server = smtplib.SMTP('smtp.gmail.com', 587)  
server.set_debuglevel(True) # show communication with the server
try:
    # identify ourselves, prompting server for supported features
	server.ehlo()
	# If we can encrypt this session, do it
	if server.has_extn('STARTTLS'):
		server.starttls()
		server.ehlo() # re-identify ourselves over TLS connection
	
	p = getpass.getpass()
	server.login('ndas1971@gmail.com', p)
	server.sendmail('ndas1971@gmail.com', ['ndas1971@gmail.com',], msg.as_string())
finally:
    server.quit()

#For multiple recipants
recipients = ['john.doe@example.com', 'john.smith@example.co.uk']
msg['To'] = ", ".join(recipients)
s.sendmail(sender, recipients, msg.as_string())
#check many examples
#https://docs.python.org/3/library/email-examples.html

##Download message using poplib 
#server mmust support POP3



import getpass, poplib
import email
from io import StringIO   # for py2.x cStringIO
from email.generator import Generator


user = 'ndas1971@gmail.com' 

Mailbox = poplib.POP3_SSL('pop.gmail.com', '995') 
Mailbox.user(user) 

p = getpass.getpass()

Mailbox.pass_(p) 
numMessages = len(Mailbox.list()[1])

#(numMsgs, totalSize) = Mailbox.stat()
print(" No of messages=%s" % numMessages )
for i in range(5):
	msg = Mailbox.retr(i+1)[1]
	f_msg = email.message_from_bytes(b"\n".join(msg))   #in py2.x, use _string 
	for header in [ 'subject', 'to', 'from' ]:
		print('%-8s: %s' % (header.upper(), f_msg[header]))
	print("\n")

# complete message

num = int(input("Input msg # for seeing complete message>"))
msg = Mailbox.retr(num+1)[1]
c_msg = email.message_from_bytes(b"\n".join(msg))  #in py2.x, use _string 
fp = StringIO()
g = Generator(fp, mangle_from_=False, maxheaderlen=60)
g.flatten(c_msg)
text = fp.getvalue()
print(text)

Mailbox.quit()





###CSV handling - std module 

#csv.reader(csvfile, dialect='excel', **fmtparams)
#Each row read from the csv file is returned as a list of strings. 
#No automatic data type conversion is performed unless the QUOTE_NONNUMERIC format option is specified 
#(in which case unquoted fields are transformed into floats).

csvreader.next()
csvreader.line_num

#newline='' , pass newline directly without translation to reader/write process 

import csv
with open('eggs.csv', newline='',) as csvfile:
    r = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
    for row in r:
        print(':'.join(row))
Spam, Spam, Spam, Spam, Spam, Baked Beans
Spam, Lovely Spam, Wonderful Spam

#csv.writer(csvfile, dialect='excel', **fmtparams)
#csvwriter.writerow(row)
#csvwriter.writerows(rows)
#DictWriter.writeheader()


import csv
with open('eggs1.csv', 'w', newline='') as csvfile:
    w = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    w.writerow(['Spam'] * 5 + ['Baked Beans'])
    w.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

#For handling header etc - use manually 

import csv
dict1 = {}

with open("test.csv", "rb") as infile:
    reader = csv.reader(infile)
    headers = next(reader)[1:]
    for row in reader:
        dict1[row[0]] = {key: int(value) for key, value in zip(headers, row[1:])}
        
#data 
,col1,col2,col3
row1,23,42,77
row2,25,39,87
row3,48,67,53
row4,14,48,66





###xml.etree.ElementTree — The ElementTree XML API (same in Py3.x and Py2.7)
#std module 
 
#Example: country_data.xml file 
  
<?xml version="1.0"?>
<data>
    <country name="Liechtenstein">
        <rank>1</rank>
        <year>2008</year>
        <gdppc>141100</gdppc>
        <neighbor name="Austria" direction="E"/>
        <neighbor name="Switzerland" direction="W"/>
    </country>
    <country name="Singapore">
        <rank>4</rank>
        <year>2011</year>
        <gdppc>59900</gdppc>
        <neighbor name="Malaysia" direction="N"/>
    </country>
    <country name="Panama">
        <rank>68</rank>
        <year>2011</year>
        <gdppc>13600</gdppc>
        <neighbor name="Costa Rica" direction="W"/>
        <neighbor name="Colombia" direction="E"/>
    </country>
</data>


#code 
import xml.etree as etree


import xml.etree.ElementTree as ET
tree = ET.parse('example.xml')
root = tree.getroot()

print(ET.tostring(root))  #give element 

#Or directly from a string:
root = ET.fromstring(country_data_as_string)

#Every element has a tag and a dictionary of attributes

>>> root.tag
'data'
>>> root.attrib
{}


#It also has children nodes over which we can iterate

for child in root:
    print(child.tag, child.attrib)
...
country {'name': 'Liechtenstein'}
country {'name': 'Singapore'}
country {'name': 'Panama'}


#Children are nested, 
#access specific child nodes by index:

>>> root[0][1].text

#Finding interesting elements

#use Element.iter()
#any tag can be given, then it would return list of those 
for neighbor in root.iter('neighbor'):
		print(neighbor.attrib)

#Use Element.findall() finds only elements with a tag which are direct children of the current element. 
#Element.find() finds the first child with a particular tag, 
#Element.text accesses the element’s text content. 
#Element.get() accesses the element’s attributes


for country in root.findall('country'):
		rank = country.find('rank').text
		name = country.get('name')
		print(name, rank)

##XPath support - limited support via findall() and find()
#findall always return list of ELement 
#find always return single Element 

import xml.etree.ElementTree as ET

# Top-level elements
root.findall(".")

# All 'neighbor' grand-children of 'country' children of the top-level
# elements
root.findall("./country/neighbor")

# Nodes with name='Singapore' that have a 'year' child
root.findall(".//year/..[@name='Singapore']")
ET.tostring(x[0])

# 'year' nodes that are children of nodes with name='Singapore'
root.findall(".//*[@name='Singapore']/year")

# All 'neighbor' nodes that are the second child of their parent
root.findall(".//neighbor[2]")

#Modifying an XML File
#to update attribute,  use Element.set()
#to update text, just assign to text 

for rank in root.iter('rank'):
		new_rank = int(rank.text) + 1
		rank.text = str(new_rank)
		rank.set('updated', 'yes')

tree.write('output.xml')


#remove elements using Element.remove(). 

>>> for country in root.findall('country'):
		rank = int(country.find('rank').text)
		if rank > 50:
			root.remove(country)

>>> tree.write('output.xml')

#Building XML documents

>>> a = ET.Element('a')
>>> b = ET.SubElement(a, 'b')
>>> c = ET.SubElement(a, 'c')
>>> d = ET.SubElement(c, 'd')
>>> ET.dump(a)
<a><b /><c><d /></c></a>

##To add new element, use Element.append()
def SubElementWithText(parent, tag, text):
    attrib = {}
    element = parent.makeelement(tag, attrib)
    parent.append(element)
    element.text = text
    return element

#Usage 
import xml.etree.ElementTree as ET

tree = ET.parse('test.xml')
root = tree.getroot()

a = root.find('a')
b = ET.SubElement(a, 'b')
c = SubElementWithText(b, 'c', 'text3')
print(ET.tostring(root))

#Parsing XML with Namespaces
#If the XML input has namespaces, 
#tags and attributes with prefixes in the form prefix:sometag 
#get expanded to {uri}sometag where the prefix is replaced by the full URI.
#Also, if there is a default namespace, that full URI gets prepended to all of the non-prefixed tags

<?xml version="1.0"?>
<actors xmlns:fictional="http://characters.example.com"
        xmlns="http://people.example.com">
    <actor>
        <name>John Cleese</name>
        <fictional:character>Lancelot</fictional:character>
        <fictional:character>Archie Leach</fictional:character>
    </actor>
    <actor>
        <name>Eric Idle</name>
        <fictional:character>Sir Robin</fictional:character>
        <fictional:character>Gunther</fictional:character>
        <fictional:character>Commander Clement</fictional:character>
    </actor>
</actors>


#Option-1 

root = fromstring(xml_text)
for actor in root.findall('{http://people.example.com}actor'):
    name = actor.find('{http://people.example.com}name')
    print(name.text)
    for char in actor.findall('{http://characters.example.com}character'):
        print(' |-->', char.text)


#Option-2

ns = {'real_person': 'http://people.example.com',
      'role': 'http://characters.example.com'}

for actor in root.findall('real_person:actor', ns):
    name = actor.find('real_person:name', ns)
    print(name.text)
    for char in actor.findall('role:character', ns):
        print(' |-->', char.text)


		


###HTML Handling 
#Below methods has method is either "xml", "html" or "text" (default is "xml"). 
xml.etree.ElementTree.tostring(element, encoding="us-ascii", method="xml")
xml.etree.ElementTree.tostringlist(element, encoding="us-ascii", method="xml")
class xml.etree.ElementTree.ElementTree(element=None, file=None)
    write(file, encoding="us-ascii", xml_declaration=None, default_namespace=None, method="xml")


#html file 
<html>
    <head>
        <title>Example page</title>
    </head>
    <body>
        <p>Moved to <a href="http://example.org/">example.org</a>
        or <a href="http://example.com/">example.com</a>.</p>
    </body>
</html>

#example 

from xml.etree.ElementTree import ElementTree
tree = ElementTree()
tree.parse("index.xhtml")
#<Element 'html' at 0xb77e6fac>
p = tree.find("body/p")     # Finds first occurrence of tag p in body
links = list(p.iter("a"))   # Returns list of all links
links
#[<Element 'a' at 0xb77ec2ac>, <Element 'a' at 0xb77ec1cc>]
for i in links:             # Iterates through all found links
    i.attrib["target"] = "blank"

tree.write("output.xhtml")





###HTTP automation 
#Automating HTTP using urllib, urllib2 and httplib
#Py2.x - urllib and urllib2 , renamed in Python 3 to urllib.request, urllib.parse, and urllib.error. 
#Py3.x urllib.request.urlopen() is equivalent to urllib2.urlopen() and urllib.urlopen() has been removed
#Use apache requests for high level API

•urllib.request for opening and reading URLs for file, ftp and http, raises urllib.error
•urllib.parse for parsing URLs
•urllib.robotparser for parsing robots.txt files

#urllib.request.urlopen(url, data=None, [timeout, ]*, cafile=None, capath=None, cadefault=False, context=None)
provide data (must be in bytes) for http POST, use urllib.parse.urlencode({k:v,..})

#example - read() gives in bytes

import urllib.request
with urllib.request.urlopen('http://www.python.org/') as f:
	print(f.read(100).decode('utf-8'))  #can use read() and readline() or can be used as iterator

#Py2.7 
import urllib2
response = urllib2.urlopen('http://python.org/')
html = response.read()
#OR
import urllib2
req = urllib2.Request('http://www.voidspace.org.uk')
response = urllib2.urlopen(req)
the_page = response.read()


#get method
import urllib.request
import urllib.parse
params = urllib.parse.urlencode({'spam': 1, 'eggs': 2, 'bacon': 0})
url = "http://www.musi-cal.com/cgi-bin/query?%s" % params
with urllib.request.urlopen(url) as f:
	print(f.read().decode('utf-8'))

#Py2.7
import urllib2
import urllib
data = {}
data['name'] = 'Somebody Here'
data['location'] = 'Northampton'
data['language'] = 'Python'
url_values = urllib.urlencode(data)
print url_values  # The order may differ. 
#name=Somebody+Here&language=Python&location=Northampton
url = 'http://www.example.com/example.cgi'
full_url = url + '?' + url_values
data = urllib2.urlopen(full_url)
the_page = data.read()





# POST method 
import urllib.request
import urllib.parse
data = urllib.parse.urlencode({'spam': 1, 'eggs': 2, 'bacon': 0})
data = data.encode('ascii')
with urllib.request.urlopen("http://requestb.in/xrbl82xr", data) as f:
	print(f.read().decode('utf-8'))


#py2.7 
import urllib
import urllib2

url = 'http://www.someserver.com/cgi-bin/register.cgi'
values = {'name' : 'Michael Foord',
          'location' : 'Northampton',
          'language' : 'Python' }

data = urllib.urlencode(values)
req = urllib2.Request(url, data)
response = urllib2.urlopen(req)
the_page = response.read()

#parse method
from urllib.parse import urlparse
o = urlparse('http://www.cwi.nl:80/%7Eguido/Python.html')
>> o
ParseResult(scheme='http', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html',
            params='', query='', fragment='')
o.scheme
'http'
o.port
80
o.geturl()
'http://www.cwi.nl:80/%7Eguido/Python.html'

#Other methods 
urllib.parse.quote(string, safe='/', encoding=None, errors=None)
Replace special characters in string using the %xx escape.
Example: quote('/El Niño/') yields '/El%20Ni%C3%B1o/'.

urllib.parse.quote_plus(string, safe='', encoding=None, errors=None)
Like quote(), but also replace spaces by plus signs
as required for quoting HTML form values when building up a query string to go into a URL
Example: quote_plus('/El Niño/') yields '%2FEl+Ni%C3%B1o%2F'.

urllib.parse.unquote(string, encoding='utf-8', errors='replace')
Replace %xx escapes by their single-character equivalent

urllib.parse.unquote_plus(string, encoding='utf-8', errors='replace')
Like unquote(), but also replace plus signs by spaces


#Py2.x httplib module has been renamed to http.client in Python 3
#don't use directly, urllib.request uses httplib to handle URLs that use HTTP and HTTPS
#use 'pip3 install requests' and use requests package in real code


###*Apache requests 
$ pip install requests 

#basic usgae 

p1 = 'http://proxy_username:proxy_password@proxy_server.com:port'
p2 = 'https://proxy_username:proxy_password@proxy_server.com:port'
proxy = {'http': p1, 'https':p2}
r = requests.get(site, proxies=proxy, auth=('site_username', 'site_password'))

#example
import requests
r = requests.get("http://www.yahoo.com")
r.text
r.status_code   # status code
r.headers  		# dict object

##RESTful API
r = requests.post(site)
r = requests.put("site/put")
r = requests.delete("site/delete")
r = requests.head("site/get")
r = requests.options("site/get")


##Get

payload1 = {'key1': 'value1', 'key2': 'value2'}
r = requests.get("http://httpbin.org/get", params=payload1)
print(r.url)  #http://httpbin.org/get?key2=value2&key1=value1
r.headers
r.text
r.json()  # it's a python dict

#For Request debugging,
>>> r.request.url
'http://httpbin.org/forms/post?delivery=12&topping=onion&custtel=123&comments=ok&custname=das&custemail=ok%40com&size=small'
>>> r.request.headers
{'Content-Length': '0', 'User-Agent': 'Mozilla/5.0', 'Connection': 'keep-alive', 'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate'}
>>> r.request.body


##POST
headers = {'User-Agent': 'Mozilla/5.0'}
payload = {'custname':'das', 'custtel': '123', 'custemail' : 'ok@com', 'size':'small',  'topping':'bacon',  'topping': 'onion',  'delivery':'12', 'comments': 'ok'}
r = requests.post("http://httpbin.org/post", data=payload, headers=headers)
r.text
r.headers
r.json()

r.request.headers
r.request.body #custname=das&custtel=123&custemail=ok@com&size=small&topping=bacon&topping=onion&delivery=12&comments=ok

##Content
r.text
r.content  # as bytes
r.json()  # json content

##Example to handle image Py3
#Install Pillow from http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow , for py2, http://www.pythonware.com/products/pil/

from PIL import Image
from io import StringIO   
i = Image.open(StringIO(r.content))




##Custom Headers

import json
payload = {'some': 'data'}
headers = {'content-type': 'application/json'}
r = requests.post(url, data=json.dumps(payload), headers=headers)

##POST a Multipart-Encoded File

files = {'file': open('report.xls', 'rb')}
r = requests.post(url, files=files)

##Cookies
#get
r.cookies['example_cookie_name']

#or sending
cookies = dict(cookies_are='working')
r = requests.get(url, cookies=cookies)

##Or persisting across session
s = requests.Session()

s.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
r = s.get("http://httpbin.org/cookies")

r.text # contains cookies from first access

##Example:
import requests
headers = {'User-Agent': 'Mozilla/5.0'}
payload = {'username':'niceusername','pass':'123456'}

session = requests.Session()
session.post('https://admin.example.com/login.php',headers=headers,data=payload)
# the session instance holds the cookie. So use it to get/post later.
# e.g. session.get('https://example.com/profile')

#Authentication with form
response = requests.get(url, auth = ('username', 'password')) 


##Example - Form conatins

<textarea id="text" class="wikitext" name="text" cols="80" rows="20">
This is where your edited text will go
</textarea>
<input type="submit" id="save" name="save" value="Submit changes">

#Code:

import requests
from bs4 import BeautifulSoup

url = "http://www.someurl.com"

username = "your_username"
password = "your_password"

session = requests.Session()
session.auth = (username, password)
response = session.get(url, verify=False)

# Getting the text of the page from the response data       
page = BeautifulSoup(response.text)

# Finding the text contained in a specific element, for instance, the 
# textarea element that contains the area where you would write a forum post
txt = page.find('textarea', id="text").string

# Finding the value of a specific attribute with name = "version" and 
# extracting the contents of the value attribute
tag = page.find('input', attrs = {'name':'version'})
ver = tag['value']

# Changing the text to whatever you want
txt = "Your text here, this will be what is written to the textarea for the post"

# construct the POST request
form_data = {
    'save' : 'Submit changes'
    'text' : txt
} 

post = session.post(url,data=form_data,verify=False)











###XML Processing - Use BeautifulSoup for HTML/XML processing
#no xpath, but css select 
$ pip install BeautifulSoup4
$ pip install requests


from bs4 import BeautifulSoup
import requests

r  = requests.get("http://www.yahoo.com")
data = r.text

soup = BeautifulSoup(data, "html.parser")
print(soup.prettify())
#extracting all the text from a page:
print(soup.get_text())

#finding all 
#Signature: find_all(name, attrs, recursive, text, limit, **kwargs)
#name can be a string(inc tag), a regular expression, a list, a function, or the value True.

for link in soup.find_all('a'):  # tag <a href=".."
		print(link.get('href'))	 # Attribute href


##Parser 
#lxml 
#from http://www.lfd.uci.edu/~gohlke/pythonlibs/#lxml

#html5lib 
$ pip install html5lib

BeautifulSoup(markup, "html.parser") #default
BeautifulSoup(markup, "lxml")  #very fast 
BeautifulSoup(markup, "lxml-xml")  #only xml parser
BeautifulSoup(markup, "xml")
BeautifulSoup(markup, "html5lib") #creates valid HTML5, •Extremely lenient, very slow


    
##	tag becomes attributes of soup object
soup = BeautifulSoup('<html><body><p class="title">data</p></body></html>', 'html.parser')
soup.html
soup.html.body.p
soup.html.body.text #or .string
soup.html.body.attrs
soup.html.body.name
soup.html.body.p['class']
soup.body
soup.body.attrs
soup.p.text  		# can call nested .p directly as well 
soup.get_text()  
soup.html.name
soup.p.parent.name

##Tag 
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag = soup.b
type(tag)
# <class 'bs4.element.Tag'>
tag.name
# u'b'
#modify 
tag.name = "blockquote"
tag
# <blockquote class="boldest">Extremely bold</blockquote>

##Attributes
tag['id']
# u'boldest'
tag.attrs
# {u'id': 'boldest'}
#add, remove, and modify a tag’s attributes
tag['id'] = 'verybold'
tag['another-attribute'] = 1
tag
# <b another-attribute="1" id="verybold"></b>

del tag['id']
del tag['another-attribute']
tag
# <b></b>

tag['id']
# KeyError: 'id'
print(tag.get('id'))
# None

##multivalued attribute
# Beautiful Soup presents the value(s) of a multi-valued attribute as a list:

css_soup = BeautifulSoup('<p class="body strikeout"></p>')
css_soup.p['class']
# ["body", "strikeout"]

css_soup = BeautifulSoup('<p class="body"></p>')
css_soup.p['class']
# ["body"]

#for non multivalued, does not convert to list 
id_soup = BeautifulSoup('<p id="my id"></p>')
id_soup.p['id']
# 'my id'


#turn a tag back into a string, multiple attribute values are consolidated:
rel_soup = BeautifulSoup('<p>Back to the <a rel="index">homepage</a></p>')
rel_soup.a['rel']
# ['index']
rel_soup.a['rel'] = ['index', 'contents']
print(rel_soup.p)
# <p>Back to the <a rel="index contents">homepage</a></p>

#use `get_attribute_list to get a value that’s always a list
id_soup.p.get_attribute_list(‘id’) # [“my id”]

#for XML, there are no multi-valued attributes:
xml_soup = BeautifulSoup('<p class="body strikeout"></p>', 'xml')
xml_soup.p['class']
# u'body strikeout'

##A string corresponds to a bit of text within a tag
#NavigableString supports most of the features of Navigating the tree and Searching the tree
tag.string
# u'Extremely bold'
type(tag.string)
# <class 'bs4.element.NavigableString'>



##	Pretty-printing
print(soup.prettify())

##Non-pretty printing
print(str(soup))

##Navigating using tag names - other than tags become soup's attributes 
.contents and .children
#A tag’s children are available in a LIST called .contents
#The .contents and .children attributes only consider a tag’s direct children
#Instead of getting as a list, iterate over a tag’s children using the .children generator:
for child in title_tag.children:
    print(child)

    
.descendants
#The .descendants attribute iterates over all of a tag’s children, recursively: 
#its direct children, the children of its direct children, and so on
for child in head_tag.descendants:
    print(child)

.string
#If a tag has only one NavigableString as child
#or tag has one child tag who has another the NavigableString child 
#that is  .string, else it is NOne 

.strings and stripped_strings
#If there’s more than one thing inside a tag, 
# Use the .strings generator or .stripped_strings generator(whitespace removed)
for string in soup.strings:
    print(repr(string))



.parent
#an element’s parent
title_tag = soup.title
title_tag
# <title>The Dormouse's story</title>
title_tag.parent
# <head><title>The Dormouse's story</title></head>
#The title string itself has a parent: the <title> tag that contains it:
title_tag.string.parent
# <title>The Dormouse's story</title>
html_tag = soup.html
type(html_tag.parent)
# <class 'bs4.BeautifulSoup'>
print(soup.parent)
# None


.parents
#iterate over all of an element’s parents 


.next_sibling and .previous_sibling
#navigate to one sibling (elements that are on the same level of the parse tree)
#Example 
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>
#Note  .next_sibling of the first <a> tag is not second <a> tag. 
#But actually, it’s a string: the comma and newline that separate the first <a> tag from the second:
link = soup.a
link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
link.next_sibling
# u',\n'

#The second <a> tag is actually the .next_sibling of the comma:
link.next_sibling.next_sibling
# <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>



.next_siblings and .previous_siblings
#iterate all siblings


.next_element and .previous_element
#next or previous element , might not be equivalent to siblings 

.next_elements and .previous_elements
#iterate all elements

##	Searching the string
find_all(tag_name, attrs_value, recursive, text, limit, **kwargs)  #returns list 
find(name, attrs, recursive, string, **kwargs) #returns first element
#The find_all() method looks through a tag’s descendants and retrieves all descendants in a list 
#name can be a string, a regular expression, a list, a function, or the value True.

#string 
soup.find_all('p')
#RE
import re
for tag in soup.find_all(re.compile("^b")):
		print(tag.name)

#List of strings to match 
soup.find_all(["a", "b"])

#True
#True matches everything it can
for tag in soup.find_all(True):
    print(tag.name)


#function
#define a function that takes an element as its only argument. 
#The function should return True if the argument matches, and False otherwise.
def has_class_but_no_id(tag):
		return tag.has_attr('class') and not tag.has_attr('id')

soup.find_all(has_class_but_no_id)

#Other forms of find_all 
#find_all(tag_name, attrs_value, recursive, text, limit, **kwargs) 
soup.find_all("title")
# [<title>The Dormouse's story</title>]

soup.find_all("p", "title")
# [<p class="title"><b>The Dormouse's story</b></p>]

soup.find_all("a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

#Any argument that’s not recognized will uses as a filter on one of a tag’s attributes
#filter an attribute based on a string, a regular expression, a list, a function, or the value True.
soup.find_all(id="link2")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
soup.find_all(href=re.compile("elsie"))
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]
soup.find_all(href=re.compile("elsie"), id='link1')
# [<a class="sister" href="http://example.com/elsie" id="link1">three</a>]

#Some attributes, like the data-* attributes in HTML 5, can not be used 
data_soup = BeautifulSoup('<div data-foo="value">foo!</div>')
data_soup.find_all(data-foo="value")
# SyntaxError: keyword can't be an expression
#use as 
data_soup.find_all(attrs={"data-foo": "value"})
# [<div data-foo="value">foo!</div>]

#special handling of class attribute(use class_)
#you can pass class_ a string, a regular expression, a function, or True:
soup.find_all("a", class_="sister")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find_all(class_=re.compile("itl"))
# [<p class="title"><b>The Dormouse's story</b></p>]

def has_six_characters(css_class):
    return css_class is not None and len(css_class) == 6

soup.find_all(class_=has_six_characters)
#Maching any of class value 
css_soup = BeautifulSoup('<p class="body strikeout"></p>')
css_soup.find_all("p", class_="strikeout")
# [<p class="body strikeout"></p>]
css_soup.find_all("p", class_="body")
# [<p class="body strikeout"></p>]

#or full value in exact order 
css_soup.find_all("p", class_="body strikeout")
# [<p class="body strikeout"></p>]
#not in other order
css_soup.find_all("p", class_="strikeout body")
# []

#or use css selector where order does not matter 
css_soup.select("p.strikeout.body")
# [<p class="body strikeout"></p>]
#or use attrs 
soup.find_all("a", attrs={"class": "sister"})



#With string , search for strings instead of tags. 
# you can pass in a string, a regular expression, a list, a function, or the value True
soup.find_all(string="Elsie")
# [u'Elsie']

soup.find_all(string=["Tillie", "Elsie", "Lacie"])
# [u'Elsie', u'Lacie', u'Tillie']

soup.find_all(string=re.compile("Dormouse"))
[u"The Dormouse's story", u"The Dormouse's story"]

def is_the_only_string_within_a_tag(s):
    """Return True if this string is the only child of its parent tag."""
    return (s == s.parent.string)

soup.find_all(string=is_the_only_string_within_a_tag)
# [u"The Dormouse's story", u"The Dormouse's story", u'Elsie', u'Lacie', u'Tillie',
#with other args of find_all 
soup.find_all("a", string="Elsie")
# [<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]

#The string argument is new in Beautiful Soup 4.4.0. 
#In earlier versions it was called text:
soup.find_all("a", text="Elsie")
# [<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]


#Limit the number of search returned 
soup.find_all("a", limit=2)
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

#By default, all descendants are searched 
#to consider direct children, pass in recursive=False
soup.html.find_all("title")
# [<title>The Dormouse's story</title>]
soup.html.find_all("title", recursive=False)
# []

##Calling a tag is like calling find_all()
#These two lines are also equivalent:
soup.find_all("a")
soup("a")

soup.title.find_all(string=True)
soup.title(string=True)



#	Other find methods
find_parents(name, attrs, text, limit, **kwargs)
find_parent(name, attrs, text, **kwargs)

find_next_siblings(name, attrs, text, limit, **kwargs)
find_next_sibling(name, attrs, text, **kwargs)

find_previous_siblings(name, attrs, text, limit, **kwargs)
find_previous_sibling(name, attrs, text, **kwargs)

find_all_next(name, attrs, text, limit, **kwargs)
find_next(name, attrs, text, **kwargs)

find_all_previous(name, attrs, text, limit, **kwargs)
find_previous(name, attrs, text, **kwargs)



##CSS Selector (supports a subset of CSS3)
#find tags:
soup.select("title")
# [<title>The Dormouse's story</title>]
soup.select("p:nth-of-type(3)")
# [<p class="story">...</p>]


#Find tags beneath other tags:
soup.select("body a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
soup.select("html head title")
# [<title>The Dormouse's story</title>]

#Find tags directly beneath other tags:
soup.select("head > title")
# [<title>The Dormouse's story</title>]

soup.select("p > a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("p > a:nth-of-type(2)")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

soup.select("p > #link1")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select("body > a")
# []

#Find the siblings of tags:
soup.select("#link1 ~ .sister")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie"  id="link3">Tillie</a>]

soup.select("#link1 + .sister")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]


#Find tags by CSS class:
soup.select(".sister")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("[class~=sister]")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]


#Find tags by ID:
soup.select("#link1")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select("a#link2")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

#Find tags that match any selector from a list of selectors:
soup.select("#link1,#link2") # [<a class=”sister” href=”http://example.com/elsie” id=”link1”>Elsie</a>, # <a class=”sister” href=”http://example.com/lacie” id=”link2”>Lacie</a>]

#Test for the existence of an attribute:
soup.select('a[href]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

#Find tags by attribute value:
soup.select('a[href="http://example.com/elsie"]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select('a[href^="http://example.com/"]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select('a[href$="tillie"]')
# [<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select('a[href*=".com/el"]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]


#Match language codes:
multilingual_soup.select('p[lang|=en]')
# [<p lang="en">Hello</p>,
#  <p lang="en-us">Howdy, y'all</p>,
#  <p lang="en-gb">Pip-pip, old fruit</p>]


#Find only the first tag that matches a selector:
soup.select_one(".sister")
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

##Changing tag names and attributes
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag = soup.b

tag.name = "blockquote"
tag['class'] = 'verybold'
tag['id'] = 1
tag
# <blockquote class="verybold" id="1">Extremely bold</blockquote>

del tag['class']
del tag['id']
tag
# <blockquote>Extremely bold</blockquote>

##Modifying .string
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)

tag = soup.a
tag.string = "New link text."
tag
# <a href="http://example.com/">New link text.</a>

##append()
#append to a tag’s contents
soup = BeautifulSoup("<a>Foo</a>")
soup.a.append("Bar")

soup
# <html><head></head><body><a>FooBar</a></body></html>
soup.a.contents
# [u'Foo', u'Bar']



#to add a string to a document
soup = BeautifulSoup("<b></b>")
tag = soup.b
tag.append("Hello")
new_string = NavigableString(" there")
tag.append(new_string)
tag
# <b>Hello there.</b>
tag.contents
# [u'Hello', u' there']


#to create a comment or some other subclass of NavigableString
from bs4 import Comment
new_comment = Comment("Nice to see you.")
tag.append(new_comment)
tag
# <b>Hello there<!--Nice to see you.--></b>
tag.contents
# [u'Hello', u' there', u'Nice to see you.']


#to create a whole new tag
soup = BeautifulSoup("<b></b>")
original_tag = soup.b

new_tag = soup.new_tag("a", href="http://www.example.com")
original_tag.append(new_tag)
original_tag
# <b><a href="http://www.example.com"></a></b>

new_tag.string = "Link text."
original_tag
# <b><a href="http://www.example.com">Link text.</a></b>

##insert()
#insert at whatever numeric position you say.
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
tag = soup.a

tag.insert(1, "but did not endorse ")
tag
# <a href="http://example.com/">I linked to but did not endorse <i>example.com</i></a>
tag.contents
# [u'I linked to ', u'but did not endorse', <i>example.com</i>]



##insert_before() and insert_after()
#inserts a tag or string immediately before/after something else in the parse tree:


soup = BeautifulSoup("<b>stop</b>")
tag = soup.new_tag("i")
tag.string = "Don't"
soup.b.string.insert_before(tag)
soup.b
# <b><i>Don't</i>stop</b>


soup.b.i.insert_after(soup.new_string(" ever "))
soup.b
# <b><i>Don't</i> ever stop</b>
soup.b.contents
# [<i>Don't</i>, u' ever ', u'stop']



##clear()
#removes the contents of a tag:
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
tag = soup.a

tag.clear()
tag
# <a href="http://example.com/"></a>


##extract()
#removes a tag or string from the tree
#returns the tag or string that was extracted:
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

i_tag = soup.i.extract()

a_tag
# <a href="http://example.com/">I linked to</a>

i_tag
# <i>example.com</i>

print(i_tag.parent)
None

#two parse trees
my_string = i_tag.string.extract()
my_string
# u'example.com'

print(my_string.parent)
# None
i_tag
# <i></i>



##decompose()
#removes a tag from the tree, then completely destroys it and its contents:


markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

soup.i.decompose()

a_tag
# <a href="http://example.com/">I linked to</a>



##replace_with()
# removes a tag or string from the tree, and replaces it with the tag or string of your choice:
#returns the tag or string that was replaced
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

new_tag = soup.new_tag("b")
new_tag.string = "example.net"
a_tag.i.replace_with(new_tag)

a_tag
# <a href="http://example.com/">I linked to <b>example.net</b></a>

##wrap()
#wraps an element in the tag you specify. 
#It returns the new wrapper:
soup = BeautifulSoup("<p>I wish I was bold.</p>")
soup.p.string.wrap(soup.new_tag("b"))
# <b>I wish I was bold.</b>

soup.p.wrap(soup.new_tag("div")
# <div><p><b>I wish I was bold.</b></p></div>


##unwrap()
# opposite of wrap(). 
#It replaces a tag with whatever’s inside that tag. 
#returns the tag that was replaced.
#It’s good for stripping out markup:
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

a_tag.i.unwrap()
a_tag
# <a href="http://example.com/">I linked to example.com</a>



##use Unicode, Dammit without using Beautiful Soup. 
#It’s useful to guess correct encoding of the text 

from bs4 import UnicodeDammit
dammit = UnicodeDammit("Sacr\xc3\xa9 bleu!")
print(dammit.unicode_markup)
# Sacré bleu!
dammit.original_encoding
# 'utf-8'


#Unicode, Dammit’s guesses will get a lot more accurate 
#if you install the chardet or cchardet Python libraries. 

#or can pass few our estimates 
dammit = UnicodeDammit("Sacr\xe9 bleu!", ["latin-1", "iso-8859-1"])
print(dammit.unicode_markup)
# Sacré bleu!
dammit.original_encoding
# 'latin-1'


#use Unicode, Dammit to convert Microsoft smart quotes to HTML or XML entities:

markup = b"<p>I just \x93love\x94 Microsoft Word\x92s smart quotes</p>"

UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="html").unicode_markup
# u'<p>I just &ldquo;love&rdquo; Microsoft Word&rsquo;s smart quotes</p>'

UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="xml").unicode_markup
# u'<p>I just &#x201C;love&#x201D; Microsoft Word&#x2019;s smart quotes</p>'


#to convert Microsoft smart quotes to ASCII quotes:
UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="ascii").unicode_markup
# u'<p>I just "love" Microsoft Word\'s smart quotes</p>'


Hopefully you’ll find this feature useful, but Beautiful Soup doesn’t use it. Beautiful Soup prefers the default behavior, which is to convert Microsoft smart quotes to Unicode characters along with everything else:


UnicodeDammit(markup, ["windows-1252"]).unicode_markup
# u'<p>I just \u201clove\u201d Microsoft Word\u2019s smart quotes</p>'


##Inconsistent encodings

#Sometimes a document is mostly in UTF-8, 
#but contains Windows-1252 characters such as  Microsoft smart quotes. 


snowmen = (u"\N{SNOWMAN}" * 3)
quote = (u"\N{LEFT DOUBLE QUOTATION MARK}I like snowmen!\N{RIGHT DOUBLE QUOTATION MARK}")
doc = snowmen.encode("utf8") + quote.encode("windows_1252")
#messy display 
print(doc)
# ????I like snowmen!?
print(doc.decode("windows-1252"))
# â˜ƒâ˜ƒâ˜ƒ“I like snowmen!”

#use UnicodeDammit.detwingle() 
new_doc = UnicodeDammit.detwingle(doc)
print(new_doc.decode("utf8"))
# ???“I like snowmen!”


##Copying Beautiful Soup objects

import copy
p_copy = copy.copy(soup.p)
print p_copy
# <p>I want <b>pizza</b> and more <b>pizza</b>!</p>

print soup.p == p_copy
# True

print soup.p is p_copy
# False

##Comparing objects for equality
#content checking 
markup = "<p>I want <b>pizza</b> and more <b>pizza</b>!</p>"
soup = BeautifulSoup(markup, 'html.parser')
first_b, second_b = soup.find_all('b')
print first_b == second_b
# True

print first_b.previous_element == second_b.previous_element
# False

##Parsing only part of a document - use SoupStrainer


from bs4 import SoupStrainer

only_a_tags = SoupStrainer("a")

only_tags_with_id_link2 = SoupStrainer(id="link2")

def is_short_string(string):
    return len(string) < 10

only_short_strings = SoupStrainer(string=is_short_string)


html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

print(BeautifulSoup(html_doc, "html.parser", parse_only=only_a_tags).prettify())
# <a class="sister" href="http://example.com/elsie" id="link1">
#  Elsie
# </a>
# <a class="sister" href="http://example.com/lacie" id="link2">
#  Lacie
# </a>
# <a class="sister" href="http://example.com/tillie" id="link3">
#  Tillie
# </a>

print(BeautifulSoup(html_doc, "html.parser", parse_only=only_tags_with_id_link2).prettify())
# <a class="sister" href="http://example.com/lacie" id="link2">
#  Lacie
# </a>

print(BeautifulSoup(html_doc, "html.parser", parse_only=only_short_strings).prettify())
# Elsie
# ,
# Lacie
# and
# Tillie
# ...
#


#You can also pass a SoupStrainer into any of the methods covered in Searching the tree. 
soup = BeautifulSoup(html_doc)
soup.find_all(only_short_strings)
# [u'\n\n', u'\n\n', u'Elsie', u',\n', u'Lacie', u' and\n', u'Tillie',
#  u'\n\n', u'...', u'\n']





###*Module Json 
#JSON (JavaScript Object Notation) specified by RFC 7159 

# to and from file 
json.dump(obj, fp, skipkeys=False, ensure_ascii=True, check_circular=True, 
   allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
#fp is opened as fp = open(filename, "w")
#indent is string which is used for indentation

json.load(fp, cls=None, object_hook=None, parse_float=None, parse_int=None, 
    parse_constant=None, object_pairs_hook=None, **kw)
#fp is opened as fp = open(filename, "r")
#parse_type, if specified, will be called with the string of every JSON 'type' to be decoded


#To and from string  
json.dumps(obj, skipkeys=False, ensure_ascii=True, check_circular=True, 
    allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)

json.loads(s, encoding=None, cls=None, object_hook=None, parse_float=None, 
    parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
    
#Jason syntax 
JSON data is written as - "name":value pairs, eg -  "firstName":"John"
name is in double quote
JSON values can be:
    A number (integer or floating point)
    A string (in double quotes)
    A Boolean (true or false)
    An array (in square brackets with , as separator )
    An object (in curly braces with "name":value)
    null


#The file type for JSON files is ".json"
#The MIME type for JSON text is "application/json"
#No comment is allowed  even with /* */, // or #
#Only one root element or object is allowed, no multiple root elements 

#conversion table
#Python			JSON
dict 			object 
list, tuple 	array 
str 			string 
int, float,     int,float
True 			true 
False 			false 
None 			null 

#Note, separators=(',', ': ') as default if indent is not None.
#The json module always produces str objects, not bytes objects
#Keys in key/value pairs of JSON are always of the type str. 
#When a dictionary is converted into JSON, all the keys of the dictionary are coerced to strings
#That is, loads(dumps(x)) != x if x has non-string keys.
 

#Example file: example.json  
[
 { "empId: 1, "details": 
                        {                       
                          "firstName": "John",
                          "lastName": "Smith",
                          "isAlive": true,
                          "age": 25,
                          "salary": 123.5,
                          "address": {
                            "streetAddress": "21 2nd Street",
                            "city": "New York",
                            "state": "NY",
                            "postalCode": "10021-3100"
                          },
                          "phoneNumbers": [
                            {
                              "type": "home",
                              "number": "212 555-1234"
                            },
                            {
                              "type": "office",
                              "number": "646 555-4567"
                            },
                            {
                              "type": "mobile",
                              "number": "123 456-7890"
                            }
                          ],
                          "children": [],
                          "spouse": null
                        }
  } , { "empId: 20, "details": 
                            {                       
                              "firstName": "Johns",
                              "lastName": "Smith",
                              "isAlive": true,
                              "age": 25,
                              "salary": 123.5,
                              "address": {
                                "streetAddress": "21 2nd Street",
                                "city": "New York",
                                "state": "NY",
                                "postalCode": "10021-3100"
                              },
                              "phoneNumbers": [
                                {
                                  "type": "home",
                                  "number": "212 555-1234"
                                },
                                {
                                  "type": "office",
                                  "number": "646 555-4567"
                                },
                                {
                                  "type": "mobile",
                                  "number": "123 456-7890"
                                }
                              ],
                              "children": [],
                              "spouse": null
                            }
    }
]

#Example reading file 
import json 
import pprint 
fp = open("data/example.json", "r")
obj = json.load(fp)
fp.close()
pprint.pprint(obj)  #check size 
[{'details': {'address': {'city': 'New York',
                          'postalCode': '10021-3100',
                          'state': 'NY',
                          'streetAddress': '21 2nd Street'},
              'age': 25,
              'children': [],
              'firstName': 'John',
              'isAlive': True,
              'lastName': 'Smith',
              'phoneNumbers': [{'number': '212 555-1234', 'type': 'home'},
                               {'number': '646 555-4567', 'type': 'office'},
                               {'number': '123 456-7890', 'type': 'mobile'}],
              'salary': 123.5,
              'spouse': None},
  'empId': 1},
 {'details': {'address': {'city': 'New York',
                          'postalCode': '10021-3100',
                          'state': 'NY',
                          'streetAddress': '21 2nd Street'},
              'age': 25,
              'children': [],
              'firstName': 'Johns',
              'isAlive': True,
              'lastName': 'Smith',
              'phoneNumbers': [{'number': '212 555-1234', 'type': 'home'},
                               {'number': '646 555-4567', 'type': 'office'},
                               {'number': '123 456-7890', 'type': 'mobile'}],
              'salary': 123.5,
              'spouse': None},
  'empId': 20}]

#manipulations 
len(obj)        #2
type(obj)       #<class 'list'>
type(obj[0])    #<class 'dict'>
with open("data/example1.json", "w") as fp1:
    json.dump(obj, fp1, indent='\t')
  
#Obj is array , all array manipulations can be used 
[emp['details']['address']['state']   for emp in obj if emp['empId'] > 10] 








###*Python Standard Library – datetime
import datetime
dir(datetime)
dir(datetime.date)
dir(datetime.time)
dir(datetime.datetime)
dir(datetime.timedelta)


##here timestamp is as returned by time.time()
classmethod date.today()
classmethod date.fromtimestamp(timestamp)
classmethod date.fromordinal(ordinal) # ordinal, where January 1 of year 1 has ordinal 1

classmethod datetime.today()
classmethod datetime.now(tz=None)
classmethod datetime.utcnow()


classmethod datetime.fromtimestamp(timestamp, tz=None)
classmethod datetime.utcfromtimestamp(timestamp)

#combining date and time 
classmethod datetime.combine(date, time)

instancemethod timedelta.total_seconds()
instancemethod date.weekday(), datetime.weekday()  #0 is Monday
instancemethod datetime.date()  #returns date 
instancemethod datetime.time()  #returns time


class datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
class datetime.date(year, month, day) #month, day 1 based, hr, min, sec are zero based
class datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]])
class datetime.time([hour[, minute[, second[, microsecond[, tzinfo]]]]])

# +, -, *(by int), /(by int) operations are valid between two timedelta
# timedelta can be added or subtraced to date or datetime , but not time

# difference of two dates or datetimes  is timedelata
# date, time, datetime, timedelta are comparable
# no arithmatic operations are supported for two time 

# date.timetuple() or datetime.timetuple() returns time.struct_time(as returned by time.localtime())
# date.ctime(), datetime.ctime, date or time or datetime.strftime(format) returns string
# classmethod datetime.strptime(date_string, format) converts string to datetime

#years before 1900 cannot be used with strftime()
date.strftime("%A %d. %B %Y")
datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M")
datetime.strftime("%A, %d. %B %Y %I:%M%p")
#format
%d 	    Day of the month as a zero-padded decimal number. 	01, 02, ..., 31 	 
%m 	    Month as a zero-padded decimal number. 	01, 02, ..., 12 	 
%y 	    Year without century as a zero-padded decimal number. 	00, 01, ..., 99 	 
%Y 	    Year with century as a decimal number. 	1970, 1988, 2001, 2013 	 
%H 	    Hour (24-hour clock) as a zero-padded decimal number. 	00, 01, ..., 23 	 
%M 	    Minute as a zero-padded decimal number. 	00, 01, ..., 59 	 
%S 	    Second as a zero-padded decimal number. 	00, 01, ..., 59 	(3)

%a 	    Weekday as locale’s abbreviated name. 	
            Sun, Mon, ..., Sat (en_US);
            So, Mo, ..., Sa (de_DE)
%A 	    Weekday as locale’s full name. 	
            Sunday, Monday, ..., Saturday (en_US);
            Sonntag, Montag, ..., Samstag (de_DE)
%b 	    Month as locale’s abbreviated name. 	
            Jan, Feb, ..., Dec (en_US);
            Jan, Feb, ..., Dez (de_DE)
%B 	    Month as locale’s full name. 	
            January, February, ..., December (en_US);
            Januar, Februar, ..., Dezember (de_DE)

%w 	    Weekday as a decimal number, where 0 is Sunday and 6 is Saturday. 	0, 1, ..., 6 	 
%I 	    Hour (12-hour clock) as a zero-padded decimal number. 	01, 02, ..., 12 	 
%p 	    Locale’s equivalent of either AM or PM. 	
            AM, PM (en_US);
            am, pm (de_DE)
%f 	    Microsecond as a decimal number, zero-padded on the left. 	000000, 000001, ..., 999999 	(4)
%z 	    UTC offset in the form +HHMM or -HHMM (empty string if the the object is naive). 	(empty), +0000, -0400, +1030 	(5)
%Z 	    Time zone name (empty string if the object is naive). 	(empty), UTC, EST, CST 	 
%j 	    Day of the year as a zero-padded decimal number. 	001, 002, ..., 366 	 
%U 	    Week number of the year (Sunday as the first day of the week) as a zero padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0. 	00, 01, ..., 53 	(6)
%W 	    Week number of the year (Monday as the first day of the week) as a decimal number. All days in a new year preceding the first Monday are considered to be in week 0. 	00, 01, ..., 53 	(6)
%c 	    Locale’s appropriate date and time representation. 	
            Tue Aug 16 21:30:00 1988 (en_US);
            Di 16 Aug 21:30:00 1988 (de_DE)
%x 	    Locale’s appropriate date representation. 	
            08/16/88 (None);
            08/16/1988 (en_US);
            16.08.1988 (de_DE)
%X 	    Locale’s appropriate time representation. 	
            21:30:00 (en_US);
            21:30:00 (de_DE)
%% 	    A literal '%' character. 	%

#Example
>>> import os
>>> s = os.stat("regex.py")
>>> s.st_mtime
1425968214.7254572
>>> import datetime
>>> dt = datetime.datetime.fromtimestamp(s.st_mtime)
>>> str(dt)
'2015-03-10 11:46:54.725457'
>>> delta = datetime.timedelta(days=1)
>>> delta
datetime.timedelta(1)
>>> import time
>>> fd = dt + delta
>>> nt = time.mktime(fd.timetuple())
>>> nt
1426054614.0
>>> os.utime("regex.py", (nt,nt))


>>> from datetime import timedelta
>>> year = timedelta(days=365)
>>> another_year = timedelta(weeks=40, days=84, hours=23, minutes=50, seconds=600)  # adds up to 365 days
>>> year.total_seconds()
31536000.0
>>> year == another_year
True
>>> ten_years = 10 * year
>>> ten_years, ten_years.days // 365
(datetime.timedelta(3650), 10)
>>> nine_years = ten_years - year
>>> nine_years, nine_years.days // 365
(datetime.timedelta(3285), 9)
>>> three_years = nine_years // 3;
>>> three_years, three_years.days // 365
(datetime.timedelta(1095), 3)
>>> abs(three_years - ten_years) == 2 * three_years + year
True

>>> import time
>>> from datetime import date
>>> today = date.today()
>>> today
datetime.date(2007, 12, 5)
>>> today == date.fromtimestamp(time.time())
True
>>> my_birthday = date(today.year, 6, 24)
>>> if my_birthday < today:
        my_birthday = my_birthday.replace(year=today.year + 1)
>>> my_birthday
datetime.date(2008, 6, 24)
>>> time_to_birthday = abs(my_birthday - today)
>>> time_to_birthday.days
202
>>> my_birthday.isoformat()
'2002-03-11'
>>> d.strftime("%d/%m/%y")
'11/03/02'
>>> my_birthday.strftime("%A %d. %B %Y")
'Monday 11. March 2002'
>>> 'The {1} is {0:%d}, the {2} is {0:%B}.'.format(my_birthday, "day", "month")
'The day is 11, the month is March.'

>>> from datetime import datetime, date, time
>>> # Using datetime.combine()
>>> d = date(2005, 7, 14)
>>> t = time(12, 30)
>>> datetime.combine(d, t)
datetime.datetime(2005, 7, 14, 12, 30)
>>> # Using datetime.now() or datetime.utcnow()
>>> datetime.now()   
datetime.datetime(2007, 12, 6, 16, 29, 43, 79043)   # GMT +1
>>> datetime.utcnow()   
datetime.datetime(2007, 12, 6, 15, 29, 43, 79060)
>>> # Using datetime.strptime()
>>> dt = datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M")
>>> dt
datetime.datetime(2006, 11, 21, 16, 30)
>>> # Using datetime.timetuple() to get tuple of all attributes
>>> tt = dt.timetuple()
>>> for it in tt:   
...     print it
...
2006    # year
11      # month
21      # day
16      # hour
30      # minute
0       # second
1       # weekday (0 = Monday)
325     # number of days since 1st January
-1      # dst - method tzinfo.dst() returned None
>>> # Date in ISO format
>>> ic = dt.isocalendar()
>>> for it in ic:   
...     print it
...
2006    # ISO year
47      # ISO week
2       # ISO weekday
>>> # Formatting datetime
>>> dt.strftime("%A, %d. %B %Y %I:%M%p")
'Tuesday, 21. November 2006 04:30PM'
>>> 'The {1} is {0:%d}, the {2} is {0:%B}, the {3} is {0:%I:%M%p}.'.format(dt, "day", "month", "time")
'The day is 21, the month is November, the time is 04:30PM.'

