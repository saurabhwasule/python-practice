import cgi
import cgitb
cgitb.enable(logdir="logs")
form = cgi.FieldStorage()

string = """
<form method="post" action="cgiEx.py">
       <p>Name: <input type="text" name="name"/></p>
	   <p>address1: <input type="text" name="addr"/></p>
	   <p>address2: <input type="text" name="addr"/></p>
	   <input type="submit" value="Submit" />
     </form>  
"""



#headers section
print("Content-Type: text/html")    # HTML is following
print()                             # blank line, end of headers

#content section
print("<HTML>")
print("<TITLE>CGI script output</TITLE>")
print("<BODY>")
print("<H1>This is my first CGI script</H1>")
print("Hello, world!")
print("</br>")
print(form.getfirst("name","").upper(), ",".join(form.getlist("addr")) )
print(string)
print("</BODY></HTML>")





