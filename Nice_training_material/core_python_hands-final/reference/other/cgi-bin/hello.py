#http://localhost:8080/cgi-bin/hello.py
import os , cgi
print("Content-Type:text/html")
print("\n")

print("<html>")
print("<body>")
print("<h1> Hello </h1>") 
cgi.print_directory()
cgi.print_environ_usage()
cgi.print_environ()
print("</body>")
print("</html>")