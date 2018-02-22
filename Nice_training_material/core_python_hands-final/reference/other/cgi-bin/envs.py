#http://localhost:8080/cgi-bin/envs.py
import os 
print("Content-Type:text/html")
print("\n")

print("<html>")

print("<body>")
print("<table border=1>")
for k in sorted(os.environ):
    print("<tr><td>%s</td><td>%s</td></tr>" % (os.environ[k],k) )
print("</table>")   
print("</body>")
print("</html>")