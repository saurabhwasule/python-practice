#http://localhost:8080/cgi-bin/hello.py
#python3: python -m http.server --cgi 8080
#py2.7: python -m CGIHTTPServer  8080
print("Content-Type: text/html")
print()
prefix = "<html><body>"
suffix="</body></html>"
print(prefix)
print("<h1>Hello </h1>")
print(suffix)