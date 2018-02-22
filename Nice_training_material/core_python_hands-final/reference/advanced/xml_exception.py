#Use BeautifulSoup for HTML/XML processing
#no xpath, but css select 
#pip install requests
#pip install BeautifulSoup4

from bs4 import BeautifulSoup
import requests

r  = requests.get("http://www.yahoo.com")
data = r.text

soup = BeautifulSoup(data)

for link in soup.find_all('a'):  # tag <a href=".."
		print(link.get('href'))	 # Attribute href

#######Exception
try:
    1/0
except Exception as e:
    print(e)
    raise RuntimeError("Bad") from e
finally:
    print("End")
