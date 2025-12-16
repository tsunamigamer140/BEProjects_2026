import requests


API_KEY = "AIzaSyBKA4u914qb_1Y-0We7H30aJ-STdSb09bU" 
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(url)
print(response.status_code)
print(response.json())
