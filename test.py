import requests

url = "https://shl-rag-assessment.onrender.com/recommend"
payload = {
    "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
}

response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response body:", response.text)
