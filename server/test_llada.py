import requests

BASE_URL = "http://localhost:8000"

r = requests.get(f"{BASE_URL}/health")
print("Health:", r.json())

r = requests.post(f"{BASE_URL}/generate", json={
    "prompt": "Explain attention mechanism in transformer?",
    "gen_length": 256,
    "block_length": 64
})
print("Response:", r.json())