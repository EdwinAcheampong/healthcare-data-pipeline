import requests
import json
import time

API_URL = "http://localhost:8000/api/v1/predict/workload"

sample_payload = {
    "model_type": "advanced",
    "input_data": {
        "age": 55,
        "encounter_count": 10,
        "condition_count": 3,
        "medication_count": 5,
        "avg_duration": 4.5,
        "healthcare_expenses": 2500.0
    }
}

def send_requests(num_requests=100):
    """Sends a specified number of requests to the API."""
    for i in range(num_requests):
        try:
            response = requests.post(API_URL, data=json.dumps(sample_payload))
            print(f"Request {i+1}/{num_requests} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(0.5) # Wait half a second between requests

if __name__ == "__main__":
    print("Starting API load generation...")
    send_requests()
    print("Load generation finished.")
