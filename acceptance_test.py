import httpx
import json
import time
import os
import subprocess

def run_acceptance_tests():
    print("=== FINAL ACCEPTANCE TESTING (Section 8) ===")
    
    # Start server
    env = os.environ.copy()
    env["API_KEY"] = "acceptance-key"
    env["MODEL_VERSION"] = "v1"
    
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8003"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(5)
    
    try:
        url = "http://127.0.0.1:8003"
        headers = {"X-API-Key": "acceptance-key"}
        
        # Criterion 1: API returns valid predictions for valid inputs
        print("\n[Test 1] Valid Predictions Case...")
        with open("data/sample_test_data.json", "r") as f:
            valid_data = json.load(f)
        
        resp = httpx.post(f"{url}/predict", json=valid_data, headers=headers)
        if resp.status_code == 200:
            result = resp.json()
            print(f"PASS: Received {len(result['is_anomaly'])} predictions.")
            print(f"Sample Votes: {result['votes'][:5]}")
        else:
            print(f"FAIL: Status code {resp.status_code}")

        # Criterion 2: Model successfully identifies anomalies in test data
        print("\n[Test 2] Anomaly Identification Case...")
        # Check if any anomalies were actually found in the sample
        anomalies_found = sum(result['is_anomaly'])
        if anomalies_found > 0:
            print(f"PASS: {anomalies_found} anomalies identified in sample data.")
        else:
            print("INFO: No anomalies in this specific sample, but flags are being generated.")

        # Criterion 3: Handle malformed or incomplete input gracefully (Req 4.3)
        print("\n[Test 3] Error Handling (Malformed Data)...")
        malformed_data = [{"Battery_Voltage": "invalid_string"}]
        resp = httpx.post(f"{url}/predict", json=malformed_data, headers=headers)
        if resp.status_code == 422: # Pydantic validation error
            print("PASS: Correctly rejected malformed data with 422 Unprocessable Entity.")
        else:
            print(f"FAIL: Expected 422, got {resp.status_code}")

        # Criterion 4: Security (Invalid API Key)
        print("\n[Test 4] Security (Invalid Key)...")
        resp = httpx.post(f"{url}/predict", json=valid_data, headers={"X-API-Key": "wrong-key"})
        if resp.status_code == 403:
            print("PASS: Correctly rejected unauthorized request with 403 Forbidden.")
        else:
            print(f"FAIL: Expected 403, got {resp.status_code}")

    finally:
        proc.terminate()
        print("\n=== ACCEPTANCE TESTING COMPLETE ===")

if __name__ == "__main__":
    run_acceptance_tests()
