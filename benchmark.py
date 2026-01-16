import httpx
import time
import json
import os
import subprocess
import signal

def run_benchmark():
    print("Starting API Latency Benchmark (Req 4.1)...")
    
    # 1. Start Server in background
    env = os.environ.copy()
    env["API_KEY"] = "benchmark-key"
    env["MODEL_VERSION"] = "v1"
    
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8001"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(5)  # Wait for startup
    
    try:
        # 2. Prepare data
        with open("data/sample_test_data.json", "r") as f:
            data = json.load(f)
        
        # 3. Perform Requests
        latencies = []
        url = "http://127.0.0.1:8001/predict"
        headers = {"X-API-Key": "benchmark-key"}
        
        print(f"Sending 10 requests to {url}...")
        with httpx.Client() as client:
            for i in range(10):
                start = time.time()
                resp = client.post(url, json=data, headers=headers)
                latencies.append((time.time() - start) * 1000)
                if resp.status_code != 200:
                    print(f"Request {i} failed: {resp.status_code}")
                
        # 4. Report
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        print(f"\n--- Benchmark Results ---")
        print(f"Average Latency: {avg_latency:.2f}ms")
        print(f"Max Latency:     {max_latency:.2f}ms")
        
        if avg_latency < 500:
            print("SUCCESS: Latency is within 500ms limit.")
        else:
            print("FAILURE: Latency exceeds 500ms limit.")
            
    finally:
        # 5. Cleanup
        proc.terminate()
        print("Server shut down.")

if __name__ == "__main__":
    run_benchmark()
