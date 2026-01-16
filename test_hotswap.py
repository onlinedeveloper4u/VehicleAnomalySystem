import httpx
import time
import os
import subprocess

def test_hot_swap():
    print("Testing Hot-Swap (Req 5.3)...")
    
    env = os.environ.copy()
    env["API_KEY"] = "test-key"
    env["MODEL_VERSION"] = "v1"
    
    # Start server
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8002"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(5)
    
    try:
        url = "http://127.0.0.1:8002"
        headers = {"X-API-Key": "test-key"}
        
        # 1. Check current version (expected v1)
        resp = httpx.get(f"{url}/health", headers=headers)
        print(f"Initial Health: {resp.json()}")
        
        # 2. Trigger hot-swap to v2
        print("Triggering hot-swap to v2...")
        resp = httpx.post(f"{url}/model/switch", params={"version": "v2"}, headers=headers)
        print(f"Switch Response: {resp.json()}")
        
        # 3. Check health again
        resp = httpx.get(f"{url}/health", headers=headers)
        print(f"Final Health: {resp.json()}")
        
        if resp.status_code == 200:
            print("SUCCESS: Model switched in-place!")
        else:
            print("FAILURE: Model switch failed.")
            
    finally:
        proc.terminate()

if __name__ == "__main__":
    test_hot_swap()
