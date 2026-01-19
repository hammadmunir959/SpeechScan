import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_async_flow(file_path):
    print(f"🚀 Uploading {file_path} for analysis...")
    
    # 1. Submit job
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    if response.status_code != 200:
        print(f"❌ Submission failed: {response.text}")
        return
    
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✅ Job submitted! ID: {job_id}")
    
    # 2. Poll for results
    print("⏳ Polling for results...")
    start_time = time.time()
    while True:
        res = requests.get(f"{BASE_URL}/results/{job_id}")
        if res.status_code != 200:
            print(f"❌ Polling failed: {res.text}")
            break
            
        status_data = res.json()
        status = status_data["status"]
        
        if status == "completed":
            print(f"🎉 Analysis Complete in {time.time() - start_time:.2f}s!")
            print("Results:")
            print(status_data["result"])
            break
        elif status == "failed":
            print(f"❌ Job failed: {status_data.get('error')}")
            break
        else:
            print(f"Status: {status}...")
            time.sleep(1)
            
        if time.time() - start_time > 60:
            print("⏰ Timeout waiting for result")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_async.py <audio_file_path>")
        sys.exit(1)
    test_async_flow(sys.argv[1])
