import sys, os, requests, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Direct lottery01 API
print("=== Test 1: Direct lottery01 API ===")
ts = int(time.time() * 1000)
url = f"https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?ts={ts}&pageNo=1"
try:
    r = requests.get(url, headers={"Accept": "application/json", "Content-Type": "application/json;charset=UTF-8"}, timeout=10, verify=False)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        items = d.get("data", {}).get("list", [])
        print(f"Items: {len(items)}")
        if items:
            print(f"First: {items[0]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: fetch_api_data_raw direct
print("\n=== Test 2: fetch_api_data_raw ===")
from helpers import fetch_api_data_raw
data = fetch_api_data_raw(retries=1, timeout=5)
print(f"Type: {type(data).__name__}")
if isinstance(data, list):
    print(f"Items: {len(data)}")
    if data:
        print(f"First: period={data[0].get('period')} cat={data[0].get('category')}")
else:
    print(f"Result: {data}")

# Test 3: fetch_api_data
print("\n=== Test 3: fetch_api_data ===")
from helpers import fetch_api_data
data = fetch_api_data(retries=1, timeout=5, bypass_cache=True)
print(f"Type: {type(data).__name__}")
if isinstance(data, list):
    print(f"Items: {len(data)}")
    if data:
        print(f"First: period={data[0].get('period')} cat={data[0].get('category')}")
else:
    print(f"Result: {data}")

# Test 4: fetch_wingobot_daily_history
print("\n=== Test 4: fetch_wingobot_daily_history ===")
from helpers import fetch_wingobot_daily_history
data = fetch_wingobot_daily_history(retries=1, timeout=5, limit=10)
print(f"Type: {type(data).__name__}")
if isinstance(data, list):
    print(f"Items: {len(data)}")
    if data:
        print(f"First: period={data[0].get('period')} cat={data[0].get('category')}")
else:
    print(f"Result: {data}")

print("\n=== Done ===")
