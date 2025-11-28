from dotenv import load_dotenv
load_dotenv()
import requests
import os
import json







BASE_URL = os.getenv("CANVAS_URL", "https://boardv28.vercel.app")
print("#### chroma_script.py CANVAS_URL : ", BASE_URL)



def board_items_process(data):
    exclude_keys = ["x","y","width","height","createdAt","updatedAt","color","rotation"]
    clean_data = []
    for item in data:
        if item.get('type') == 'ehrHub' or item.get('type') == 'zone' or item.get('type') == 'button':
            pass
        else:   
            clean_item = {}
            for k,v in item.items():
                if k not in exclude_keys:
                    clean_item[k] = v
            clean_data.append(clean_item)


    return clean_data

def get_board_items():
    url = BASE_URL + "/api/board-items"
    data = []
    
    # 1. Try fetching from API
    try:
        print(f"🌍 Fetching from: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            try:
                data = response.json()
                data = board_items_process(data)
                # Save to cache
                os.makedirs("output", exist_ok=True)
                with open(f"output/board_items.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                print(f"✅ Fetched {len(data)} items from API")
                return data
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response from API. Text: {response.text[:100]}...")
        else:
            print(f"⚠️ API Error: Status {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ API Connection failed: {e}")

        # 2. Fallback to local file
        local_path = f"output/board_items.json"
        if os.path.exists(local_path):
            print(f"📂 Falling back to local cache: {local_path}")
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"✅ Loaded {len(data)} items from cache")
                    return data
            except Exception as e:
                print(f"❌ Failed to load local cache: {e}")
            
    return []



get_board_items()