import os
import json
from datetime import datetime

order_state = {
    "drinkType": "latte",
    "size": "small",
    "milk": "oat",
    "extras": ["sugar"],
    "name": "TestUser"
}

def save_order_to_json():
    folder = os.path.join(os.path.dirname(__file__), "orderlist")
    print("Saving to folder:", folder)

    os.makedirs(folder, exist_ok=True)

    filename = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(folder, filename)
    print("Saving file:", filepath)

    with open(filepath, "w") as f:
        json.dump(order_state, f, indent=2)

    print("Saved successfully:", filepath)

save_order_to_json()
