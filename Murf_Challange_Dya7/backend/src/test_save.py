from pathlib import Path
import json
from datetime import datetime

BASE = Path(__file__).parent
ORDERS = BASE / "orders"
ORDERS.mkdir(exist_ok=True)

file = ORDERS / f"test_order_{datetime.now().strftime('%H%M%S')}.json"

data = {"test": True}

with open(file, "w") as f:
    json.dump(data, f, indent=2)

print("SAVED:", file)
