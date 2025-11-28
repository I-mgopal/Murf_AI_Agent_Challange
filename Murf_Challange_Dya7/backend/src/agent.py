import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import deepgram, murf, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("shopping_agent")
load_dotenv(".env.local")

# ---------------------------------------------
# Paths
# ---------------------------------------------
BASE_DIR = Path(__file__).parent
CATALOG_FILE = BASE_DIR / "catalog_day7.json"
ORDERS_DIR = BASE_DIR / "orders"
ORDERS_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------
# Load Catalog
# ---------------------------------------------
def load_catalog():
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)

CATALOG = load_catalog()

# ---------------------------------------------
# Recipe Map
# ---------------------------------------------
RECIPES = {
    "peanut butter sandwich": ["bread_wholewheat", "peanut_butter"],
    "pasta": ["pasta_spaghetti", "tomato_sauce"],
}

# ---------------------------------------------
# State
# ---------------------------------------------
CART = {}
ORDER_INFO = {
    "name": None,
    "address": None,
}

# ---------------------------------------------
# Tools
# ---------------------------------------------
@function_tool
async def add_to_cart(context: RunContext, item_id: str, quantity: int):
    CART[item_id] = CART.get(item_id, 0) + quantity
    return {"added": item_id, "qty": CART[item_id]}

@function_tool
async def remove_from_cart(context: RunContext, item_id: str):
    if item_id in CART:
        del CART[item_id]
        return {"removed": True}
    return {"removed": False}

@function_tool
async def save_order(context: RunContext, customer_name: str, address: str):
    order_items = []
    total = 0

    for item_id, qty in CART.items():
        product = next((x for x in CATALOG if x["id"] == item_id), None)
        if not product:
            continue

        subtotal = product["price"] * qty
        total += subtotal

        order_items.append({
            "id": item_id,
            "name": product["name"],
            "qty": qty,
            "price": product["price"],
            "subtotal": subtotal
        })

    order_data = {
        "customer_name": customer_name,
        "address": address,
        "items": order_items,
        "total": total,
        "timestamp": datetime.now().isoformat()
    }

    filename = ORDERS_DIR / f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(order_data, f, indent=2)

    CART.clear()
    ORDER_INFO["name"] = None
    ORDER_INFO["address"] = None

    return {"saved": True, "file": str(filename)}


# ---------------------------------------------
# Agent
# ---------------------------------------------
class ShoppingAgent(Agent):
    def __init__(self):
        instructions = """
        You are a friendly shopping assistant for SwiftCart.
        You help users order groceries and simple meal ingredients.

        Ordering flow:
        • Detect when user says: "place my order", "I'm done", "that's all".
        • Ask for name.
        • Ask for address.
        • Use save_order() tool.

        Rules:
        • ALWAYS use the tools instead of explaining them.
        • Do not invent items that are not in the catalog.
        """
        super().__init__(
            instructions=instructions,
            tools=[add_to_cart, remove_from_cart, save_order]
        )

    async def on_message(self, msg, ctx):
        text = msg.text.lower()

        # ---------------------------------------
        # Step 1 — Detect order completion
        # ---------------------------------------
        if "place my order" in text or "i'm done" in text or "that's all" in text:
            await ctx.send_message("Sure! What name should I place the order under?")
            return

        # ---------------------------------------
        # Step 2 — Capture customer name
        # ---------------------------------------
        if ORDER_INFO["name"] is None and text.replace(" ", "").isalpha():
            ORDER_INFO["name"] = msg.text
            await ctx.send_message("Great! What’s the full delivery address?")
            return

        # ---------------------------------------
        # Step 3 — Capture address + Save Order
        # ---------------------------------------
        if ORDER_INFO["name"] is not None and ORDER_INFO["address"] is None and len(text.split()) > 3:
            ORDER_INFO["address"] = msg.text

            result = await save_order.invoke(ctx, {
                "customer_name": ORDER_INFO["name"],
                "address": ORDER_INFO["address"]
            })

            await ctx.send_message(f"Your order has been placed successfully! Saved to: {result['file']}")
            return

        # ---------------------------------------
        # Show Cart
        # ---------------------------------------
        if "cart" in text:
            if not CART:
                await ctx.send_message("Your cart is currently empty.")
            else:
                summary = []
                for item_id, qty in CART.items():
                    product = next(x for x in CATALOG if x["id"] == item_id)
                    summary.append(f"{product['name']} x{qty}")
                await ctx.send_message("Your cart contains: " + ", ".join(summary))
            return

        # ---------------------------------------
        # Recipes (multi-add)
        # ---------------------------------------
        for recipe, ids in RECIPES.items():
            if recipe in text:
                for item_id in ids:
                    await add_to_cart.invoke(ctx, {"item_id": item_id, "quantity": 1})

                names = [next(x for x in CATALOG if x["id"] == i)["name"] for i in ids]
                await ctx.send_message(
                    f"I added {', '.join(names)} for your {recipe}."
                )
                return

        # ---------------------------------------
        # Add single item
        # ---------------------------------------
        for product in CATALOG:
            if product["name"].lower() in text or any(t in text for t in product["tags"]):
                await add_to_cart.invoke(ctx, {
                    "item_id": product["id"],
                    "quantity": 1
                })
                await ctx.send_message(f"Added {product['name']} to your cart.")
                return

        # ---------------------------------------
        # Default response
        # ---------------------------------------
        await ctx.send_message("Sure — what else would you like to add?")


# ---------------------------------------------
# Prewarm
# ---------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# ---------------------------------------------
# Entrypoint
# ---------------------------------------------
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True
    )

    await session.start(
        agent=ShoppingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
