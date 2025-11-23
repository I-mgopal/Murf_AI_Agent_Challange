import logging
import json
import os
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
    metrics,
    tokenize,
    function_tool,
    RunContext,
    MetricsCollectedEvent,  # ✔ correct import
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("coffeeAgent")

load_dotenv(".env.local")

# -------------------------------------------------------------------
# JSON SAVE DIRECTORY (backend/src/orderlist)
# -------------------------------------------------------------------
ORDER_DIR = Path(__file__).parent / "orderlist"
ORDER_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# TOOL: save_order (LLM will call this once order is complete)
# -------------------------------------------------------------------
@function_tool
async def save_order(
    context: RunContext,
    drinkType: str,
    size: str,
    milk: str,
    extras: list[str],
    name: str,
):
    """
    Save a completed coffee order to a JSON file.
    The LLM must call this only after confirming all fields with the user.
    """

    order_state = {
        "drinkType": drinkType,
        "size": size,
        "milk": milk,
        "extras": extras,
        "name": name,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"order_{ts}.json"
    filepath = ORDER_DIR / filename

    print(f"DEBUG >>> saving JSON file to: {filepath}")
    logger.info(f"Saving order JSON to: {filepath}")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(order_state, f, indent=2)

        return {
            "status": "saved",
            "filepath": str(filepath),
            "order": order_state,
        }

    except Exception as e:
        logger.error(f"Failed to save order: {e}")
        return {"status": "error", "error": str(e)}


# -------------------------------------------------------------------
# Build the barista agent
# -------------------------------------------------------------------
def create_barista_agent() -> Agent:
    instructions = """
    You are Mathew, a friendly barista at MoonBrew Cafe.

    Your job is to help the user place a complete coffee order.

    You conceptually maintain this order structure:
    {
      "drinkType": "string",
      "size": "string",
      "milk": "string",
      "extras": ["string"],
      "name": "string"
    }

    Rules:
    - Greet the user warmly as Mathew.
    - Ask for the user's name first and remember it.
    - Ask clarifying questions one-by-one until:
        drinkType is known
        size is known
        milk is known
        extras are gathered or confirmed
    - Never assume—always confirm each answer.

    When all fields are known AND the user confirms:
    - Call the save_order tool exactly once.
      Pass:
        drinkType, size, milk, extras, name

    After the tool returns:
    - Read a short, neat summary back to the user.
    - Say that the order has been saved.
    - Keep tone friendly and concise. No emojis, no bold formatting.
    """

    return Agent(
        instructions=instructions,
        tools=[save_order],  # Attach tool so LLM can call it
    )


# -------------------------------------------------------------------
# PREWARM
# -------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with our Barista Agent
    await session.start(
        agent=create_barista_agent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
