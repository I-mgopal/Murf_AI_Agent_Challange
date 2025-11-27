import json
import logging
from pathlib import Path
from datetime import datetime

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
    MetricsCollectedEvent,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("fraud_agent")
load_dotenv(".env.local")

# -------------------------------------------------------
# "Database" File (JSON)
# -------------------------------------------------------
BASE_DIR = Path(__file__).parent  # backend/src
DB_FILE = BASE_DIR / "day6_fraud_cases.json"


def load_cases() -> list[dict]:
    if not DB_FILE.exists():
        logger.error(f"Fraud DB file not found at: {DB_FILE}")
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.error("Fraud DB JSON is not a list.")
            return []
    except Exception as e:
        logger.error(f"Error reading fraud DB: {e}")
        return []


def save_cases(cases: list[dict]) -> None:
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
        logger.info(f"Fraud DB updated at {DB_FILE}")
    except Exception as e:
        logger.error(f"Error writing fraud DB: {e}")


def find_case_by_username(username: str) -> dict | None:
    uname = username.strip().lower()
    for c in load_cases():
        if c.get("userName", "").strip().lower() == uname:
            return c
    return None


# -------------------------------------------------------
# Tools: get_case + update_case
# -------------------------------------------------------
@function_tool
async def get_fraud_case(context: RunContext, userName: str):
    """
    Lookup a fraud case for the given userName (case-insensitive).

    Returns:
      - found: bool
      - case: the fraud case object if found
    """
    case = find_case_by_username(userName)
    if not case:
        return {"found": False, "case": None}

    # SECURITY: do not send full card number etc (we only store masked)
    return {
        "found": True,
        "case": case,
    }


@function_tool
async def update_fraud_case(
    context: RunContext,
    userName: str,
    status: str,
    outcomeNote: str,
):
    """
    Update a fraud case status for the given userName.

    Args:
      userName: Customer name (matching DB).
      status: one of pending_review, confirmed_safe, confirmed_fraud, verification_failed.
      outcomeNote: Short summary of what happened.
    """
    username_lower = userName.strip().lower()
    cases = load_cases()
    updated = False

    for c in cases:
        if c.get("userName", "").strip().lower() == username_lower:
            c["status"] = status
            c["outcomeNote"] = outcomeNote
            c["updatedAt"] = datetime.now().isoformat(timespec="seconds")
            updated = True
            logger.info(f"Updated fraud case for {c.get('userName')}: {status} - {outcomeNote}")
            break

    if updated:
        save_cases(cases)
        return {"success": True, "status": status, "note": outcomeNote}
    else:
        return {"success": False, "error": "Case not found for username."}


# -------------------------------------------------------
# Fraud Agent Persona
# -------------------------------------------------------
class FraudAgent(Agent):
    def __init__(self):
        instructions = """
        You are a calm, professional fraud detection representative
        for a fictional Indian bank called Safeguard Bank India.

        Primary goal:
        - Handle a single fraud alert call for a suspicious card transaction.
        - Use the tools get_fraud_case and update_fraud_case to interact with
          the fraud case database.

        VERY IMPORTANT SAFETY:
        - NEVER ask for full card numbers, PINs, passwords, or OTPs.
        - Only use non-sensitive verification like a security question
          that is provided in the case data.
        - Treat this as a demo / sandbox environment only.

        Call flow you MUST follow:

        1) Introduction
           - Introduce yourself as "Safeguard Bank India fraud department".
           - Confirm you are calling about a suspicious card transaction.
           - Ask politely for the customer's first name to locate their case.

        2) Load fraud case
           - Once user gives a name, call get_fraud_case(userName).
           - If found:
                * Do NOT immediately read all details.
                * Read only non-sensitive identity-related fields needed
                  for verification.
             If not found:
                * Say you cannot find a case for that name and end politely.

        3) Basic verification
           - Use the securityQuestion from the fraud case.
           - Ask the user that question.
           - Compare their answer with securityAnswer from the case.
           - Never say or read out the securityAnswer itself.
           - If the answer does not match after one or two attempts:
                * Call update_fraud_case with status="verification_failed"
                  and a short outcomeNote.
                * Politely say you cannot proceed without verification, and end.

        4) Read suspicious transaction details (only after verification)
           - Read:
                * merchantName
                * transactionAmount (with currency)
                * masked cardEnding (like 'ending in 4242')
                * approximate transactionTime
                * transactionLocation
           - Then ask clearly:
                * "Did you make this transaction yourself?" (yes/no style).

        5) Branch on answer:
           - If user confirms they DID make the transaction:
                * Call update_fraud_case with:
                    status="confirmed_safe"
                    outcomeNote="Customer confirmed the transaction is legitimate."
                * Tell them the transaction will remain approved and no block is placed.
           - If user says they did NOT make the transaction:
                * Call update_fraud_case with:
                    status="confirmed_fraud"
                    outcomeNote="Customer denied the transaction; mock card block and dispute."
                * Tell them you will block the card and raise a dispute
                  (all mock, no real action), and reassure next steps.

        6) End-of-call summary
           - Briefly recap what happened:
                * Whether transaction was confirmed safe or fraudulent.
                * What action (mock) was taken (e.g., card blocked, dispute raised).
           - Thank the user for their time and say goodbye.

        Tone:
        - Calm, professional, reassuring.
        - No jokes or casual slang.
        - Clear, short sentences.
        - Remind the user this is about a demo/sandbox scenario if needed.

        Tools usage:
        - Use get_fraud_case exactly once per call, after the user gives their name.
        - Use update_fraud_case exactly once at the end, when you know the outcome.
        - Do NOT expose internal DB fields like securityAnswer directly;
          only use them to internally decide if verification passed.

        """
        super().__init__(instructions=instructions, tools=[get_fraud_case, update_fraud_case])


# -------------------------------------------------------
# Prewarm VAD
# -------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------------------------------------
# Entrypoint
# -------------------------------------------------------
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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FraudAgent(),
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
