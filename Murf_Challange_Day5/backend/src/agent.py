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
    metrics,
    tokenize,
    function_tool,
    RunContext,
    MetricsCollectedEvent,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sdr_agent")
load_dotenv(".env.local")

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).parent            # backend/src
FAQ_FILE = BASE_DIR.parent / "shared-data" / "day1_moonbill_faq.json"
LEADS_DIR = BASE_DIR / "leads"
LEADS_DIR.mkdir(parents=True, exist_ok=True)


def load_faq() -> list[dict]:
    if not FAQ_FILE.exists():
        logger.error(f"FAQ file not found at: {FAQ_FILE}")
        return []
    try:
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.error("FAQ file is not a list")
            return []
    except Exception as e:
        logger.error(f"Error loading FAQ: {e}")
        return []


FAQ_DATA = load_faq()


def find_faq_matches(query: str, max_results: int = 3) -> list[dict]:
    """Very simple keyword match over question + tags."""
    q = query.lower()
    matches = []
    for item in FAQ_DATA:
        text = (item.get("question", "") + " " + " ".join(item.get("tags", []))).lower()
        score = 0
        for token in q.split():
            if token in text:
                score += 1
        if score > 0:
            matches.append((score, item))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:max_results]]


# --------------------------------------------------
# Tools
# --------------------------------------------------
@function_tool
async def search_faq(context: RunContext, question: str):
    """
    Search the Moonbill FAQ and return relevant entries.

    Use this tool whenever the user asks about:
    - what the product does
    - who it is for
    - pricing / free tier
    - integrations
    - support

    Args:
      question: The user's question in natural language.
    """
    results = find_faq_matches(question)
    if not results:
        return {
            "found": False,
            "answer": "I could not find a direct FAQ match. You may need to clarify or I can answer more generally.",
        }
    # Return combined answer text
    answers = []
    for r in results:
        answers.append(f"Q: {r['question']}\nA: {r['answer']}")
    return {
        "found": True,
        "answer": "\n\n".join(answers),
    }


@function_tool
async def save_lead(
    context: RunContext,
    name: str,
    company: str,
    email: str,
    role: str,
    use_case: str,
    team_size: str,
    timeline: str,
):
    """
    Save a qualified lead to a JSON file.

    Call this tool once when:
    - The user seems done with the conversation.
    - You have at least their name, email, company (or say 'unknown' if not given).

    Args:
      name: Prospect's name.
      company: Company name.
      email: Email address.
      role: Their role (e.g. founder, engineer).
      use_case: What they want to use Moonbill for.
      team_size: Size of their team or org.
      timeline: When they want to start (now / soon / later).
    """
    lead = {
        "name": name,
        "company": company,
        "email": email,
        "role": role,
        "use_case": use_case,
        "team_size": team_size,
        "timeline": timeline,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = LEADS_DIR / f"lead_{timestamp}.json"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(lead, f, indent=2)

        logger.info(f"Saved lead JSON to {filename}")
        print(f"DEBUG >>> Saved lead JSON to: {filename}")
        return {
            "status": "saved",
            "file": str(filename),
            "lead": lead,
        }
    except Exception as e:
        logger.error(f"Failed to save lead: {e}")
        return {"status": "error", "error": str(e)}


# --------------------------------------------------
# SDR Agent
# --------------------------------------------------
class SDRAgent(Agent):
    def __init__(self):
        instructions = """
        You are a Sales Development Representative (SDR) for Moonbill and Your name is jhonathan,
        an Indian SaaS startup that helps businesses manage online payments,
        subscriptions, and invoicing.

        Your goals:
        1) Greet visitors warmly and professionally.
        2) Ask what brought them here and what they are working on.
        3) Understand their needs and use case.
        4) Answer questions about Moonbill using the FAQ tool.
        5) Collect key lead details and save them via the save_lead tool.
        6) Give a short summary at the end.

        FAQ usage:
        - When the user asks about product, pricing, free tier, who it's for,
          integrations, or support, call search_faq with their question.
        - Use the answer returned. Do not invent extra details not in the FAQ.

        Lead fields to collect:
        - name
        - company
        - email
        - role
        - use_case
        - team_size
        - timeline (now / soon / later)

        Conversation style:
        - Start by greeting and asking what brought them here.
        - Ask natural follow-up questions to fill the lead fields.
        - Do NOT dump all questions at once. Ask them gradually.
        - Keep the conversation focused on their needs and Moonbill's fit.

        End-of-call behavior:
        - When the user says something like "that's all", "I'm done", or "thanks",
          you should:
            1) Confirm you have their details.
            2) Call save_lead exactly once with the information you've collected.
            3) After the tool returns, give a short summary:
               - who they are
               - what they want to use Moonbill for
               - their timeline
            4) Thank them and close politely.

        Important:
        - Do not claim Moonbill can do things that are not mentioned in the FAQ.
        - If the FAQ doesn't cover something, say you don’t have that detail
          and steer back to areas you can help with.
        - Stay concise, friendly and clear. No emojis or fancy formatting.
        """

        super().__init__(instructions=instructions, tools=[search_faq, save_lead])


# --------------------------------------------------
# VAD Prewarm
# --------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------
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
        agent=SDRAgent(),
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
