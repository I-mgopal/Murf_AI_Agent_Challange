import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from datetime import datetime

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
    MetricsCollectedEvent,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ------------------------------------------------------
# Logging + Env
# ------------------------------------------------------
logger = logging.getLogger("day4_tutor")
load_dotenv(".env.local")

# ------------------------------------------------------
# File Paths
# ------------------------------------------------------
BASE_DIR = Path(__file__).parent
CONTENT_FILE = BASE_DIR.parent / "shared-data" / "day4_tutor_content.json"

# ------------------------------------------------------
# Load Tutor JSON Content
# ------------------------------------------------------
def load_content():
    try:
        if CONTENT_FILE.exists():
            with open(CONTENT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        logger.error("Tutor content JSON missing or invalid.")
    except Exception as e:
        logger.error(f"Failed to load content file: {e}")
    return []

COURSE_CONTENT = load_content()

# ------------------------------------------------------
# Configure Murf Voices for Modes
# ------------------------------------------------------
VOICE_BY_MODE = {
    "learn": "en-US-matthew",
    "quiz": "en-US-alicia",
    "teach_back": "en-US-ken",
}

# ------------------------------------------------------
# Helper: Find Concept
# ------------------------------------------------------
def find_concept_from_text(text: str):
    text = text.lower()
    for c in COURSE_CONTENT:
        if c["id"].lower() in text:
            return c
        if c["title"].lower() in text:
            return c
    return None

# ------------------------------------------------------
# MAIN AGENT (Three Modes)
# ------------------------------------------------------
class TeachTheTutor(Agent):
    def __init__(self):
        concepts_list = ", ".join(c["title"] for c in COURSE_CONTENT)

        super().__init__(
            instructions=f"""
            You are an Active Recall Coach with 3 modes:
            - learn: explain concepts clearly
            - quiz: ask questions from the content file
            - teach_back: ask the user to explain concepts

            Rules:
            - Use ONLY content from the JSON file.
            - Switch modes when the user says "switch to learn/quiz/teach back".
            - Be friendly, helpful, and concise.
            - Never invent concepts not in the JSON.

            Available concepts:
            {concepts_list}
            """
        )

        self.mode = None
        self.current_concept = None
        self.phase = "await_mode"

    # --------------------------------------------------
    # Change Murf Voice Based on Mode
    # --------------------------------------------------
    async def _set_voice(self, context, mode):
        voice_model = VOICE_BY_MODE.get(mode)
        if not voice_model:
            return

        new_tts = murf.TTS(
            voice=voice_model,
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        )

        await context.session.update_tts(new_tts)
        logger.info(f"Voice changed to {voice_model} for mode {mode}")

    # --------------------------------------------------
    async def on_join(self, context):
        await context.send_speech(
            "Hello! I'm your Active Recall Coach. "
            "Would you like to learn, be quizzed, or try teach-back mode?"
        )
        self.phase = "await_mode"

    # --------------------------------------------------
    async def on_user_message(self, msg, context):
        text_raw = msg.text.strip()
        text = text_raw.lower()

        # -------- Mode Switching ANYTIME --------
        if "learn" in text:
            await self._switch_mode("learn", context)
            return
        if "quiz" in text:
            await self._switch_mode("quiz", context)
            return
        if "teach" in text:
            await self._switch_mode("teach_back", context)
            return

        # If no mode chosen yet
        if self.mode is None:
            await context.send_speech("Pick a mode: learn, quiz, or teach-back.")
            return

        # -------- Mode Handlers --------
        if self.mode == "learn":
            await self._mode_learn(text_raw, context)
        elif self.mode == "quiz":
            await self._mode_quiz(text_raw, context)
        elif self.mode == "teach_back":
            await self._mode_teach_back(text_raw, context)

    # --------------------------------------------------
    # Switch Modes
    # --------------------------------------------------
    async def _switch_mode(self, mode, context):
        self.mode = mode
        self.current_concept = None
        self.phase = "await_concept"

        await self._set_voice(context, mode)

        if mode == "learn":
            await context.send_speech("You're now in learn mode. Which concept?")
        elif mode == "quiz":
            await context.send_speech("Quiz mode activated. Which concept?")
        elif mode == "teach_back":
            await context.send_speech("Teach-back mode. What concept will you explain?")

    # --------------------------------------------------
    # LEARN MODE
    # --------------------------------------------------
    async def _mode_learn(self, user_text, context):
        if self.current_concept is None:
            concept = find_concept_from_text(user_text)
            if not concept:
                await context.send_speech(
                    "Which concept do you want to learn? Try 'variables' or 'loops'."
                )
                return

            self.current_concept = concept
            await context.send_speech(
                f"{concept['title']}: {concept['summary']}"
            )
            await context.send_speech(
                "Would you like another concept, or switch modes?"
            )
            return

        if "another" in user_text.lower():
            self.current_concept = None
            await context.send_speech("Sure — which concept next?")
        else:
            await context.send_speech(
                "Say 'another concept' or switch to quiz / teach-back mode."
            )

    # --------------------------------------------------
    # QUIZ MODE
    # --------------------------------------------------
    async def _mode_quiz(self, user_text, context):
        if self.current_concept is None:
            concept = find_concept_from_text(user_text)
            if not concept:
                await context.send_speech("Which concept should I quiz you on?")
                return

            self.current_concept = concept
            self.phase = "quiz_wait_answer"
            await context.send_speech(concept["sample_question"])
            return

        # Wait for user answer
        if self.phase == "quiz_wait_answer":
            await context.send_speech("Thanks! Here's a reminder of the concept:")
            await context.send_speech(self.current_concept["summary"])
            await context.send_speech(
                "Another question, another concept, or switch modes?"
            )
            # stays in quiz mode

    # --------------------------------------------------
    # TEACH-BACK MODE
    # --------------------------------------------------
    async def _mode_teach_back(self, user_text, context):
        if self.current_concept is None:
            concept = find_concept_from_text(user_text)
            if not concept:
                await context.send_speech("Which concept will you teach back?")
                return

            self.current_concept = concept
            self.phase = "teach_wait_answer"
            await context.send_speech(
                f"Teach this back to me: {concept['sample_question']}"
            )
            return

        # Waiting for user’s explanation
        if self.phase == "teach_wait_answer":
            wc = len(user_text.split())
            if wc < 8:
                fb = "That was short — try adding what it does and why it's useful."
            elif wc < 20:
                fb = "Nice! You covered the basics. Add an example to make it stronger."
            else:
                fb = "Great explanation! Clear structure and solid detail."

            await context.send_speech(fb)
            await context.send_speech(
                "Here’s a clean summary: " + self.current_concept["summary"]
            )
            await context.send_speech(
                "Want another concept, or switch modes?"
            )

# ------------------------------------------------------
# PREWARM
# ------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------
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

    collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=TeachTheTutor(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

# ------------------------------------------------------
# RUN
# ------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
