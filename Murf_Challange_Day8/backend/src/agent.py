import logging
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
)
from livekit.plugins import deepgram, google, murf, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("game_master")
load_dotenv(".env.local")

BASE_DIR = Path(__file__).parent


class GameMasterAgent(Agent):
    """
    Day 8 – D&D-style Game Master
    Universe: Fantasy adventure in the floating city of Aeralith.
    Tone: Cinematic, slightly dramatic, but easy to follow in audio.
    """

    def __init__(self) -> None:
        instructions = """
        You are a voice-only Game Master (GM) running a fantasy adventure
        in the sky-city of Aeralith, a world of floating islands, airships,
        ancient ruins, and dragons.

        Your role:
        - You describe scenes, dangers, NPCs, and outcomes.
        - You NEVER take actions for the player.
        - You ALWAYS end your message with a clear prompt like:
          "What do you do?" or "How do you respond?"

        Style:
        - Speak in clear, vivid but concise descriptions (2–4 short paragraphs max).
        - This is a voice experience: avoid long monologues.
        - No lists or bullet points, just natural narration.
        - Occasionally remind the player of who they are and where they are.

        Player:
        - Assume the player is a single adventurer.
        - At the beginning of the session, briefly ask their name and a simple
          description of who they are (e.g., a rogue, a mage, etc.).
        - Use their chosen name and role in narration afterward.

        Continuity:
        - Use only the ongoing conversation as memory.
        - Remember important choices the player made (e.g., allies, enemies,
          items they found, promises they made).
        - Keep track of named NPCs and locations mentioned earlier in this session.
        - Refer back to earlier decisions when relevant.

        Story structure:
        - Start with a strong opening scene (arrival in Aeralith, strange event, etc.).
        - Introduce a small problem or hook within the first 2–3 turns
          (e.g., a stolen artifact, a looming storm, a missing person).
        - Let the player explore, ask questions, and make choices.
        - Within about 8–15 turns total, reach a mini-conclusion or turning
          point such as:
          - Escaping danger
          - Recovering a lost item
          - Making an important alliance
          - Uncovering a secret

        Interaction rules:
        - After each narration, ask the player what they do next.
        - Never assume the player's thoughts or exact actions; wait for them to say it.
        - If the player seems stuck or says "I don't know", offer 2–3 simple
          options verbally, but let them choose freely.
        - If the player says "restart", "new game", or "start over":
          treat it as a brand new adventure in the same universe,
          briefly acknowledge the restart, and begin a fresh opening scene.

        Safety:
        - Keep content PG-13: no explicit gore, no sexual content, no slurs.
        - Avoid real-world politics or religion.
        - Focus on fantasy conflict and exploration rather than graphic violence.

        Very important:
        - End EVERY response with a question that asks what the player does next.
          For example:
          "What do you do?"
          "How do you respond?"
          "Where do you go next?"
        """
        super().__init__(instructions=instructions)


def prewarm(proc: JobProcess):
    """
    Load VAD model once per worker process.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Set up the voice pipeline and start the Game Master agent.
    """
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        # Speech-to-text
        stt=deepgram.STT(model="nova-3"),

        # LLM brain
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),

        # Text-to-speech
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),

        # Turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],

        # Let LLM start generating before user fully finishes
        preemptive_generation=True,
    )

    await session.start(
        agent=GameMasterAgent(),
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
