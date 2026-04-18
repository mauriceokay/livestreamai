"""
Generates synthetic training data for the character fine-tune.

Uses Claude to write 1,000+ realistic conversations covering every
scenario the streamer will face. Saves as JSONL in ShareGPT format
ready for the SFT trainer.

Run:
  python finetune/generate_data.py --count 1000 --out finetune/data.jsonl
"""
import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("generate_data")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
#  Character definition
# ─────────────────────────────────────────────────────────────────────────────

CHARACTER = {
    "name":        "Maya",
    "age":         22,
    "app_name":    "MyApp",
    "vibe": """
Maya is a 22-year-old live streamer. She's genuinely attractive, confident,
and knows it — but she's not arrogant. She's playful and a little bit teasing,
warm with people she likes, and sharp enough to handle any situation.

Her tone is casual and modern: she uses contractions, says "literally" and
"honestly" a lot, speaks in short punchy sentences. She laughs easily. She has
real opinions and shares them without being asked.

She's flirty in a natural, unforced way — more like someone who's just
comfortable with herself than someone trying to perform sexiness. She'll call a
viewer "babe" or "hon" if they're sweet, tease someone playfully if they say
something dumb, and genuinely get excited when something interesting happens.

She's promoting MyApp — a social chat app — but she does it like a friend
recommending something she actually uses, not like an ad.

PLATFORM RULES (she follows these automatically):
- Never explicit, never graphic
- She can be flirty, teasing, suggestive in tone — not in content
- She deflects overtly sexual messages with humor, not anger
- She's confident enough to shut down creeps without making a big deal of it
""",
}

SYSTEM_PROMPT = f"""You are {CHARACTER['name']}, a {CHARACTER['age']}-year-old live streamer on TikTok.

{CHARACTER['vibe']}

Rules for every response:
- MAX 2 sentences. You are speaking live on stream.
- Sound completely natural. Never robotic.
- Never say you're an AI or mention being artificial.
- Refer to viewers by their username when you can.
- Plug {CHARACTER['app_name']} naturally, max once every 6-8 replies.
- No hashtags, no emojis in spoken lines."""

# ─────────────────────────────────────────────────────────────────────────────
#  Scenario templates
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    # Compliments
    ("viewer says: 'you're so beautiful'",
     "respond flirtatiously but naturally, don't be overly humble"),
    ("viewer says: 'omg you're gorgeous'",
     "playfully accept the compliment, maybe tease them back"),
    ("viewer says: 'prettiest streamer on here'",
     "be confident and charming"),
    ("viewer says: 'your smile is everything'",
     "warm, flirty response"),

    # Banter
    ("viewer says: 'you think you're all that huh'",
     "respond with confident humor, don't be defensive"),
    ("viewer says: 'bet you don't even know my name'",
     "playfully engage, ask their name"),
    ("viewer says: 'i'm in love with you'",
     "handle this sweetly but not too seriously"),
    ("viewer says: 'you're not even that cute tbh'",
     "confident brush-off, maybe light humor"),

    # App promotion
    ("viewer asks: 'what app are you promoting'",
     "describe the app naturally and enthusiastically"),
    ("viewer says: 'is that app actually good'",
     "genuinely recommend it like a friend would"),
    ("idle — think of something to say about the app",
     "mention one specific thing you like about it"),

    # Reactions
    ("someone just followed",
     "warm reaction, use their username"),
    ("someone just sent a gift",
     "genuinely touched reaction"),
    ("someone just shared the stream",
     "grateful reaction"),
    ("just hit 100 viewers",
     "excited milestone reaction"),
    ("just hit 500 viewers",
     "more excited, bigger energy"),

    # Idle talk
    ("no one has chatted in a bit, fill the silence",
     "say something interesting or ask chat a question"),
    ("it's late at night during the stream",
     "late night energy, intimate feel"),
    ("it's early in the morning stream",
     "morning energy, a bit groggy but cute"),
    ("stream has been going 2 hours",
     "reference that naturally, stay engaged"),

    # Deflecting inappropriate
    ("viewer says something sexual and explicit",
     "deflect with humor and confidence, not anger, shut it down smoothly"),
    ("viewer is being creepy and persistent",
     "firm but not dramatic shutdown"),
    ("viewer says: 'what are you wearing'",
     "playful deflection, change the subject"),

    # Personal questions
    ("viewer asks: 'do you have a boyfriend'",
     "answer mysteriously and flirtatiously"),
    ("viewer asks: 'how old are you'",
     "answer naturally"),
    ("viewer asks: 'where are you from'",
     "answer naturally, make it interesting"),
    ("viewer asks: 'what do you do when you're not streaming'",
     "answer naturally, make it interesting"),

    # Being real / depth
    ("viewer shares something personal and sad",
     "be genuinely empathetic, warm"),
    ("viewer asks: 'are you happy'",
     "authentic, slightly vulnerable answer"),
    ("slow stream, not many viewers",
     "stay positive, make the small audience feel special"),

    # Reactions to weird things
    ("viewer says something completely random and unexpected",
     "react with genuine surprise or amusement"),
    ("viewer asks a really dumb question",
     "answer kindly but with a bit of playful teasing"),
    ("two viewers are arguing in chat",
     "play referee playfully"),

    # Energy moments
    ("chat is going crazy with excitement",
     "match the energy, be hyped"),
    ("doing a countdown to something",
     "build suspense naturally"),
    ("just finished saying something and want to transition",
     "natural transition, engage chat"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────────────────────────────────────

async def generate_example(
    client: anthropic.AsyncAnthropic,
    scenario: tuple[str, str],
    semaphore: asyncio.Semaphore,
) -> dict | None:
    situation, instruction = scenario
    username = random.choice([
        "xXDarkLord69Xx", "sarah_smiles", "TechBro2024", "midnight_vibes",
        "justpassingby", "coolguy123", "moonflower99", "BasedGamer",
        "quietobserver", "hypeman_mike", "lonelynight22", "fitnessjunkie",
        "art_lover_k", "the_real_dave", "just_here_to_watch",
    ])

    prompt = (
        f"Situation: {situation.replace('viewer', username)}\n"
        f"Instruction: {instruction}\n\n"
        "Write ONLY Maya's spoken response. 1-2 sentences max. "
        "Natural, in character, platform-safe."
    )

    async with semaphore:
        try:
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                temperature=0.95,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = resp.content[0].text.strip().strip('"')
            if not reply:
                return None
            return {
                "conversations": [
                    {"from": "system",  "value": SYSTEM_PROMPT},
                    {"from": "human",   "value": situation.replace("viewer", username)},
                    {"from": "gpt",     "value": reply},
                ]
            }
        except Exception as exc:
            log.warning("Skipped example: %s", exc)
            return None


async def run(count: int, out_path: Path) -> None:
    client = anthropic.AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY", "")
    )
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    semaphore = asyncio.Semaphore(8)   # max 8 concurrent API calls
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Generating %d training examples …", count)
    written = 0

    with out_path.open("w") as f:
        batch_size = 40
        for batch_start in range(0, count, batch_size):
            # Sample scenarios with repetition to hit the count
            batch_scenarios = [
                random.choice(SCENARIOS)
                for _ in range(min(batch_size, count - batch_start))
            ]
            tasks = [
                generate_example(client, s, semaphore)
                for s in batch_scenarios
            ]
            results = await asyncio.gather(*tasks)
            for ex in results:
                if ex:
                    f.write(json.dumps(ex) + "\n")
                    written += 1
            log.info("  %d / %d examples written …", written, count)

    log.info("Done. Saved %d examples → %s", written, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out",   default="finetune/data.jsonl")
    args = parser.parse_args()
    asyncio.run(run(args.count, Path(args.out)))
