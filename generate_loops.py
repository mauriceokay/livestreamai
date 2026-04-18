"""
One-time loop generation script.

Generates the animated video loop library from a single face image.
Run this once after setup.py, before starting the stream.

What it creates (in assets/loops/):
  idle_0.mp4 … idle_2.mp4          — face at rest, subtle movement
  talking_0.mp4 … talking_4.mp4    — natural mouth movement
  react_happy_0.mp4 …              — upbeat / excited expression
  react_gift_0.mp4 …               — grateful / delighted expression
  look_away_0.mp4 …                — glances off screen then back
  drink_water_0.mp4 …              — takes a sip, natural pause
  laugh_0.mp4 …                    — genuine laugh reaction
  thinking_0.mp4 …                 — thoughtful pause, slight head tilt

Each clip is 3–4 seconds and loops seamlessly in the stream.

Usage:
  python generate_loops.py                    # default (latsync)
  python generate_loops.py --model hallo2     # most realistic
  python generate_loops.py --model emo        # diffusion, very natural
  python generate_loops.py --model wav2lip    # CPU fallback
  python generate_loops.py --only look_away   # regenerate one category
"""
import argparse
import asyncio
import io
import logging
import os
import shutil
import wave
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("generate_loops")


# ─────────────────────────────────────────────────────────────────────────────
#  Script lines for each loop category
#  These are fed to TTS → LipSync to create the pre-rendered clips.
#  The animation is what matters — exact phrasing is never heard on stream.
# ─────────────────────────────────────────────────────────────────────────────

LOOP_SCRIPTS: dict[str, list[str]] = {
    "idle": [
        "Mm-hmm.",
        "Yeah.",
        "Okay.",
    ],
    "talking": [
        "Oh that's so interesting, tell me more about that.",
        "Yeah I totally get what you mean, that's such a good point.",
        "Honestly same, I feel like everyone goes through that.",
        "Oh wow, okay, I did not expect that at all.",
        "Right, right, yeah that makes a lot of sense actually.",
    ],
    "react_happy": [
        "Oh my gosh, that's so sweet, thank you so much!",
        "Aww you're literally the best, I love this community!",
        "No way, that just made my whole day, seriously!",
    ],
    "react_gift": [
        "Oh wow, thank you so much, that means the world to me!",
        "Oh my gosh, you didn't have to do that, thank you!",
        "That is so generous, genuinely, thank you!",
    ],

    # ── Behavior clips: make the stream feel human ─────────────────────────
    "look_away": [
        # Brief glance off-screen then back — looks like reacting to notifications
        "Oh wait, one sec.",
        "Hold on, let me just—",
        "Sorry, okay I'm back.",
    ],
    "drink_water": [
        # Short pause with minimal speech — animation carries it
        "Mm.",
        "Okay.",
        "Mmm, yeah.",
    ],
    "laugh": [
        "Oh my god, that's so funny, I literally cannot.",
        "Stop, I'm actually dead right now, that got me.",
        "No wait, that actually made me laugh, I wasn't ready.",
    ],
    "thinking": [
        "Hmm, okay, let me actually think about that for a sec.",
        "I mean... yeah, you know what, that's actually a really good point.",
        "That's a good question, honestly I've been thinking about that too.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(model: str, device: str, only: str | None) -> None:
    from config import Config
    from tts import build_tts
    from lipsync import LipSync

    cfg = Config().load_from_env()
    cfg.lipsync.model = model
    cfg.lipsync.device = device

    out_dir = Path("assets/loops")
    out_dir.mkdir(parents=True, exist_ok=True)

    face = cfg.lipsync.base_face_image
    if not Path(face).exists():
        log.error("Face image not found: %s", face)
        log.error("Put a portrait photo at %s and re-run.", face)
        return

    tts = build_tts(cfg.tts)
    lipsync = LipSync(cfg.lipsync)

    categories = [only] if only else list(LOOP_SCRIPTS.keys())

    for category in categories:
        scripts = LOOP_SCRIPTS[category]
        log.info("Generating '%s' loops (%d clips) …", category, len(scripts))

        for idx, line in enumerate(scripts):
            dest = out_dir / f"{category}_{idx}.mp4"
            if dest.exists():
                log.info("  [skip] %s already exists", dest.name)
                continue

            log.info("  [%d/%d] TTS: %r", idx + 1, len(scripts), line[:50])
            try:
                wav = await tts.synthesize(line)
                log.info("  [%d/%d] LipSync (%s) …", idx + 1, len(scripts), model)
                mp4 = await lipsync.generate_chunk(wav)
                dest.write_bytes(mp4)
                log.info("  [%d/%d] Saved → %s", idx + 1, len(scripts), dest)
            except Exception as exc:
                log.error("  [%d/%d] FAILED: %s", idx + 1, len(scripts), exc)

        if category == "idle":
            _make_ken_burns_fallback(face, out_dir)

    log.info("")
    log.info("Loop library complete → %s", out_dir)
    _print_summary(out_dir)


def _make_ken_burns_fallback(face_path: str, out_dir: Path) -> None:
    """
    Subtle zoom-in/out idle clip from the static face image using ffmpeg's
    zoompan filter.  No ML required — runs in under a second.
    """
    dest = out_dir / "idle_static.mp4"
    if dest.exists():
        return

    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-loop", "1",
        "-i", face_path,
        "-vf", (
            "scale=8000:-1,"
            "zoompan=z='if(lte(zoom,1.0),1.05,max(1.001,zoom-0.0015))':"
            "d=75:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "s=720x1280,"
            "fps=25"
        ),
        "-t", "3",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        log.info("  Created Ken Burns fallback → %s", dest.name)
    else:
        log.warning("  Ken Burns fallback failed (non-critical): %s",
                    result.stderr.decode()[:200])


def _print_summary(out_dir: Path) -> None:
    clips = list(out_dir.glob("*.mp4"))
    by_cat: dict[str, list[str]] = {}
    for c in sorted(clips):
        cat = c.stem.rsplit("_", 1)[0]
        by_cat.setdefault(cat, []).append(c.name)

    core_cats    = {"idle", "talking", "react_happy", "react_gift"}
    behavior_cats = {"look_away", "drink_water", "laugh", "thinking"}

    print(f"\n{'═'*55}")
    print("  Loop library summary")
    print(f"{'═'*55}")
    for cat, names in sorted(by_cat.items()):
        tag = "[behavior]" if cat in behavior_cats else "[core]    "
        status = "[ok]" if names else "[MISSING] "
        print(f"  {status}  {tag}  {cat}: {len(names)} clip(s)")
    print()
    missing_core = core_cats - set(by_cat.keys())
    if missing_core:
        print(f"  WARNING: Missing core categories: {', '.join(sorted(missing_core))}")
        print(f"  Run: python generate_loops.py")
    else:
        print("  Next step:  python pipeline.py")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pre-rendered video loop library."
    )
    parser.add_argument(
        "--model",
        choices=["latsync", "wav2lip", "emo", "hallo2"],
        default=os.getenv("LIPSYNC_MODEL", "latsync"),
        help="hallo2/emo = most realistic (diffusion, needs GPU); "
             "latsync = fast & good; wav2lip = CPU fallback",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("LIPSYNC_DEVICE", "cuda"),
    )
    parser.add_argument(
        "--only",
        choices=list(LOOP_SCRIPTS.keys()),
        default=None,
        help="Regenerate a single category (e.g. --only look_away)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model, args.device, args.only))
