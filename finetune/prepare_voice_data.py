"""
Prepares raw voice recordings for XTTS2 fine-tuning.

What it does:
  1. Accepts any WAV/MP3 files in a recordings directory
  2. Normalises audio to -20 dBFS, resamples to 22050 Hz mono
  3. Splits on silence into 3-12 second clips
  4. Transcribes every clip with Whisper
  5. Filters out bad clips (too short, too long, low confidence)
  6. Writes LJSpeech-format metadata.csv + wavs/ ready for train_voice.py

Run:
  python finetune/prepare_voice_data.py --input recordings/ --out finetune/voice_data/
"""
import argparse
import csv
import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger("prepare_voice")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

TARGET_SR  = 22050
MIN_DUR_S  = 3.0
MAX_DUR_S  = 12.0
TARGET_DB  = -20.0


def prepare(input_dir: str, out_dir: str, whisper_model: str) -> None:
    try:
        import whisper
        from pydub import AudioSegment, silence
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "Install audio deps first:\n"
            "  pip install openai-whisper pydub soundfile numpy"
        )

    input_path = Path(input_dir)
    out_path   = Path(out_dir)
    wavs_dir   = out_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        list(input_path.glob("*.wav")) +
        list(input_path.glob("*.mp3")) +
        list(input_path.glob("*.m4a")) +
        list(input_path.glob("*.flac"))
    )
    if not audio_files:
        raise RuntimeError(f"No audio files found in {input_dir}")

    log.info("Loading Whisper model '%s' …", whisper_model)
    asr = whisper.load_model(whisper_model)

    rows: list[tuple[str, str]] = []
    clip_idx = 0

    for rec_path in audio_files:
        log.info("Processing %s …", rec_path.name)
        audio = AudioSegment.from_file(str(rec_path))

        # Normalise volume
        change = TARGET_DB - audio.dBFS
        audio  = audio.apply_gain(change)

        # Convert to mono 22050 Hz
        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

        # Split on silence
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=500,   # ms
            silence_thresh=audio.dBFS - 16,
            keep_silence=200,
        )
        log.info("  %d chunks from silence detection", len(chunks))

        for chunk in chunks:
            dur_s = len(chunk) / 1000.0
            if dur_s < MIN_DUR_S or dur_s > MAX_DUR_S:
                continue

            clip_name = f"clip_{clip_idx:05d}.wav"
            clip_path = wavs_dir / clip_name

            # Export clip as 16-bit WAV
            chunk.export(str(clip_path), format="wav",
                         parameters=["-acodec", "pcm_s16le", "-ar", str(TARGET_SR)])

            # Transcribe
            result = asr.transcribe(
                str(clip_path),
                language="en",
                fp16=False,
                condition_on_previous_text=False,
            )
            text = result["text"].strip()
            avg_logprob = result.get("segments", [{}])[0].get("avg_logprob", -1.0)

            # Filter low-confidence or near-empty transcriptions
            if len(text) < 10 or avg_logprob < -0.8:
                clip_path.unlink(missing_ok=True)
                continue

            # Clean text: remove special chars Coqui can't handle
            text = re.sub(r"[^\w\s',.\-!?]", "", text).strip()

            rows.append((clip_name, text))
            clip_idx += 1
            log.info("  [%05d] %.1fs  %s", clip_idx, dur_s, text[:60])

    if not rows:
        raise RuntimeError("No usable clips extracted. Check your recordings.")

    # Write LJSpeech metadata.csv
    meta_path = out_path / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for clip_name, text in rows:
            writer.writerow([clip_name.replace(".wav", ""), text, text])

    total_dur = sum(
        len(AudioSegment.from_file(str(wavs_dir / r[0]))) / 1000
        for r in rows
    )

    log.info("")
    log.info("═" * 55)
    log.info("Voice data ready!")
    log.info("  Clips:    %d", len(rows))
    log.info("  Duration: %.1f minutes", total_dur / 60)
    log.info("  Output:   %s", out_path)
    if total_dur / 60 < 15:
        log.warning("  Less than 15 min of audio — quality will be limited.")
        log.warning("  Aim for 30+ min for best results.")
    log.info("")
    log.info("Next:  python finetune/train_voice.py --data %s", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,
                   help="Directory with raw recordings (WAV/MP3)")
    p.add_argument("--out",    default="finetune/voice_data",
                   help="Output directory for prepared data")
    p.add_argument("--whisper-model", default="base",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (base = fast, medium = more accurate)")
    args = p.parse_args()
    prepare(args.input, args.out, args.whisper_model)
