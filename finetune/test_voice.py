"""
Quick sanity-check for a fine-tuned (or zero-shot) XTTS2 voice clone.

Usage:
  # Test the fine-tuned model:
  python finetune/test_voice.py --model models/xtts_finetuned

  # Zero-shot clone (no fine-tune, just a reference wav):
  python finetune/test_voice.py --reference path/to/reference.wav

Outputs one WAV per test sentence to --out_dir (default: /tmp/voice_test/).
"""
import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("test_voice")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

TEST_SENTENCES = [
    "Hey, what's up everyone — so glad you're here with me tonight!",
    "Honestly I'm obsessed with this app, like you have to download it.",
    "Oh my god, that's literally so funny, I can't even.",
    "Okay chat, hot take time — pineapple on pizza, yes or no?",
    "You guys are actually the best, I swear I look forward to this every night.",
]


def test(model_dir: str | None, reference_wav: str | None, out_dir: str,
         device: str) -> None:
    try:
        from TTS.api import TTS  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Install Coqui TTS:\n"
            "  pip install coqui-tts"
        )

    import torch

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_gpu = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not use_gpu:
        log.warning("CUDA not available — falling back to CPU")

    if model_dir:
        model_path = Path(model_dir)
        config_path = model_path / "config.json"
        vocab_path  = model_path / "vocab.json"
        ref_wav     = reference_wav or str(model_path / "reference.wav")

        if not config_path.exists():
            raise RuntimeError(f"No config.json in {model_dir}. Did training complete?")
        if not Path(ref_wav).exists():
            raise RuntimeError(
                f"Reference WAV not found: {ref_wav}\n"
                "Pass --reference path/to/clip.wav"
            )

        log.info("Loading fine-tuned XTTS2 from %s …", model_dir)
        tts = TTS(
            model_path=str(model_path),
            config_path=str(config_path),
            progress_bar=False,
        ).to("cuda" if use_gpu else "cpu")

    else:
        if not reference_wav:
            raise RuntimeError("Provide either --model or --reference")
        ref_wav = reference_wav
        if not Path(ref_wav).exists():
            raise RuntimeError(f"Reference WAV not found: {ref_wav}")

        log.info("Loading zero-shot XTTS2 (tts_models/multilingual/multi-dataset/xtts_v2) …")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                  progress_bar=False).to("cuda" if use_gpu else "cpu")

    log.info("Reference WAV: %s", ref_wav)
    log.info("Output dir:    %s", out_path)
    log.info("")

    for i, sentence in enumerate(TEST_SENTENCES):
        out_file = out_path / f"test_{i:02d}.wav"
        log.info("[%d/%d] %s", i + 1, len(TEST_SENTENCES), sentence[:70])
        tts.tts_to_file(
            text=sentence,
            speaker_wav=ref_wav,
            language="en",
            file_path=str(out_file),
        )
        log.info("       → %s", out_file)

    log.info("")
    log.info("═" * 55)
    log.info("Done! Listen to the files in:  %s", out_path)
    log.info("")
    log.info("If it sounds good, update .env:")
    log.info("  TTS_BACKEND=xtts_local")
    if model_dir:
        log.info("  XTTS_CHECKPOINT_DIR=%s", model_dir)
    log.info("  XTTS_REFERENCE_WAV=%s", ref_wav)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   help="Fine-tuned model directory (from train_voice.py)")
    p.add_argument("--reference",
                   help="Reference WAV for voice conditioning (zero-shot or override)")
    p.add_argument("--out_dir", default="/tmp/voice_test",
                   help="Directory to write output WAV files")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    if not args.model and not args.reference:
        p.error("Provide --model (fine-tuned) or --reference (zero-shot)")

    test(args.model, args.reference, args.out_dir, args.device)
