"""
Fine-tunes XTTS2 on your character's voice recordings.

XTTS2 (Coqui) is a multilingual TTS model that clones voices extremely
well. Fine-tuning locks the model to your specific character's voice,
giving much better consistency than zero-shot cloning.

Hardware:
  Minimum:  RTX 3080 (10 GB VRAM)  — use --batch 8
  Good:     RTX 3090 / 4090 (24 GB) — use --batch 16
  Ideal:    A100 (40 GB)             — use --batch 32

Cloud cost: ~$10–30 on RunPod/Vast.ai (RTX 4090 ~$0.35/hr, takes ~2-4 hrs)

Run:
  python finetune/train_voice.py --data finetune/voice_data/
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger("train_voice")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT  = PROJECT_ROOT / "models" / "xtts_finetuned"
SPEAKER_NAME = "maya"


def train(data_dir: str, out_dir: str, epochs: int, batch_size: int) -> None:
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import (
            GPTTrainer, GPTTrainerConfig,
        )
        from TTS.tts.models.xtts import XttsAudioConfig
    except ImportError:
        raise RuntimeError(
            "Install Coqui TTS trainer:\n"
            "  pip install coqui-tts trainer\n"
            "  or: pip install -r finetune/requirements_train.txt"
        )

    data_path = Path(data_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Verify data
    meta_csv  = data_path / "metadata.csv"
    wavs_dir  = data_path / "wavs"
    if not meta_csv.exists():
        raise RuntimeError(
            f"metadata.csv not found in {data_dir}.\n"
            "Run: python finetune/prepare_voice_data.py first."
        )

    # Count clips
    n_clips = sum(1 for _ in open(meta_csv)) if meta_csv.exists() else 0
    log.info("Dataset: %d clips in %s", n_clips, data_dir)

    # ── Dataset config ────────────────────────────────────────────────────
    dataset_cfg = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=str(meta_csv),
        path=str(data_path),
        language="en",
    )

    # ── Audio config ──────────────────────────────────────────────────────
    audio_cfg = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # ── Model config ──────────────────────────────────────────────────────
    model_cfg = GPTTrainerConfig(
        epochs=epochs,
        output_path=str(out_path),
        model_args={
            "kv_cache": True,
        },
        run_name="xtts_maya",
        project_name="ai_streamer_voice",
        run_description="Maya voice clone",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_cfg,
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        target_loss="loss",
        print_eval=False,
        use_phonemes=False,
        phonemizer_backend="espeak",
        phoneme_language="en-us",
        compute_input_seq_cache=True,
        datasets=[dataset_cfg],
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[
            {
                "text": "Hey, what's up everyone, so glad you're here with me tonight!",
                "speaker_wav": str(wavs_dir / _first_wav(wavs_dir)),
                "language": "en",
            },
            {
                "text": "Honestly I'm obsessed with this app, like you have to download it.",
                "speaker_wav": str(wavs_dir / _first_wav(wavs_dir)),
                "language": "en",
            },
        ],
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    model = GPTTrainer.init_from_config(model_cfg)

    train_samples, eval_samples = load_tts_samples(
        dataset_cfg,
        eval_split=True,
        eval_split_max_size=model_cfg.eval_split_max_size,
        eval_split_size=model_cfg.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=1,
        ),
        model_cfg,
        output_path=str(out_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    log.info("Starting XTTS2 fine-tune — output: %s", out_path)
    trainer.fit()

    # Copy a reference clip for zero-shot inference conditioning
    ref_src = wavs_dir / _first_wav(wavs_dir)
    ref_dst = out_path / "reference.wav"
    shutil.copy2(str(ref_src), str(ref_dst))

    log.info("")
    log.info("═" * 55)
    log.info("Voice fine-tune complete!")
    log.info("  Model:     %s", out_path)
    log.info("  Reference: %s", ref_dst)
    log.info("")
    log.info("Test it:")
    log.info("  python finetune/test_voice.py --model %s", out_path)
    log.info("")
    log.info("Then in .env:")
    log.info("  TTS_BACKEND=xtts_local")
    log.info("  XTTS_CHECKPOINT_DIR=%s", out_path)


def _first_wav(wavs_dir: Path) -> str:
    wavs = sorted(wavs_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No WAV files found in {wavs_dir}")
    return wavs[0].name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True,
                   help="Prepared voice data directory (from prepare_voice_data.py)")
    p.add_argument("--out",    default=str(DEFAULT_OUT))
    p.add_argument("--epochs", type=int, default=6,
                   help="Training epochs (default 6, more = better but slower)")
    p.add_argument("--batch",  type=int, default=16,
                   help="Batch size (8=RTX 3080, 16=RTX 4090, 32=A100)")
    args = p.parse_args()
    train(args.data, args.out, args.epochs, args.batch)
