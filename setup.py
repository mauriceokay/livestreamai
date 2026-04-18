"""
One-shot setup script for the AI livestreamer pipeline.

What it does:
  1. Checks prerequisites (Python 3.10+, git, ffmpeg, CUDA)
  2. Creates required directories (checkpoints/, assets/)
  3. Clones and installs the chosen lip-sync model
  4. Downloads model checkpoints from Hugging Face / Google Drive
  5. Installs Python dependencies
  6. Copies .env.example → .env if it doesn't exist yet

Usage:
  python setup.py --model hallo2           # most realistic (diffusion, GPU)
  python setup.py --model emo              # EMO diffusion (GPU)
  python setup.py --model latsync         # fast & good (GPU, recommended default)
  python setup.py --model wav2lip         # CPU fallback
  python setup.py --model latsync --no-cuda  # force CPU
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint sources
# ─────────────────────────────────────────────────────────────────────────────

LATSYNC_HF_REPO = "chunyu-li/LatentSync"
LATSYNC_FILES = [
    # (hf_filename, local_dest)
    ("latentsync_unet.pt",      "checkpoints/latentsync_unet.pt"),
    ("whisper/tiny.pt",         "checkpoints/whisper/tiny.pt"),
]

# Wav2Lip weights (hosted on a public HuggingFace mirror — avoids gdown fragility)
WAV2LIP_HF_REPO = "numz/wav2lip_studio"
WAV2LIP_FILES = [
    ("Wav2Lip/wav2lip_gan.pth", "checkpoints/wav2lip_gan.pth"),
]

FACE_DET_URL = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
)
FACE_DET_DEST = "checkpoints/face_detection/s3fd.pth"

# EMO (HumanAIGC/EMO) — diffusion-based talking head
EMO_REPO = "https://github.com/HumanAIGC/EMO.git"
EMO_HF_REPO = "HumanAIGC/EMO"
EMO_FILES = [
    ("unet.pth",           "checkpoints/emo/unet.pth"),
    ("vae.pth",            "checkpoints/emo/vae.pth"),
    ("image_encoder.pth",  "checkpoints/emo/image_encoder.pth"),
]

# Hallo2 (fudan-generative-vision/hallo2) — high-quality long-form talking head
HALLO2_REPO = "https://github.com/fudan-generative-vision/hallo2.git"
HALLO2_HF_REPO = "fudan-generative-vision/hallo2"
HALLO2_FILES = [
    ("hallo2/net.pth",      "checkpoints/hallo2/net.pth"),
    ("motion_module.pth",   "checkpoints/hallo2/motion_module.pth"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        _die(f"Command failed: {' '.join(cmd)}")


def _die(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _step(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def _check_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _python() -> str:
    return sys.executable


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — prerequisites
# ─────────────────────────────────────────────────────────────────────────────

def check_prerequisites(require_cuda: bool) -> None:
    _step("Checking prerequisites")

    ver = sys.version_info
    if ver < (3, 10):
        _die(f"Python 3.10+ required (found {ver.major}.{ver.minor})")
    _ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")

    for tool in ("git", "ffmpeg"):
        if not _check_cmd(tool):
            _die(f"'{tool}' not found — please install it and re-run.")
        _ok(tool)

    if require_cuda:
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                print(
                    "\n  [WARN] CUDA not available. LatentSync will run on CPU "
                    "(very slow). Use --no-cuda to suppress this warning or "
                    "switch to --model wav2lip for faster CPU inference."
                )
            else:
                _ok(f"CUDA {torch.version.cuda} — GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("  [WARN] torch not yet installed — CUDA check skipped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — directory structure
# ─────────────────────────────────────────────────────────────────────────────

def create_dirs() -> None:
    _step("Creating directory structure")
    for d in ["checkpoints/whisper", "checkpoints/face_detection", "assets"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        _ok(d)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — clone lip-sync repo
# ─────────────────────────────────────────────────────────────────────────────

def clone_latsync() -> None:
    _step("Cloning LatentSync")
    if Path("LatentSync/.git").exists():
        _ok("LatentSync already cloned — skipping")
        return
    run(["git", "clone", "--depth=1",
         "https://github.com/bytedance/LatentSync.git", "LatentSync"])
    _ok("Cloned LatentSync")


def clone_wav2lip() -> None:
    _step("Cloning Wav2Lip")
    if Path("Wav2Lip/.git").exists():
        _ok("Wav2Lip already cloned — skipping")
        return
    run(["git", "clone", "--depth=1",
         "https://github.com/Rudrabha/Wav2Lip.git", "Wav2Lip"])
    _ok("Cloned Wav2Lip")


def clone_emo() -> None:
    _step("Cloning EMO (HumanAIGC)")
    if Path("EMO/.git").exists():
        _ok("EMO already cloned — skipping")
        return
    run(["git", "clone", "--depth=1", EMO_REPO, "EMO"])
    _ok("Cloned EMO")


def clone_hallo2() -> None:
    _step("Cloning Hallo2 (fudan-generative-vision)")
    if Path("hallo2/.git").exists():
        _ok("hallo2 already cloned — skipping")
        return
    run(["git", "clone", "--depth=1", HALLO2_REPO, "hallo2"])
    _ok("Cloned hallo2")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — download checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _hf_download(repo: str, filename: str, dest: str) -> None:
    dest_path = Path(dest)
    if dest_path.exists():
        _ok(f"{dest} already exists — skipping")
        return
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        run([_python(), "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download  # type: ignore

    print(f"  Downloading {filename} from {repo} …")
    local = hf_hub_download(repo_id=repo, filename=filename)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dest_path)
    _ok(f"Saved → {dest}")


def _url_download(url: str, dest: str) -> None:
    dest_path = Path(dest)
    if dest_path.exists():
        _ok(f"{dest} already exists — skipping")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url.split('/')[-1]} …")

    def _progress(count: int, block: int, total: int) -> None:
        pct = min(count * block / total * 100, 100)
        print(f"\r    {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()
    _ok(f"Saved → {dest}")


def download_latsync_checkpoints() -> None:
    _step("Downloading LatentSync checkpoints (Hugging Face)")
    for hf_file, dest in LATSYNC_FILES:
        _hf_download(LATSYNC_HF_REPO, hf_file, dest)


def download_wav2lip_checkpoints() -> None:
    _step("Downloading Wav2Lip checkpoints")
    for hf_file, dest in WAV2LIP_FILES:
        _hf_download(WAV2LIP_HF_REPO, hf_file, dest)
    _step("Downloading face detection model")
    _url_download(FACE_DET_URL, FACE_DET_DEST)
    # Copy face detection weight to where Wav2Lip expects it
    wav2lip_det = Path("Wav2Lip/face_detection/detection/sfd/s3fd.pth")
    if not wav2lip_det.exists() and Path(FACE_DET_DEST).exists():
        wav2lip_det.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FACE_DET_DEST, wav2lip_det)
        _ok(f"Copied face detection weights → {wav2lip_det}")


def download_emo_checkpoints() -> None:
    _step("Downloading EMO checkpoints (Hugging Face)")
    Path("checkpoints/emo").mkdir(parents=True, exist_ok=True)
    for hf_file, dest in EMO_FILES:
        _hf_download(EMO_HF_REPO, hf_file, dest)


def download_hallo2_checkpoints() -> None:
    _step("Downloading Hallo2 checkpoints (Hugging Face)")
    Path("checkpoints/hallo2").mkdir(parents=True, exist_ok=True)
    for hf_file, dest in HALLO2_FILES:
        _hf_download(HALLO2_HF_REPO, hf_file, dest)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5 — pip install
# ─────────────────────────────────────────────────────────────────────────────

def install_deps(model: str) -> None:
    _step("Installing Python dependencies")
    run([_python(), "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    if model == "latsync" and Path("LatentSync").exists():
        run([_python(), "-m", "pip", "install", "-q", "-e", "LatentSync/"])
        _ok("LatentSync package installed")
    elif model == "wav2lip" and Path("Wav2Lip").exists():
        wav2lip_req = Path("Wav2Lip/requirements.txt")
        if wav2lip_req.exists():
            run([_python(), "-m", "pip", "install", "-q", "-r", str(wav2lip_req)])
        _ok("Wav2Lip dependencies installed")
    elif model == "emo" and Path("EMO").exists():
        emo_req = Path("EMO/requirements.txt")
        if emo_req.exists():
            run([_python(), "-m", "pip", "install", "-q", "-r", str(emo_req)])
        _ok("EMO dependencies installed")
    elif model == "hallo2" and Path("hallo2").exists():
        hallo2_req = Path("hallo2/requirements.txt")
        if hallo2_req.exists():
            run([_python(), "-m", "pip", "install", "-q", "-r", str(hallo2_req)])
        _ok("Hallo2 dependencies installed")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 6 — .env file
# ─────────────────────────────────────────────────────────────────────────────

def setup_env(model: str, device: str) -> None:
    _step("Setting up .env")
    if Path(".env").exists():
        _ok(".env already exists — not overwriting")
        return
    shutil.copy(".env.example", ".env")

    # Patch the model/device defaults into the new .env
    content = Path(".env").read_text()
    content = content.replace(
        "LIPSYNC_MODEL=latsync",
        f"LIPSYNC_MODEL={model}",
    )
    content = content.replace(
        "LIPSYNC_DEVICE=cuda",
        f"LIPSYNC_DEVICE={device}",
    )
    Path(".env").write_text(content)
    _ok(".env created from .env.example")


# ─────────────────────────────────────────────────────────────────────────────
#  Final checklist
# ─────────────────────────────────────────────────────────────────────────────

def print_checklist(model: str) -> None:
    ckpt_path = {
        "latsync": "checkpoints/latentsync_unet.pt",
        "wav2lip": "checkpoints/wav2lip_gan.pth",
        "emo":     "checkpoints/emo/unet.pth",
        "hallo2":  "checkpoints/hallo2/net.pth",
    }.get(model, "checkpoints/latentsync_unet.pt")
    face_ok = Path("assets/face.jpg").exists()
    ckpt_ok = Path(ckpt_path).exists()
    env_ok = Path(".env").exists()

    print(f"\n{'═'*60}")
    print("  Setup complete! Checklist:")
    print(f"{'═'*60}")
    print(f"  {'[x]' if ckpt_ok  else '[ ]'} Checkpoint:  {ckpt_path}")
    print(f"  {'[x]' if face_ok  else '[ ]'} Face image:  assets/face.jpg")
    print(f"  {'[x]' if env_ok   else '[ ]'} Env file:    .env")
    print()

    if not face_ok:
        print("  ACTION NEEDED:")
        print("    Put a portrait photo of your AI persona at:  assets/face.jpg")
        print("    • Use a front-facing, well-lit headshot (at least 512×512 px)")
        print("    • JPG or PNG both work — rename to face.jpg")
        print()

    if env_ok:
        print("  Fill in your API keys in .env, then run:")
        print("    python pipeline.py")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up the AI livestreamer pipeline."
    )
    parser.add_argument(
        "--model",
        choices=["latsync", "wav2lip", "emo", "hallo2"],
        default="latsync",
        help="hallo2/emo = most realistic (diffusion, A100/4090 required); "
             "latsync = fast & good; wav2lip = CPU fallback",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Skip CUDA check / use CPU (slower but works without a GPU)",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Skip checkpoint downloads (useful if you already have them)",
    )
    args = parser.parse_args()

    device = "cpu" if args.no_cuda else "cuda"
    require_cuda = not args.no_cuda and args.model == "latsync"

    print("\n  AI Livestreamer — setup")
    print(f"  Model: {args.model}   Device: {device}\n")

    check_prerequisites(require_cuda)
    create_dirs()

    if args.model == "latsync":
        clone_latsync()
        if not args.skip_checkpoints:
            download_latsync_checkpoints()
    elif args.model == "wav2lip":
        clone_wav2lip()
        if not args.skip_checkpoints:
            download_wav2lip_checkpoints()
    elif args.model == "emo":
        clone_emo()
        if not args.skip_checkpoints:
            download_emo_checkpoints()
    elif args.model == "hallo2":
        clone_hallo2()
        if not args.skip_checkpoints:
            download_hallo2_checkpoints()

    install_deps(args.model)
    setup_env(args.model, device)
    print_checklist(args.model)


if __name__ == "__main__":
    main()
