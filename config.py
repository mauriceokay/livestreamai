import os
from dataclasses import dataclass, field


@dataclass
class PersonaConfig:
    name: str = "Maya"
    age: int = 24
    personality: str = "friendly, energetic, loves discovering cool apps and connecting people"
    app_name: str = "MyApp"
    app_description: str = "a new chat app that makes it easy to connect with friends and meet new people"
    voice_description: str = "warm, upbeat, conversational"


@dataclass
class StreamDestination:
    name: str           # "tiktok" | "youtube" | "facebook" | "instagram"
    rtmp_url: str
    stream_key: str
    enabled: bool = True

    @property
    def target(self) -> str:
        return f"{self.rtmp_url.rstrip('/')}/{self.stream_key}"

    @property
    def safe_target(self) -> str:
        """URL without the stream key — safe to log."""
        return self.rtmp_url


# Default RTMP base URLs per platform
PLATFORM_RTMP: dict[str, str] = {
    "tiktok":    "rtmps://live-push.tiktok.com/live/",
    "youtube":   "rtmp://a.rtmp.youtube.com/live2/",
    "facebook":  "rtmps://live-api-s.facebook.com:443/rtmp/",
    "instagram": "rtmps://edgetee-upload-lax5-1.instagram.com:443/rtmp/",
}


@dataclass
class StreamConfig:
    destinations: list[StreamDestination] = field(default_factory=list)
    width: int = 720
    height: int = 1280          # portrait 9:16 for TikTok / Shorts / Reels
    fps: int = 25
    video_bitrate: str = "2500k"
    audio_bitrate: str = "128k"
    audio_sample_rate: int = 44100

    # AI disclosure overlay (required by TikTok / Meta ToS)
    show_disclosure: bool = True
    disclosure_text: str = "AI Generated"

    # Optional lower-third app promo banner
    show_promo_banner: bool = False
    promo_text: str = ""

    # Auto-reconnect on stream failure
    reconnect_delay_s: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class TTSConfig:
    # backend: "elevenlabs" | "kokoro" | "coqui"
    backend: str = "elevenlabs"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_turbo_v2_5"
    kokoro_voice: str = "af_heart"
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    max_retries: int = 3
    retry_base_delay_s: float = 1.0


@dataclass
class LLMConfig:
    # backend: "claude" | "ollama" | "local"
    # "local" = your fine-tuned model served by Ollama after finetune/export_to_ollama.py
    backend: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"
    ollama_model: str = "llama3.2"     # set to "maya" after fine-tune export
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_s: float = 15.0
    max_tokens: int = 120
    temperature: float = 0.85
    max_retries: int = 3
    retry_base_delay_s: float = 1.0


@dataclass
class LipSyncConfig:
    # model: "latsync" | "wav2lip"
    model: str = "latsync"
    base_face_image: str = "assets/face.jpg"
    latsync_checkpoint: str = "checkpoints/latentsync_unet.pt"
    wav2lip_checkpoint: str = "checkpoints/wav2lip_gan.pth"
    device: str = "cuda"
    fps: int = 25
    face_det_batch_size: int = 4
    wav2lip_batch_size: int = 128
    loops_dir: str = "assets/loops"
    subprocess_timeout_s: int = 120


@dataclass
class ModeratorConfig:
    enabled: bool = True
    max_msg_per_minute: int = 3      # per user
    max_msg_length: int = 200
    block_spam: bool = True
    spam_window: int = 5             # compare last N messages for duplicates


@dataclass
class Config:
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    lipsync: LipSyncConfig = field(default_factory=LipSyncConfig)
    moderator: ModeratorConfig = field(default_factory=ModeratorConfig)

    tiktok_username: str = ""
    idle_interval_s: float = 12.0
    max_queue_depth: int = 4

    def load_from_env(self) -> "Config":
        # ── Persona ──────────────────────────────────────────────────────────
        if v := os.getenv("PERSONA_NAME"):
            self.persona.name = v
        if v := os.getenv("APP_NAME"):
            self.persona.app_name = v
        if v := os.getenv("APP_DESCRIPTION"):
            self.persona.app_description = v

        # ── TikTok chat reading ───────────────────────────────────────────────
        if v := os.getenv("TIKTOK_USERNAME"):
            self.tiktok_username = v

        # ── Stream destinations (per-platform env vars) ───────────────────────
        for platform in ("tiktok", "youtube", "facebook", "instagram"):
            key = os.getenv(f"{platform.upper()}_STREAM_KEY")
            if not key:
                continue
            url = (
                os.getenv(f"{platform.upper()}_RTMP_URL")
                or PLATFORM_RTMP.get(platform, "")
            )
            self.stream.destinations.append(
                StreamDestination(name=platform, rtmp_url=url, stream_key=key)
            )

        # Backward-compat: single RTMP_URL + STREAM_KEY
        if not self.stream.destinations:
            url = os.getenv("RTMP_URL", "")
            key = os.getenv("STREAM_KEY", "")
            if url and key:
                platform = os.getenv("PLATFORM", "tiktok")
                self.stream.destinations.append(
                    StreamDestination(name=platform, rtmp_url=url, stream_key=key)
                )

        # ── Overlays ─────────────────────────────────────────────────────────
        if v := os.getenv("SHOW_DISCLOSURE"):
            self.stream.show_disclosure = v.lower() not in ("0", "false", "no")
        if v := os.getenv("DISCLOSURE_TEXT"):
            self.stream.disclosure_text = v
        if v := os.getenv("PROMO_TEXT"):
            self.stream.promo_text = v
            self.stream.show_promo_banner = True
        elif self.persona.app_name and self.persona.app_name != "MyApp":
            self.stream.promo_text = f"Download {self.persona.app_name} — link in bio!"
            self.stream.show_promo_banner = True

        # ── LLM ──────────────────────────────────────────────────────────────
        if v := os.getenv("LLM_BACKEND"):
            self.llm.backend = v
        if v := os.getenv("ANTHROPIC_API_KEY"):
            self.llm.anthropic_api_key = v
        if v := os.getenv("OLLAMA_MODEL"):
            self.llm.ollama_model = v

        # ── TTS ──────────────────────────────────────────────────────────────
        if v := os.getenv("TTS_BACKEND"):
            self.tts.backend = v
        if v := os.getenv("ELEVENLABS_API_KEY"):
            self.tts.elevenlabs_api_key = v
        if v := os.getenv("ELEVENLABS_VOICE_ID"):
            self.tts.elevenlabs_voice_id = v

        # ── Lip-sync ─────────────────────────────────────────────────────────
        if v := os.getenv("LIPSYNC_MODEL"):
            self.lipsync.model = v
        if v := os.getenv("LIPSYNC_DEVICE"):
            self.lipsync.device = v

        return self
