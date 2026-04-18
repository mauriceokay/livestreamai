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
class StreamConfig:
    # platform: "tiktok" | "youtube"
    platform: str = "tiktok"
    rtmp_url: str = ""        # TikTok: rtmps://live-push.tiktok.com/live/
                              # YouTube: rtmp://a.rtmp.youtube.com/live2
    stream_key: str = ""
    width: int = 720
    height: int = 1280        # portrait (9:16) for TikTok / YouTube Shorts
    fps: int = 25
    video_bitrate: str = "2500k"
    audio_bitrate: str = "128k"
    audio_sample_rate: int = 44100

    # AI disclosure overlay (required by TikTok ToS)
    show_disclosure: bool = True
    disclosure_text: str = "AI Generated"

    # Optional lower-third app promo banner
    show_promo_banner: bool = True
    promo_text: str = ""      # e.g. "Download MyApp — link in bio!"


@dataclass
class TTSConfig:
    # backend: "elevenlabs" | "kokoro" | "coqui"
    backend: str = "elevenlabs"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel — replace with your clone
    elevenlabs_model: str = "eleven_turbo_v2_5"          # lowest-latency ElevenLabs model
    kokoro_voice: str = "af_heart"
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"


@dataclass
class LLMConfig:
    # backend: "claude" | "ollama"
    backend: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"      # fastest Claude for real-time
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    max_tokens: int = 120         # short responses = natural live-stream feel
    temperature: float = 0.85


@dataclass
class LipSyncConfig:
    # model: "latsync" | "wav2lip"
    model: str = "latsync"
    base_face_image: str = "assets/face.jpg"
    latsync_checkpoint: str = "checkpoints/latentsync_unet.pt"
    wav2lip_checkpoint: str = "checkpoints/wav2lip_gan.pth"
    device: str = "cuda"          # "cuda" or "cpu"
    fps: int = 25
    face_det_batch_size: int = 4
    wav2lip_batch_size: int = 128
    loops_dir: str = "assets/loops"


@dataclass
class Config:
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    lipsync: LipSyncConfig = field(default_factory=LipSyncConfig)

    tiktok_username: str = ""
    idle_interval_s: float = 12.0     # seconds between unprompted remarks
    max_queue_depth: int = 4          # drop oldest chunk if pipeline falls behind

    def load_from_env(self) -> "Config":
        if v := os.getenv("PLATFORM"):
            self.stream.platform = v
        if v := os.getenv("TIKTOK_USERNAME"):
            self.tiktok_username = v
        if v := os.getenv("RTMP_URL"):
            self.stream.rtmp_url = v
        if v := os.getenv("STREAM_KEY"):
            self.stream.stream_key = v
        if v := os.getenv("SHOW_DISCLOSURE"):
            self.stream.show_disclosure = v.lower() not in ("0", "false", "no")
        if v := os.getenv("DISCLOSURE_TEXT"):
            self.stream.disclosure_text = v
        if v := os.getenv("PROMO_TEXT"):
            self.stream.promo_text = v
            self.stream.show_promo_banner = bool(v)
        if v := os.getenv("ELEVENLABS_API_KEY"):
            self.tts.elevenlabs_api_key = v
        if v := os.getenv("ELEVENLABS_VOICE_ID"):
            self.tts.elevenlabs_voice_id = v
        if v := os.getenv("ANTHROPIC_API_KEY"):
            self.llm.anthropic_api_key = v
        if v := os.getenv("TTS_BACKEND"):
            self.tts.backend = v
        if v := os.getenv("LLM_BACKEND"):
            self.llm.backend = v
        if v := os.getenv("LIPSYNC_MODEL"):
            self.lipsync.model = v
        if v := os.getenv("LIPSYNC_DEVICE"):
            self.lipsync.device = v
        if v := os.getenv("PERSONA_NAME"):
            self.persona.name = v
        if v := os.getenv("APP_NAME"):
            self.persona.app_name = v
            if not os.getenv("PROMO_TEXT"):
                self.stream.promo_text = f"Download {v} — link in bio!"
                self.stream.show_promo_banner = True
        if v := os.getenv("APP_DESCRIPTION"):
            self.persona.app_description = v
        return self
