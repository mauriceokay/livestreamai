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
    rtmp_url: str = ""       # e.g. rtmps://live-push.tiktok.com/live/
    stream_key: str = ""
    width: int = 720
    height: int = 1280       # portrait for TikTok
    fps: int = 25
    video_bitrate: str = "2500k"
    audio_bitrate: str = "128k"
    audio_sample_rate: int = 44100


@dataclass
class TTSConfig:
    # backend: "elevenlabs" | "kokoro" | "coqui"
    backend: str = "elevenlabs"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel — change to your clone
    elevenlabs_model: str = "eleven_turbo_v2_5"         # lowest-latency model
    kokoro_voice: str = "af_heart"                      # Kokoro voice name
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    chunk_silence_ms: int = 50   # padding silence appended to each chunk


@dataclass
class LLMConfig:
    # backend: "claude" | "ollama"
    backend: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"     # fastest Claude for real-time
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    max_tokens: int = 120        # keep responses short for live feel
    temperature: float = 0.85


@dataclass
class LipSyncConfig:
    # model: "latsync" | "wav2lip"
    model: str = "latsync"
    base_face_image: str = "assets/face.jpg"            # reference portrait photo
    latsync_checkpoint: str = "checkpoints/latsync.ckpt"
    wav2lip_checkpoint: str = "checkpoints/wav2lip_gan.pth"
    device: str = "cuda"         # "cuda" or "cpu"
    fps: int = 25
    face_det_batch_size: int = 4
    wav2lip_batch_size: int = 128


@dataclass
class Config:
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    lipsync: LipSyncConfig = field(default_factory=LipSyncConfig)

    tiktok_username: str = ""
    chunk_duration_s: float = 1.5    # target seconds per TTS+lipsync chunk
    idle_interval_s: float = 12.0    # seconds between unprompted remarks
    max_queue_depth: int = 3         # drop old chunks if pipeline falls behind

    def load_from_env(self) -> "Config":
        """Override config values from environment variables."""
        if v := os.getenv("TIKTOK_USERNAME"):
            self.tiktok_username = v
        if v := os.getenv("RTMP_URL"):
            self.stream.rtmp_url = v
        if v := os.getenv("STREAM_KEY"):
            self.stream.stream_key = v
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
        if v := os.getenv("APP_DESCRIPTION"):
            self.persona.app_description = v
        return self
