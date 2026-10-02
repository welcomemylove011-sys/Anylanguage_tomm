import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

ELEVENLABS_API_KEY = os.getenv(
    "ELEVENLABS_API_KEY",
    ""
)


OUTPUT_DIR = "outputs"


MAX_VIDEO_SIZE_MB = 500
