import os
import re
import uuid
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import requests
import yt_dlp
from faster_whisper import WhisperModel


# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ELEVENLABS_MODEL = os.getenv(
    "ELEVENLABS_MODEL",
    "eleven_multilingual_v2"
)

ELEVENLABS_VOICE_ID = os.getenv(
    "ELEVENLABS_VOICE_ID",
    ""
).strip()

MAX_MINUTES = int(
    os.getenv("MAX_VIDEO_MINUTES", "15")
)

WHISPER = None
ELEVEN_VOICE_CACHE = None


# =========================================================
# SYSTEM
# =========================================================

def run_cmd(cmd):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if p.returncode != 0:
        raise RuntimeError(
            p.stderr[-5000:] or "Command failed"
        )

    return p.stdout


def check_tools():
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise RuntimeError(
                f"{name} မတွေ့ပါ။ FFmpeg install လုပ်ပါ။"
            )


def media_duration(path):
    result = run_cmd([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])

    return float(result.strip())


# =========================================================
# VIDEO DOWNLOAD
# =========================================================

def download_video(url, job, progress=None):

    url = (url or "").strip()

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(
            "မှန်ကန်တဲ့ public video URL ထည့်ပါ။"
        )

    output_template = str(
        job / "source.%(ext)s"
    )

    options = {
        "outtmpl": output_template,

        "format":
            "bv*[ext=mp4]+ba[ext=m4a]/"
            "b[ext=mp4]/b",

        "merge_output_format": "mp4",

        "noplaylist": True,

        "retries": 3,

        "fragment_retries": 3,

        "quiet": True,

        "no_warnings": True,
    }

    if progress:
        progress(
            0.07,
            "⬇️ Video download လုပ်နေပါတယ်..."
        )

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])

    except Exception as e:
        raise RuntimeError(
            f"Video download မအောင်မြင်ပါ:\n{e}"
        )

    files = sorted(
        job.glob("source.*")
    )

    if not files:
        raise RuntimeError(
            "Downloaded video file မတွေ့ပါ။"
        )

    return files[0]


# =========================================================
# AUDIO
# =========================================================

def extract_audio(video, job):

    audio = job / "audio.wav"

    run_cmd([
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio)
    ])

    return audio


# =========================================================
# FASTER WHISPER
# =========================================================

def get_whisper():

    global WHISPER

    if WHISPER is None:

        WHISPER = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

    return WHISPER


def merge_segments(items):

    if not items:
        return []

    merged = [
        dict(items[0])
    ]

    for current in items[1:]:

        previous = merged[-1]

        gap = (
            current["start"]
            - previous["end"]
        )

        combined_duration = (
            current["end"]
            - previous["start"]
        )

        if (
            gap <= 0.55
            and combined_duration <= 9
        ):

            previous["end"] = current["end"]

            previous["text"] = (
                previous["text"]
                + " "
                + current["text"]
            ).strip()

        else:

            merged.append(
                dict(current)
            )

    return merged


def transcribe(audio, progress=None):

    if progress:
        progress(
            0.20,
            f"🎧 Faster-Whisper {WHISPER_MODEL} loading..."
        )

    model = get_whisper()

    segments, info = model.transcribe(
        str(audio),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True
    )

    items = []

    for segment in segments:

        text = (
            segment.text or ""
        ).strip()

        if not text:
            continue

        items.append({
            "start":
                float(segment.start),

            "end":
                float(segment.end),

            "text":
                text
        })

    if not items:
        raise RuntimeError(
            "Video ထဲမှာ speech မတွေ့ပါ။"
        )

    detected_language = getattr(
        info,
        "language",
        "unknown"
    )

    if progress:
        progress(
            0.38,
            f"🎧 Transcription ပြီးပါပြီ — {detected_language}"
        )

    return merge_segments(items)


# =========================================================
# GEMINI TRANSLATION
# =========================================================

def gemini_generate(prompt):

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY မတွေ့ပါ။"
        )

    models = [
        GEMINI_MODEL,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    used = set()

    for model in models:

        if not model or model in used:
            continue

        used.add(model)

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )

        response = requests.post(
            url,
            params={
                "key": GEMINI_API_KEY
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2
                }
            },
            timeout=300
        )

        if response.status_code == 404:
            continue

        if response.status_code >= 400:

            try:
                detail = (
                    response.json()
                    .get("error", {})
                    .get("message", response.text)
                )
            except Exception:
                detail = response.text

            raise RuntimeError(
                f"Gemini API error "
                f"{response.status_code}: {detail}"
            )

        data = response.json()

        try:

            parts = (
                data["candidates"][0]
                ["content"]["parts"]
            )

            result = "".join(
                p.get("text", "")
                for p in parts
            ).strip()

            if result:
                return result

        except Exception:
            pass

    raise RuntimeError(
        "Gemini model ကို အသုံးပြုလို့မရပါ။"
    )


def translate_batch(items):

    source = "\n".join(
        f"{i + 1}. {item['text']}"
        for i, item in enumerate(items)
    )

    prompt = f"""
You are a professional Burmese dubbing translator.

Translate every numbered line into natural,
smooth spoken Burmese.

Rules:
- Preserve the original meaning.
- Do not summarize.
- Do not omit anything.
- Keep names and numbers correct.
- Make Burmese natural for voice dubbing.
- Return exactly the same numbered lines.
- Do not add explanations.

SOURCE:

{source}
"""

    result = gemini_generate(
        prompt
    )

    parsed = {}

    for line in result.splitlines():

        match = re.match(
            r"^\s*(\d+)\s*[.)-]\s*(.*)$",
            line.strip()
        )

        if match:

            parsed[
                int(match.group(1))
            ] = match.group(2).strip()

    if len(parsed) == len(items):

        return [
            parsed.get(
                i + 1,
                items[i]["text"]
            )
            for i in range(len(items))
        ]

    # Fallback
    return [
        gemini_generate(
            "Translate this into natural spoken Burmese. "
            "Return only the Burmese translation:\n\n"
            + item["text"]
        )
        for item in items
    ]


def translate_segments(
    segments,
    progress=None
):

    result = []

    total = len(segments)

    batch_size = 10

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = segments[
            start:start + batch_size
        ]

        translations = translate_batch(
            batch
        )

        for source, translated in zip(
            batch,
            translations
        ):

            item = dict(source)

            item["translation"] = (
                translated
                or source["text"]
            ).strip()

            result.append(item)

        done = min(
            total,
            start + len(batch)
        )

        if progress:

            progress(
                0.40
                + (
                    0.18
                    * done
                    / max(1, total)
                ),
                f"🌐 Burmese translation "
                f"{done}/{total}"
            )

    return result


# =========================================================
# ELEVENLABS
# =========================================================

def get_eleven_voice_id():

    global ELEVEN_VOICE_CACHE

    if ELEVENLABS_VOICE_ID:
        return ELEVENLABS_VOICE_ID

    if ELEVEN_VOICE_CACHE:
        return ELEVEN_VOICE_CACHE

    response = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers={
            "xi-api-key":
                ELEVENLABS_API_KEY
        },
        timeout=60
    )

    if response.status_code >= 400:

        try:
            detail = (
                response.json()
                .get("detail", response.text)
            )
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"ElevenLabs voices error "
            f"{response.status_code}: {detail}"
        )

    voices = (
        response.json()
        .get("voices", [])
    )

    if not voices:
        raise RuntimeError(
            "ElevenLabs voice မတွေ့ပါ။"
        )

    # Prefer a voice supporting multilingual model
    for voice in voices:

        model_ids = (
            voice.get(
                "high_quality_base_model_ids",
                []
            )
            or []
        )

        if ELEVENLABS_MODEL in model_ids:

            ELEVEN_VOICE_CACHE = (
                voice.get("voice_id")
            )

            if ELEVEN_VOICE_CACHE:
                return ELEVEN_VOICE_CACHE

    ELEVEN_VOICE_CACHE = (
        voices[0].get("voice_id")
    )

    if not ELEVEN_VOICE_CACHE:
        raise RuntimeError(
            "ElevenLabs voice ID မတွေ့ပါ။"
        )

    return ELEVEN_VOICE_CACHE


def elevenlabs_tts(
    text,
    output_file
):

    if not ELEVENLABS_API_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY မတွေ့ပါ။"
        )

    voice_id = (
        get_eleven_voice_id()
    )

    response = requests.post(

        f"https://api.elevenlabs.io/"
        f"v1/text-to-speech/{voice_id}",

        headers={
            "xi-api-key":
                ELEVENLABS_API_KEY,

            "Accept":
                "audio/mpeg",

            "Content-Type":
                "application/json"
        },

        json={
            "text": text[:5000],

            "model_id":
                ELEVENLABS_MODEL,

            "voice_settings": {
                "stability": 0.5,

                "similarity_boost":
                    0.75,

                "style": 0.0,

                "use_speaker_boost":
                    True
            }
        },

        timeout=300
    )

    if response.status_code >= 400:

        try:
            detail = (
                response.json()
                .get(
                    "detail",
                    response.text
                )
            )
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"ElevenLabs API error "
            f"{response.status_code}: "
            f"{detail}"
        )

    output_file.write_bytes(
        response.content
    )

    return output_file


# =========================================================
# AUDIO TIMING
# =========================================================

def atempo_filter(
    factor
):

    filters = []

    f = factor

    while f < 0.5:

        filters.append(
            "atempo=0.5"
        )

        f /= 0.5

    while f > 2.0:

        filters.append(
            "atempo=2.0"
        )

        f /= 2.0

    filters.append(
        f"atempo={f:.6f}"
    )

    return ",".join(
        filters
    )


def fit_audio(
    source,
    output,
    duration
):

    source_duration = (
        media_duration(source)
    )

    if source_duration <= 0:
        raise RuntimeError(
            "TTS audio duration မမှန်ပါ။"
        )

    factor = (
        source_duration
        / max(duration, 0.05)
    )

    run_cmd([
        "ffmpeg",
        "-y",

        "-i",
        str(source),

        "-filter:a",
        atempo_filter(factor),

        "-t",
        f"{duration:.3f}",

        "-ac",
        "2",

        "-ar",
        "48000",

        str(output)
    ])

    return output


# =========================================================
# BUILD BURMESE DUB
# =========================================================

def build_dub_audio(
    segments,
    video_duration,
    job,
    progress=None
):

    clips = []

    for i, segment in enumerate(
        segments
    ):

        duration = max(
            0.12,

            min(
                video_duration
                - segment["start"],

                segment["end"]
                - segment["start"]
            )
        )

        if duration <= 0.12:
            continue

        raw = (
            job
            / f"tts_{i:04d}.mp3"
        )

        fitted = (
            job
            / f"fit_{i:04d}.wav"
        )

        elevenlabs_tts(
            segment["translation"],
            raw
        )

        fit_audio(
            raw,
            fitted,
            duration
        )

        clips.append(
            (
                segment["start"],
                fitted
            )
        )

        if progress:

            progress(
                0.58
                + (
                    0.25
                    * (i + 1)
                    / max(
                        1,
                        len(segments)
                    )
                ),

                f"🗣️ ElevenLabs "
                f"{i + 1}/{len(segments)}"
            )

    if not clips:

        raise RuntimeError(
            "Burmese TTS audio မထွက်ပါ။"
        )

    inputs = []

    filters = []

    for i, (
        start,
        clip
    ) in enumerate(clips):

        inputs += [
            "-i",
            str(clip)
        ]

        delay = max(
            0,
            int(start * 1000)
        )

        filters.append(
            f"[{i}:a]"
            f"adelay={delay}|{delay}"
            f"[a{i}]"
        )

    mix_inputs = "".join(
        f"[a{i}]"
        for i in range(
            len(clips)
        )
    )

    filters.append(
        f"{mix_inputs}"
        f"amix="
        f"inputs={len(clips)}:"
        f"duration=longest:"
        f"normalize=0"
        f"[dub]"
    )

    output = (
        job
        / "burmese_dub.wav"
    )

    run_cmd([
        "ffmpeg",
        "-y",

        *inputs,

        "-filter_complex",
        ";".join(filters),

        "-map",
        "[dub]",

        "-t",
        f"{video_duration:.3f}",

        "-ac",
        "2",

        "-ar",
        "48000",

        str(output)
    ])

    return output


# =========================================================
# SUBTITLES
# =========================================================

def srt_time(seconds):

    seconds = max(
        0,
        float(seconds)
    )

    milliseconds = int(
        round(
            (
                seconds
                - int(seconds)
            )
            * 1000
        )
    )

    total = int(seconds)

    hours = total // 3600

    minutes = (
        total % 3600
    ) // 60

    secs = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def write_srt(
    segments,
    output
):

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        for i, segment in enumerate(
            segments,
            1
        ):

            f.write(
                f"{i}\n"
            )

            f.write(
                f"{srt_time(segment['start'])}"
                f" --> "
                f"{srt_time(segment['end'])}\n"
            )

            f.write(
                segment[
                    "translation"
                ].strip()
                + "\n\n"
            )


# =========================================================
# FINAL RENDER
# =========================================================

def mux_video(
    video,
    dub,
    output,
    original_volume,
    srt=None
):

    volume = max(
        0,
        min(
            1,
            float(original_volume)
        )
    )

    audio_filter = (
        f"[0:a]volume={volume:.3f}[orig];"
        f"[1:a]volume=1.0[dub];"
        f"[orig][dub]"
        f"amix=inputs=2:"
        f"duration=longest:"
        f"normalize=0[a]"
    )

    if srt:

        command = [
            "ffmpeg",
            "-y",

            "-i",
            str(video),

            "-i",
            str(dub),

            "-i",
            str(srt),

            "-filter_complex",
            audio_filter,

            "-map",
            "0:v:0",

            "-map",
            "[a]",

            "-map",
            "2:0",

            "-c:v",
            "copy",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-c:s",
            "mov_text",

            "-metadata:s:s:0",
            "language=mya",

            "-metadata:s:s:0",
            "title=Burmese",

            "-shortest",

            str(output)
        ]

    else:

        command = [
            "ffmpeg",
            "-y",

            "-i",
            str(video),

            "-i",
            str(dub),

            "-filter_complex",
            audio_filter,

            "-map",
            "0:v:0",

            "-map",
            "[a]",

            "-c:v",
            "copy",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-shortest",

            str(output)
        ]

    run_cmd(command)

    return output


# =========================================================
# MAIN PIPELINE
# =================
