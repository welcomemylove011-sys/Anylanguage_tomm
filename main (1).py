import os, re, json, uuid, shutil, subprocess, tempfile, time
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import requests
import yt_dlp

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

MAX_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "15"))
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-4000:] or "Command failed")
    return p.stdout.strip()


def ffmpeg_exists():
    return shutil.which("ffmpeg") is not None


def duration_seconds(path):
    out = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(out)


def safe_name(text):
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text or "video")
    return text.strip("_")[:60] or "video"


def get_video_path(value):
    if not value:
        return None
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return str(value)


def platform_name(url):
    host = urlparse(url).netloc.lower().replace("www.", "")
    if "youtube.com" in host or host == "youtu.be":
        return "YouTube"
    if "facebook.com" in host or "fb.watch" in host:
        return "Facebook"
    if "tiktok.com" in host:
        return "TikTok"
    if "threads.net" in host or "threads.com" in host:
        return "Threads"
    return host or "Unknown"


def download_video(url, job):
    if not url or not url.strip():
        raise ValueError("Video URL ထည့်ပါ။")
    url = url.strip()
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("မှန်ကန်တဲ့ http/https video URL ထည့်ပါ။")

    outtmpl = str(job / "downloaded.%(ext)s")
    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "retries": 2,
        "fragment_retries": 2,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info and info.get("duration") and info["duration"] > MAX_MINUTES * 60:
                raise ValueError(f"Video အရှည်က {MAX_MINUTES} မိနစ်ထက်ပိုနေပါတယ်။")
    except Exception as e:
        msg = str(e)
        if "Sign in to confirm" in msg or "not a bot" in msg.lower():
            raise RuntimeError(
                "ဒီ video site က downloader ကို anti-bot/authentication နဲ့ တားထားပါတယ်။ "
                "Public video တစ်ခုနဲ့ စမ်းပါ။ Login/cookie bypass မလုပ်ထားပါ။"
            )
        if "Unsupported URL" in msg or "No video formats found" in msg:
            raise RuntimeError(
                f"{platform_name(url)} URL ကို downloader က မထောက်ပံ့နိုင်သေးပါ။ "
                "Video ကို upload နည်းနဲ့ တိုက်ရိုက်တင်ပြီး ဆက်လုပ်နိုင်ပါတယ်။"
            )
        raise RuntimeError(f"Video download မအောင်မြင်ပါ: {msg[-1200:]}")

    candidates = list(job.glob("downloaded.*"))
    candidates = [p for p in candidates if p.suffix.lower() not in (".part", ".ytdl")]
    if not candidates:
        raise RuntimeError("Download ပြီးပေမယ့် video file မတွေ့ပါ။")
    return max(candidates, key=lambda x: x.stat().st_size)


def extract_audio(video, job):
    audio = job / "audio.mp3"
    run_cmd([
        "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", "64k", str(audio)
    ])
    return audio


def split_audio(audio, job):
    parts_dir = job / "chunks"
    parts_dir.mkdir(exist_ok=True)
    run_cmd([
        "ffmpeg", "-y", "-i", str(audio), "-f", "segment", "-segment_time", "600",
        "-reset_timestamps", "1", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(parts_dir / "chunk_%03d.mp3")
    ])
    return sorted(parts_dir.glob("chunk_*.mp3"))


def openai_post(path, data=None, files=None, timeout=300):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY မတွေ့ပါ။ GitHub Codespaces Secret ကို စစ်ပါ။")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r = requests.post(path, headers=headers, data=data, files=files, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"OpenAI API error {r.status_code}: {detail}")
    return r


def transcribe_chunk(chunk, offset, progress=None):
    with open(chunk, "rb") as f:
        r = openai_post(
            "https://api.openai.com/v1/audio/transcriptions",
            data={"model": TRANSCRIBE_MODEL, "response_format": "verbose_json"},
            files={"file": (chunk.name, f, "audio/mpeg")},
            timeout=600,
        )
    data = r.json()
    result = []
    for s in data.get("segments", []) or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        result.append({
            "start": float(s.get("start", 0)) + offset,
            "end": float(s.get("end", 0)) + offset,
            "text": text,
        })
    if not result and data.get("text"):
        result.append({"start": offset, "end": offset + 1.0, "text": data["text"].strip()})
    return result


def transcribe(audio, job, progress=None):
    chunks = split_audio(audio, job)
    all_segments = []
    for i, chunk in enumerate(chunks):
        if progress:
            progress(0.22 + 0.18 * (i / max(1, len(chunks))), f"🎧 Speech recognition {i+1}/{len(chunks)}")
        all_segments.extend(transcribe_chunk(chunk, i * 600.0))
    all_segments.sort(key=lambda x: x["start"])
    return merge_segments(all_segments)


def merge_segments(segments):
    merged = []
    for s in segments:
        if s["end"] <= s["start"] or not s["text"].strip():
            continue
        if merged:
            prev = merged[-1]
            gap = s["start"] - prev["end"]
            combined_duration = s["end"] - prev["start"]
            # Fewer TTS calls while keeping reasonably close timing.
            if gap < 0.55 and combined_duration <= 9.0:
                prev["text"] += " " + s["text"]
                prev["end"] = s["end"]
                continue
        merged.append(dict(s))
    return merged


def response_text(prompt):
    payload = {
        "model": TEXT_MODEL,
        "input": prompt,
        "max_output_tokens": 4000,
    }
    r = openai_post("https://api.openai.com/v1/responses", data={"model": TEXT_MODEL, "input": prompt, "max_output_tokens": 4000}, timeout=300)
    data = r.json()
    if data.get("output_text"):
        return data["output_text"].strip()
    pieces = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                pieces.append(c["text"])
    return "\n".join(pieces).strip()


def translate_batch(items):
    numbered = "\n".join(f"{i+1}. {x['text']}" for i, x in enumerate(items))
    prompt = f"""
You are a professional Burmese dubbing translator.
Translate every numbered line below into natural spoken Burmese (မြန်မာစကားပြောပုံစံ).
Rules:
- Preserve meaning, names, numbers and important details.
- Do not summarize or omit lines.
- Make it sound natural when spoken aloud.
- Return exactly the same number of numbered lines, one translation per line.
- Do not add commentary.

SOURCE:
{numbered}
"""
    text = response_text(prompt)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(\d+)\s*[.)-]\s*(.*)$", line)
        if m:
            lines.append((int(m.group(1)), m.group(2).strip()))
    if len(lines) == len(items):
        mapping = {n: t for n, t in lines}
        return [mapping.get(i + 1, items[i]["text"]) for i in range(len(items))]

    # Fallback: one segment at a time if the model did not preserve numbering.
    out = []
    for x in items:
        out.append(response_text(
            "Translate the following sentence into natural spoken Burmese. "
            "Return only the Burmese translation.\n\n" + x["text"]
        ))
    return out


def translate_segments(segments, progress=None):
    out = []
    batch_size = 10
    total = len(segments)
    for start in range(0, total, batch_size):
        batch = segments[start:start + batch_size]
        translations = translate_batch(batch)
        for s, t in zip(batch, translations):
            item = dict(s)
            item["translation"] = (t or s["text"]).strip()
            out.append(item)
        if progress:
            progress(0.40 + 0.18 * (min(total, start + len(batch)) / max(1, total)), f"🌐 Burmese translation {min(total, start+len(batch))}/{total}")
    return out


def tts(text, out_path):
    r = openai_post(
        "https://api.openai.com/v1/audio/speech",
        data={
            "model": TTS_MODEL,
            "voice": TTS_VOICE,
            "input": text[:4000],
            "response_format": "mp3",
            "instructions": "Speak natural, clear Burmese in a warm conversational style.",
        },
        timeout=300,
    )
    out_path.write_bytes(r.content)
    return out_path


def atempo_filter(factor):
    # atempo accepts 0.5..2.0 per filter; chain when necessary.
    factor = max(0.05, factor)
    filters = []
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={factor:.6f}")
    return ",".join(filters)


def fit_audio_to_duration(src, dst, target):
    src_dur = max(0.05, duration_seconds(src))
    factor = src_dur / max(0.05, target)
    af = atempo_filter(factor)
    run_cmd([
        "ffmpeg", "-y", "-i", str(src), "-af", f"{af},apad", "-t", f"{target:.3f}",
        "-ac", "1", "-ar", "48000", "-b:a", "128k", str(dst)
    ])


def build_dub_audio(segments, video_duration, job, progress=None):
    clips = []
    for i, s in enumerate(segments):
        dur = max(0.12, min(video_duration - s["start"], s["end"] - s["start"]))
        if dur <= 0.12:
            continue
        raw = job / f"tts_{i:04d}.mp3"
        fitted = job / f"fit_{i:04d}.wav"
        tts(s["translation"], raw)
        fit_audio_to_duration(raw, fitted, dur)
        clips.append((s["start"], fitted))
        if progress:
            progress(0.58 + 0.25 * ((i + 1) / max(1, len(segments))), f"🗣️ Burmese voice {i+1}/{len(segments)}")

    if not clips:
        raise RuntimeError("Burmese TTS audio မထွက်ပါ။")

    inputs = []
    filters = []
    for i, (start, clip) in enumerate(clips):
        inputs += ["-i", str(clip)]
        ms = max(0, int(start * 1000))
        filters.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
    mix_inputs = "".join(f"[a{i}]" for i in range(len(clips)))
    filters.append(f"{mix_inputs}amix=inputs={len(clips)}:duration=longest:normalize=0[dub]")
    dub = job / "burmese_dub.wav"
    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", ";".join(filters), "-map", "[dub]", "-t", f"{video_duration:.3f}", "-ac", "2", "-ar", "48000", str(dub)]
    run_cmd(cmd)
    return dub


def write_srt(segments, path):
    def ts(sec):
        sec = max(0, float(sec))
        ms = int(round((sec - int(sec)) * 1000))
        total = int(sec)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if ms >= 1000:
            s += 1; ms -= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s["translation"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def mux_video(video, dub_audio, srt_path, original_volume, job, add_subtitles):
    out = OUTPUT_DIR / f"burmese_dub_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
    # Keep the original video stream untouched where possible; replace audio.
    if add_subtitles:
        # Soft subtitle track in MP4; players can turn it on/off.
        cmd = [
            "ffmpeg", "-y", "-i", str(video), "-i", str(dub_audio), "-i", str(srt_path),
            "-filter_complex", f"[0:a]volume={float(original_volume):.3f}[orig];[orig][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
            "-map", "0:v:0", "-map", "[a]", "-map", "2:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
            "-metadata:s:s:0", "language=mya", "-metadata:s:s:0", "title=Burmese", str(out)
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(video), "-i", str(dub_audio),
            "-filter_complex", f"[0:a]volume={float(original_volume):.3f}[orig];[orig][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
            "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)
        ]
    run_cmd(cmd)
    return out


def process_video(url, upload, original_volume, add_subtitles, progress=gr.Progress()):
    if not ffmpeg_exists() or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe မတွေ့ပါ။ Codespace terminal မှာ `sudo apt-get update && sudo apt-get install -y ffmpeg` လုပ်ပါ။")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY မတွေ့ပါ။ GitHub Codespaces Secrets ကို စစ်ပါ။")
    source_upload = get_video_path(upload)
    if not source_upload and not (url and url.strip()):
        raise ValueError("Video URL ထည့်ပါ သို့မဟုတ် video file upload လုပ်ပါ။")

    job = Path(tempfile.mkdtemp(prefix="dub_"))
    try:
        progress(0.02, "🚀 Job စတင်နေပါတယ်…")
        if source_upload:
            video = Path(source_upload)
            if not video.exists():
                raise RuntimeError("Uploaded video file မတွေ့ပါ။")
            work_video = job / f"source{video.suffix.lower() or '.mp4'}"
            shutil.copy2(video, work_video)
        else:
            progress(0.08, f"⬇️ {platform_name(url)} video download လုပ်နေပါတယ်…")
            work_video = download_video(url, job)

        video_dur = duration_seconds(work_video)
        if video_dur > MAX_MINUTES * 60:
            raise ValueError(f"Video အရှည်က {MAX_MINUTES} မိနစ် limit ထက်ပိုနေပါတယ်။")

        progress(0.18, "🎧 Original audio ထုတ်နေပါတယ်…")
        audio = extract_audio(work_video, job)
        segments = transcribe(audio, job, progress)
        if not segments:
            raise RuntimeError("Speech မတွေ့ပါ။ Video ထဲမှာ စကားပြောသံရှိမရှိ စစ်ပါ။")

        progress(0.40, f"📝 {len(segments)} speech segments ရပါပြီ။")
        segments = translate_segments(segments, progress)
        srt = job / "burmese_subtitles.srt"
        write_srt(segments, srt)
        dub = build_dub_audio(segments, video_dur, job, progress)
        progress(0.87, "🎬 Burmese voice ကို video နဲ့ ပေါင်းနေပါတယ်…")
        final = mux_video(work_video, dub, srt, float(original_volume), job, bool(add_subtitles))
        final_srt = OUTPUT_DIR / (final.stem + ".srt")
        shutil.copy2(srt, final_srt)
        progress(1.0, "✅ Burmese dubbed video ပြီးပါပြီ")
        return str(final), str(final_srt), f"✅ ပြီးပါပြီ\n\nSource: {platform_name(url) if not source_upload else 'Uploaded file'}\nDuration: {video_dur/60:.1f} min\nSpeech segments: {len(segments)}\nOriginal audio volume: {float(original_volume):.2f}"
    except Exception:
        raise
    finally:
        shutil.rmtree(job, ignore_errors=True)


CSS = """
.gradio-container {max-width: 1180px !important;}
.hero {padding: 22px; border-radius: 20px; background: linear-gradient(135deg,#111827,#4f46e5); color:white; margin-bottom:16px;}
.hero h1 {font-size: 34px; margin-bottom: 8px;}
.note {padding:12px 14px; border-radius:12px; background:#f3f4f6;}
"""

with gr.Blocks(title="AI Burmese Video Dubbing", css=CSS, theme=gr.themes.Soft()) as app:
    gr.HTML("""
    <div class='hero'>
      <h1>🎬 AI Burmese Video Dubbing</h1>
      <div>YouTube / Facebook / TikTok public video → Detect speech → Burmese translate → Burmese voice → Final MP4</div>
    </div>
    """)

    with gr.Tab("🎬 Video Dubbing"):
        with gr.Row():
            with gr.Column(scale=2):
                url = gr.Textbox(label="Video URL", placeholder="https://www.youtube.com/... / https://www.tiktok.com/... / Facebook public video URL")
                upload = gr.Video(label="သို့မဟုတ် Video File တိုက်ရိုက် Upload", sources=["upload"], type="filepath")
                gr.Markdown("**Public video ကိုသာ URL နဲ့ download လုပ်ပါတယ်။** Site တစ်ခုချင်းစီရဲ့ anti-bot/login restriction ကြောင့် URL တချို့ မရနိုင်ပါ။")
            with gr.Column(scale=1):
                original_volume = gr.Slider(0, 1, value=0.18, step=0.02, label="Original audio volume")
                add_subtitles = gr.Checkbox(value=True, label="Burmese subtitles ထည့်မယ်")
                start = gr.Button("🚀 Translate & Dub to Burmese", variant="primary", size="lg")

        status = gr.Textbox(label="Status", lines=7)
        with gr.Row():
            result = gr.Video(label="🎞️ Final Burmese Dubbed Video")
            srt_file = gr.File(label="📄 Burmese SRT")

        start.click(process_video, [url, upload, original_volume, add_subtitles], [result, srt_file, status])

    with gr.Tab("ℹ️ How it works"):
        gr.Markdown("""
### Pipeline
1. Video URL ကို download (သို့) upload file ကိုယူမယ်
2. Audio ကို extract လုပ်မယ်
3. Speech language ကို AI နဲ့ detect/transcribe လုပ်မယ်
4. စကားပြောစာသားကို သဘာဝကျ Burmese ပြန်ဆိုမယ်
5. Burmese TTS အသံထုတ်မယ်
6. Original audio ကို လျှော့ပြီး Burmese voice ကို mix လုပ်မယ်
7. Burmese subtitle track + SRT ထုတ်မယ်
8. Final MP4 ပြန်ပေးမယ်

**မှတ်ချက်:** ဒီ version က original speech ကို AI voice-isolation နဲ့ 100% ဖယ်ရှားတာမဟုတ်ဘဲ original audio ကို လျှော့ပြီး Burmese voice ကို အပေါ်ကနေ mix လုပ်ပါတယ်။ Music/background sound ကို တတ်နိုင်သမျှ ထိန်းထားဖို့ ဒီနည်းကိုသုံးထားပါတယ်။
        """)

if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
