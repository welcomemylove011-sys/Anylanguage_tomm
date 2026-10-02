
import os, uuid, shutil, subprocess, threading, re, base64, json, traceback, time, socket
from urllib.parse import urlparse
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests

BASE=Path(__file__).resolve().parent; UPLOADS=BASE/'uploads'; JOBS=BASE/'jobs'
UPLOADS.mkdir(exist_ok=True); JOBS.mkdir(exist_ok=True)
app=FastAPI(title='The Quick Plot AI Video Translator')
app.mount('/static',StaticFiles(directory=str(BASE)),name='static')
jobs={}; lock=threading.Lock()
GEMINI_KEY=os.getenv('GEMINI_API_KEY','').strip()
GEMINI_TTS_MODEL=os.getenv('GEMINI_TTS_MODEL','gemini-3.8-flash-lite-tts')
GEMINI_TEXT_MODEL=os.getenv('GEMINI_TEXT_MODEL','gemini-3.8-flash')
TEXT_MODEL_FALLBACKS=['gemini-3.7-flash','gemini-3.5-flash','gemini-3.5-flash-lite']
TTS_MODEL_FALLBACKS=['gemini-3.8-flash-tts','gemini-2.5-flash-preview-tts']
GEMINI_TRANSCRIBE_MODEL=os.getenv('GEMINI_TRANSCRIBE_MODEL','gemini-3.5-transcribe')
GEMINI_IMAGE_MODEL=os.getenv('GEMINI_IMAGE_MODEL','gemini-3.1-flash-image')
MUSIC_DIR=BASE/'music'; MUSIC_DIR.mkdir(exist_ok=True)

PREBUILT=[('Zephyr','Bright'),('Puck','Upbeat'),('Charon','Informative'),('Kore','Firm'),('Fenrir','Excitable'),('Leda','Youthful'),
('Orus','Firm'),('Aoede','Breezy'),('Callirrhoe','Easy-going'),('Autonoe','Bright'),('Enceladus','Breathy'),('Iapetus','Clear'),
('Umbriel','Easy-going'),('Algieba','Smooth'),('Despina','Smooth'),('Erinome','Clear'),('Algenib','Gravelly'),('Rasalgethi','Informative'),
('Laomedeia','Upbeat'),('Achernar','Soft'),('Alnilam','Firm'),('Schedar','Even'),('Gacrux','Mature'),('Pulcherrima','Forward'),
('Achird','Friendly'),('Zubenelgenubi','Casual'),('Vindemiatrix','Gentle'),('Sadachbia','Lively'),('Sadaltager','Knowledgeable'),('Sulafat','Warm')]
STYLE_PRESETS={'natural':'natural conversational Burmese, warm, human, relaxed pacing, clear pronunciation',
'narrator':'professional documentary narrator, confident, smooth, engaging, natural pauses',
'friendly':'friendly, warm, approachable, cheerful but not exaggerated, conversational',
'news':'clear news presenter, composed, authoritative, crisp diction, moderate pace',
'story':'expressive storyteller, warm emotion, varied pacing, vivid but natural delivery',
'calm':'calm, gentle, reassuring, soft delivery, unhurried natural pauses',
'energetic':'energetic, enthusiastic, lively, upbeat, natural conversational rhythm'}

def update(j,**kw):
    with lock:
        if j in jobs: jobs[j].update(kw)
def cmd(c):
    p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode:
        pretty=' '.join(str(x) for x in c)
        detail=p.stderr[-7000:] or 'command failed'
        raise RuntimeError(f'Command failed: {pretty}\n{detail}')
    return p.stdout
def duration(p):
    return float(cmd(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(p)]).strip())
def extract(v,a): cmd(['ffmpeg','-y','-i',str(v),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(a)])

def _gemini_call(payload,timeout=240):
    r=requests.post('https://generativelanguage.googleapis.com/v1beta/interactions',
        headers={'x-goog-api-key':GEMINI_KEY,'Content-Type':'application/json'},json=payload,timeout=timeout)
    if r.ok: return r.json()
    try: detail=r.json().get('error',{}).get('message',r.text)
    except Exception: detail=r.text
    return None, r.status_code, str(detail)

def gemini_request(payload,timeout=240):
    if not GEMINI_KEY: raise RuntimeError('GEMINI_API_KEY is missing. Add it in Space Settings → Secrets.')
    model=payload.get('model','')
    candidates=[model]
    if model==GEMINI_TEXT_MODEL: candidates += [m for m in TEXT_MODEL_FALLBACKS if m not in candidates]
    if model==GEMINI_TTS_MODEL: candidates += [m for m in TTS_MODEL_FALLBACKS if m not in candidates]
    last='Unknown Gemini error'
    for candidate in candidates:
        body=dict(payload); body['model']=candidate
        result=_gemini_call(body,timeout)
        if isinstance(result,tuple):
            _, status, detail=result
            last=f'{candidate} ({status}): {detail}'
            # Retry a different model for temporary capacity/rate-limit/server errors.
            if status not in (429,500,502,503,504): break
            continue
        if candidate != model:
            payload['model']=candidate
        return result
    raise RuntimeError(f'Gemini API error: {last}. The app automatically tried fallback models; please retry if Google capacity is temporarily full.')
def output_text(data):
    if data.get('output_text'): return data['output_text'].strip()
    for step in reversed(data.get('steps',[])):
        for c in step.get('content',[]) if isinstance(step,dict) else []:
            if isinstance(c,dict) and c.get('type')=='text' and c.get('text'): return c['text'].strip()
    return ''

def transcribe_with_gemini(audio_path):
    meta={'file':{'display_name':audio_path.name}}
    r=requests.post('https://generativelanguage.googleapis.com/upload/v1beta/files',
        headers={'x-goog-api-key':GEMINI_KEY,'X-Goog-Upload-Protocol':'resumable','X-Goog-Upload-Command':'start',
                 'X-Goog-Upload-Header-Content-Length':str(audio_path.stat().st_size),
                 'X-Goog-Upload-Header-Content-Type':'audio/wav','Content-Type':'application/json'},
        data=json.dumps(meta),timeout=60)
    if not r.ok: raise RuntimeError(f'Gemini file upload failed: {r.text[-2000:]}')
    endpoint=r.headers.get('X-Goog-Upload-URL')
    if not endpoint: raise RuntimeError('Gemini file upload did not return an upload URL.')
    with audio_path.open('rb') as f:
        r2=requests.post(endpoint,headers={'Content-Length':str(audio_path.stat().st_size),'X-Goog-Upload-Offset':'0',
            'X-Goog-Upload-Command':'upload, finalize','Content-Type':'audio/wav'},data=f,timeout=300)
    if not r2.ok: raise RuntimeError(f'Gemini file finalize failed: {r2.text[-2000:]}')
    fo=r2.json().get('file',r2.json()); uri=fo.get('uri'); mime=fo.get('mimeType','audio/wav')
    if not uri: raise RuntimeError('Gemini file upload returned no URI.')
    payload={'model':GEMINI_TRANSCRIBE_MODEL,
        'input':[{'type':'text','text':'Transcribe this audio. Automatically detect the spoken language. Preserve code-switching. Return only the transcript; do not translate.'},
                 {'type':'audio','uri':uri,'mime_type':mime}],
        'generation_config':{'transcription_config':{'language_codes':[]}},'store':False}
    text=output_text(gemini_request(payload,300))
    if not text: raise RuntimeError('Gemini Transcribe returned no transcript.')
    return text

def split_text(text,max_chars=1800):
    text=re.sub(r'\s+',' ',text).strip()
    if not text: return []
    sentences=re.split(r'(?<=[.!?။！？])\s+',text)
    chunks=[]; cur=''
    for s in sentences:
        if len(cur)+len(s)+1<=max_chars: cur=(cur+' '+s).strip()
        else:
            if cur: chunks.append(cur)
            while len(s)>max_chars:
                cut=s.rfind(' ',0,max_chars)
                if cut<200: cut=max_chars
                chunks.append(s[:cut].strip()); s=s[cut:].strip()
            cur=s
    if cur: chunks.append(cur)
    return chunks

def natural_myanmar(text):
    outs=[]
    for chunk in split_text(text,1800):
        prompt="""Translate the spoken content below into natural, modern conversational Myanmar (Burmese).
Output ONLY the Myanmar translation.
Preserve meaning, names, numbers and important technical terms.
Do not translate word-for-word when that sounds unnatural.
Write like a Myanmar person is speaking naturally in a video.
Prefer conversational endings such as "တယ်", "မယ်", "ပါတယ်", "ပါမယ်", "လို့ရတယ်", "ဖြစ်တယ်" when appropriate.
Avoid stiff written endings such as "သည်", "မည်", "၏" unless genuinely necessary.
Do not add headings, explanations, quotation marks or commentary.

SOURCE:
""" + chunk
        out=output_text(gemini_request({'model':GEMINI_TEXT_MODEL,'input':prompt,'store':False},240))
        if out: outs.append(out)
    result='\n'.join(outs).strip()
    if not result: raise RuntimeError('Myanmar translation returned no text.')
    return result

def gemini_tts(text,out_wav,voice,style,custom_style=''):
    style_text=STYLE_PRESETS.get(style,STYLE_PRESETS['natural'])
    if custom_style.strip(): style_text += ', '+custom_style.strip()[:500]
    parts=[]
    for i,chunk in enumerate(split_text(text,850)):
        payload={'model':GEMINI_TTS_MODEL,'input':[{'type':'user_input','content':[{'type':'text','text':chunk,
            'annotations':[{'type':'speech_metadata','style':style_text}]}]}],
            'response_format':{'type':'audio','mime_type':'audio/wav','sample_rate':24000},
            'generation_config':{'speech_config':[{'voice':voice}]},'store':False}
        data=gemini_request(payload,300); audio=data.get('output_audio',{}).get('data')
        if not audio:
            for step in reversed(data.get('steps',[])):
                for c in step.get('content',[]) if isinstance(step,dict) else []:
                    if isinstance(c,dict) and c.get('type')=='audio' and c.get('data'): audio=c['data']; break
                if audio: break
        if not audio: raise RuntimeError('Gemini TTS returned no audio data.')
        p=out_wav.parent/f'{out_wav.stem}_{i:03d}.wav'; p.write_bytes(base64.b64decode(audio)); parts.append(p)
    if len(parts)==1: shutil.copyfile(parts[0],out_wav)
    else:
        concat=out_wav.parent/'tts_concat.txt'
        concat.write_text(''.join(f"file '{p.as_posix()}'\n" for p in parts),encoding='utf-8')
        cmd(['ffmpeg','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(out_wav)])

def upload_gemini_file(path, mime_type):
    meta={'file':{'display_name':path.name}}
    r=requests.post('https://generativelanguage.googleapis.com/upload/v1beta/files', headers={'x-goog-api-key':GEMINI_KEY,'X-Goog-Upload-Protocol':'resumable','X-Goog-Upload-Command':'start','X-Goog-Upload-Header-Content-Length':str(path.stat().st_size),'X-Goog-Upload-Header-Content-Type':mime_type,'Content-Type':'application/json'}, data=json.dumps(meta), timeout=60)
    if not r.ok: raise RuntimeError(f'Gemini file upload failed: {r.text[-2000:]}')
    endpoint=r.headers.get('X-Goog-Upload-URL')
    if not endpoint: raise RuntimeError('Gemini file upload did not return an upload URL.')
    with path.open('rb') as f:
        r2=requests.post(endpoint, headers={'Content-Length':str(path.stat().st_size),'X-Goog-Upload-Offset':'0','X-Goog-Upload-Command':'upload, finalize','Content-Type':mime_type}, data=f, timeout=600)
    if not r2.ok: raise RuntimeError(f'Gemini file finalize failed: {r2.text[-2000:]}')
    fo=r2.json().get('file',r2.json()); uri=fo.get('uri'); mime=fo.get('mimeType',mime_type)
    if not uri: raise RuntimeError('Gemini file upload returned no URI.')
    return uri,mime

def generate_thumbnail(video_path, transcript, out_path):
    uri,mime=upload_gemini_file(video_path,'video/mp4' if video_path.suffix.lower()=='.mp4' else 'video/*')
    prompt='Create a compelling YouTube-style 16:9 thumbnail for this video. Use the video as the visual reference and understand its main subject/topic. Choose one strong, emotionally interesting hero moment or subject. Make it clean, cinematic, high-contrast, mobile-readable, and professional. Do not copy logos or copyrighted artwork. Keep one clear focal point and space for a short headline. If text is useful, use a very short English or Myanmar headline based on the topic; otherwise use no text. Avoid misleading clickbait.\n\nTranscript context:\n'+(transcript or '')[:5000]
    data=gemini_request({'model':GEMINI_IMAGE_MODEL,'input':[{'type':'video','uri':uri,'mime_type':mime},{'type':'text','text':prompt}], 'response_format':{'type':'image','mime_type':'image/jpeg','aspect_ratio':'16:9','image_size':'1K'},'store':False},300)
    image=data.get('output_image',{}).get('data')
    if not image:
        for step in reversed(data.get('steps',[])):
            for c in step.get('content',[]) if isinstance(step,dict) else []:
                if isinstance(c,dict) and c.get('type')=='image' and c.get('data'): image=c['data']; break
            if image: break
    if not image: raise RuntimeError('Gemini thumbnail generation returned no image.')
    out_path.write_bytes(base64.b64decode(image))

def safe_music_path(name):
    allowed={'none':None,'piano':'original-piano.wav','guitar':'original-acoustic-guitar.wav','dj':'original-dj-beat.wav'}
    fn=allowed.get(name); return MUSIC_DIR/fn if fn else None

def srt_time(sec):
    ms=int(round(max(0,sec)*1000)); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'
def make_srt(text,total,path):
    chunks=[x.strip() for x in re.split(r'\n+|(?<=[.!?။！？])\s+',text.strip()) if x.strip()]
    if not chunks: return
    weights=[max(1,len(x)) for x in chunks]; totalw=sum(weights); cur=0; lines=[]
    for i,c in enumerate(chunks,1):
        end=total if i==len(chunks) else min(total,cur+total*weights[i-1]/totalw)
        lines.append(f'{i}\n{srt_time(cur)} --> {srt_time(end)}\n{c}\n'); cur=end
    path.write_text('\n'.join(lines),encoding='utf-8')

def ass_time(sec):
    cs=int(round(max(0,sec)*100)); h,cs=divmod(cs,360000); m,cs=divmod(cs,6000); s,cs=divmod(cs,100)
    return f'{h}:{m:02d}:{s:02d}.{cs:02d}'

def hex_to_ass(value, alpha=0):
    value=(value or '#FFFFFF').strip().lstrip('#')
    if not re.fullmatch(r'[0-9a-fA-F]{6}', value):
        value='FFFFFF'
    rr,gg,bb=value[0:2],value[2:4],value[4:6]
    return f'&H{int(alpha):02X}{bb}{gg}{rr}'

def make_ass(text,total,path,font_size=52,scale_x=100,scale_y=100,position_x=50,position_y=88,
             alignment=2,text_color='#FFFFFF',outline_width=3,outline_color='#000000',
             shadow=1,background=True,background_color='#000000',background_opacity=55):
    chunks=[x.strip() for x in re.split(r'\n+|(?<=[.!?။！？])\s+',text.strip()) if x.strip()]
    if not chunks: return
    weights=[max(1,len(x)) for x in chunks]; totalw=sum(weights); cur=0
    try:
        alignment=int(alignment)
        alignment=alignment if 1 <= alignment <= 9 else 2
    except Exception: alignment=2
    try:
        font_size=max(10,min(180,int(float(font_size))))
        scale_x=max(50,min(200,int(float(scale_x))))
        scale_y=max(50,min(200,int(float(scale_y))))
        px=max(0,min(100,float(position_x))); py=max(0,min(100,float(position_y)))
        outline_width=max(0,min(20,float(outline_width)))
        shadow=max(0,min(20,float(shadow)))
        bg_opacity=max(0,min(100,float(background_opacity)))
    except Exception:
        font_size,scale_x,scale_y,px,py,outline_width,shadow,bg_opacity=52,100,100,50,88,3,1,55
    x=int(1920*px/100); y=int(1080*py/100)
    # ASS alpha: 00 = opaque, FF = transparent.
    bg_alpha=255-int(255*(bg_opacity/100)) if background else 255
    border_style=3 if background else 1
    lines=[
        '[Script Info]','ScriptType: v4.00+','PlayResX: 1920','PlayResY: 1080','WrapStyle: 2',
        'ScaledBorderAndShadow: yes','',
        '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
        f'Style: Myanmar,Noto Sans Myanmar,{font_size},{hex_to_ass(text_color)},{hex_to_ass(text_color)},{hex_to_ass(outline_color)},{hex_to_ass(background_color,bg_alpha)},0,0,{border_style},{outline_width:.1f},{shadow:.1f},{alignment},30,30,30,1',
        '',
        '[Events]','Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text'
    ]
    for i,c in enumerate(chunks):
        end=total if i==len(chunks)-1 else min(total,cur+total*weights[i]/totalw)
        safe=c.replace('\\','\\\\').replace('\n','\\N').replace('{','\\{').replace('}','\\}')
        # Exact draggable position, independent of screen margins.
        tag=f'{{\\an{alignment}\\pos({x},{y})\\fscx{scale_x}\\fscy{scale_y}}}'
        lines.append(f'Dialogue: 0,{ass_time(cur)},{ass_time(end)},Myanmar,,0,0,0,,{tag}{safe}')
        cur=end
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def ffmpeg_filters():
    try:
        out=cmd(['ffmpeg','-hide_banner','-filters'])
        names=set()
        for line in out.splitlines():
            m=re.match(r'^\s*[TSC.]+\s+(\S+)\s',line)
            if m: names.add(m.group(1))
        return names
    except Exception:
        return set()

def _filter_path(path):
    # FFmpeg filtergraph escaping for a local POSIX path.
    return str(path).replace('\\','/').replace(':','\\:').replace("'","\\'")

def render(v,a,out,d,subtitle_mode,color_preset,brightness,contrast,saturation,
           subtitle_font_size=52,subtitle_scale_x=100,subtitle_scale_y=100,
           subtitle_x=50,subtitle_y=88,subtitle_alignment=2,subtitle_color='#FFFFFF',
           subtitle_outline_width=3,subtitle_outline_color='#000000',subtitle_shadow=1,
           subtitle_background=True,subtitle_background_color='#000000',subtitle_background_opacity=55,music_track='none',music_volume=0.12):
    presets={'original':(0,1,1),'cinematic':(-.02,1.10,.92),'warm':(.02,1.06,1.08),'cool':(-.01,1.04,.96),
             'bright':(.06,1.02,1.04),'dark':(-.05,1.08,.96),'vivid':(.01,1.12,1.18)}
    pb,pc,ps=presets.get(color_preset,presets['original']); b=float(brightness)+pb; c=float(contrast)*pc; s=float(saturation)*ps
    vfparts=[]
    if abs(b)>.001 or abs(c-1)>.001 or abs(s-1)>.001:
        vfparts.append(f'eq=brightness={b:.3f}:contrast={c:.3f}:saturation={s:.3f}')

    # Use ASS directly for Myanmar subtitles. This avoids the older `subtitles`
    # filter path that caused failures in some FFmpeg builds.
    ap=out.parent/'myanmar.ass'
    if subtitle_mode=='myanmar':
        txt=out.parent/'myanmar.txt'
        if not txt.exists():
            raise RuntimeError('Myanmar subtitle text was not created.')
        make_ass(txt.read_text(encoding='utf-8'),d,ap,subtitle_font_size,subtitle_scale_x,subtitle_scale_y,
                  subtitle_x,subtitle_y,subtitle_alignment,subtitle_color,subtitle_outline_width,
                  subtitle_outline_color,subtitle_shadow,subtitle_background,subtitle_background_color,
                  subtitle_background_opacity)
        filters=ffmpeg_filters()
        if 'ass' not in filters:
            raise RuntimeError('This FFmpeg build does not contain the ASS/libass filter. Rebuild the Space from the supplied Dockerfile.')
        vfparts.append(f"ass={_filter_path(ap)}:fontsdir=/usr/share/fonts/truetype/noto")

    # Separate -vf and -af graphs are much more reliable than a combined
    # filter_complex for this workflow.
    ad=duration(a)
    afparts=['aresample=48000']
    if ad>d*1.02 and d>0:
        ratio=ad/d
        while ratio>2:
            afparts.append('atempo=2.0'); ratio/=2
        afparts.append(f'atempo={max(.5,ratio):.5f}')
    afparts += ['apad','atrim=duration={:.3f}'.format(max(.1,d)),'asetpts=N/SR/TB']

    music_path=safe_music_path(music_track)
    args=['ffmpeg','-y','-i',str(v),'-i',str(a)]
    if music_path and music_path.exists(): args += ['-stream_loop','-1','-i',str(music_path)]
    args += ['-map','0:v:0']
    if vfparts: args += ['-vf',','.join(vfparts)]
    if music_path and music_path.exists():
        mv=max(0.0,min(1.0,float(music_volume)))
        graph=f"[1:a]{','.join(afparts)}[voice];[2:a]aresample=48000,volume={mv:.3f},atrim=duration={max(.1,d):.3f},asetpts=N/SR/TB[music];[voice][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        args += ['-filter_complex',graph,'-map','[aout]']
    else:
        args += ['-map','1:a:0','-af',','.join(afparts)]
    args += ['-c:v','libx264','-preset','veryfast','-crf','23','-c:a','aac','-b:a','128k','-movflags','+faststart','-t',str(d)]
    if subtitle_mode=='keep': args += ['-map','0:s?','-c:s','mov_text']
    args += [str(out)]
    cmd(args)

def work(j,input_path,voice,style,custom_style,subtitle_mode,color_preset,brightness,contrast,saturation,
         subtitle_font_size,subtitle_scale_x,subtitle_scale_y,subtitle_x,subtitle_y,subtitle_alignment,
         subtitle_color,subtitle_outline_width,subtitle_outline_color,subtitle_shadow,subtitle_background,
         subtitle_background_color,subtitle_background_opacity,music_track='none',music_volume=0.12,auto_thumbnail=True):
    w=JOBS/j; w.mkdir(exist_ok=True)
    try:
        update(j,status='processing',progress=5,message='Extracting audio…')
        d=duration(input_path); audio=w/'audio.wav'; extract(input_path,audio)
        update(j,progress=20,message='Auto-detecting language and transcribing speech…')
        tr=transcribe_with_gemini(audio); (w/'transcript.txt').write_text(tr,encoding='utf-8')
        update(j,progress=40,message='Translating to natural conversational Myanmar…')
        my=natural_myanmar(tr); (w/'myanmar.txt').write_text(my,encoding='utf-8')
        if subtitle_mode=='myanmar': make_srt(my,d,w/'myanmar.srt')
        update(j,progress=62,message=f'Generating natural Myanmar voice with {voice}…')
        t=w/'myanmar.wav'; gemini_tts(my,t,voice,style,custom_style)
        update(j,progress=82,message='Applying subtitles and colour grading…')
        out=w/'final_myanmar.mp4'; render(input_path,t,out,d,subtitle_mode,color_preset,brightness,contrast,saturation,
               subtitle_font_size,subtitle_scale_x,subtitle_scale_y,subtitle_x,subtitle_y,subtitle_alignment,
               subtitle_color,subtitle_outline_width,subtitle_outline_color,subtitle_shadow,
               subtitle_background,subtitle_background_color,subtitle_background_opacity,music_track,music_volume)
        thumb_url=None
        if auto_thumbnail:
            update(j,progress=94,message='Generating AI thumbnail…')
            try:
                thumb=w/'thumbnail.jpg'; generate_thumbnail(input_path,tr,thumb)
                thumb_url=f'/api/jobs/{j}/thumbnail'
            except Exception as thumb_err:
                print(f'JOB {j} THUMBNAIL WARNING: {thumb_err}', flush=True)
                update(j,message='Video finished; thumbnail generation was unavailable.')
        update(j,status='done',progress=100,message='Finished',transcript=tr,translation=my,video_url=f'/api/jobs/{j}/video',thumbnail_url=thumb_url)
    except Exception as e:
        detail=str(e)
        print(f'JOB {j} FAILED: {detail}', flush=True)
        print(traceback.format_exc(), flush=True)
        update(j,status='error',progress=0,message=detail)

@app.get('/')
def root(): return FileResponse(BASE/'index.html')
@app.get('/api/health')
def health():
    return {'ok':True,'gemini_configured':bool(GEMINI_KEY),'tts_model':GEMINI_TTS_MODEL,'text_model':GEMINI_TEXT_MODEL,'transcribe_model':GEMINI_TRANSCRIBE_MODEL,'image_model':GEMINI_IMAGE_MODEL}
@app.get('/api/voices')
def voices():
    if not GEMINI_KEY: return {'voices':[{'id':v,'name':v,'description':d} for v,d in PREBUILT],'live':False}
    try:
        r=requests.get('https://generativelanguage.googleapis.com/v1beta/voices',headers={'x-goog-api-key':GEMINI_KEY},
                        params={'language_code':'my','page_size':1000},timeout=30)
        if r.ok:
            vals=r.json().get('voices',[])
            if vals: return {'voices':[{'id':v.get('id') or v.get('name','').split('/')[-1],
                'name':v.get('display_name') or v.get('id') or 'Voice','description':v.get('description',''),
                'gender':v.get('gender',''),'accent':v.get('accent',''),'persona':v.get('persona','')} for v in vals],'live':True}
    except Exception: pass
    return {'voices':[{'id':v,'name':v,'description':d} for v,d in PREBUILT],'live':False}
ALLOWED_VIDEO_HOSTS={
    'youtube.com','www.youtube.com','youtu.be','m.youtube.com',
    'facebook.com','www.facebook.com','m.facebook.com','fb.watch',
    'tiktok.com','www.tiktok.com','vm.tiktok.com',
    'threads.net','www.threads.net','threads.com','www.threads.com'
}
def validate_video_url(value):
    try:
        u=urlparse(value.strip())
        host=(u.hostname or '').lower().rstrip('.')
        if u.scheme not in ('http','https') or not host:
            raise ValueError
        if not any(host==h or host.endswith('.'+h) for h in ALLOWED_VIDEO_HOSTS):
            raise ValueError
        return value.strip()
    except Exception:
        raise HTTPException(400,'Only YouTube, Facebook, TikTok, and Threads video links are supported.')

def _youtube_cookie_file():
    """
    Optional YouTube authentication for server-side yt-dlp.

    Preferred Space Secret:
      YOUTUBE_COOKIES_B64 = base64-encoded Netscape cookies.txt

    Also accepts:
      YOUTUBE_COOKIES = raw Netscape cookies.txt

    The cookie file is written only to the ephemeral Space container and is
    never returned by the API.
    """
    b64 = os.getenv('YOUTUBE_COOKIES_B64', '').strip()
    raw = os.getenv('YOUTUBE_COOKIES', '').strip()
    if not b64 and not raw:
        return None

    try:
        if b64:
            data = base64.b64decode(b64, validate=True)
        else:
            data = raw.encode('utf-8')

        if not data.startswith(b'# HTTP Cookie File') and not data.startswith(b'# Netscape HTTP Cookie File'):
            raise RuntimeError(
                'YOUTUBE_COOKIES_B64/YOUTUBE_COOKIES is not a Netscape cookies.txt file. '
                'Export YouTube cookies in Netscape format.'
            )

        path = BASE / '.youtube-cookies.txt'
        path.write_bytes(data)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return path
    except Exception as e:
        raise RuntimeError(f'Could not prepare YouTube cookies: {e}')


def _js_runtime_args():
    """
    Use an installed JS runtime when available. Recent yt-dlp YouTube
    extraction can use an external JS runtime for its EJS challenge.
    """
    for runtime in ('deno', 'node', 'nodejs', 'bun'):
        if shutil.which(runtime):
            return ['--js-runtimes', runtime]
    return []


def _is_threads_url(url):
    host = (urlparse(url).hostname or '').lower().rstrip('.')
    return host in {'threads.com', 'www.threads.com', 'threads.net', 'www.threads.net'} or host.endswith('.threads.com') or host.endswith('.threads.net')


def _is_youtube_url(url):
    host = (urlparse(url).hostname or '').lower().rstrip('.')
    return host in {
        'youtube.com', 'www.youtube.com', 'm.youtube.com',
        'youtu.be', 'www.youtu.be'
    } or host.endswith('.youtube.com')


def download_video_url(url, destination):
    """Download one video with platform-specific extractors and fallbacks.

    YouTube keeps the tested mweb + bgutil PO-token flow.
    Threads uses the dedicated yt-dlp Threads extractor plugin because
    upstream yt-dlp does not currently include a native Threads extractor.
    Facebook/TikTok continue through normal yt-dlp extraction.
    """
    template = str(destination.with_suffix('.%(ext)s'))
    is_yt = _is_youtube_url(url)
    is_threads = _is_threads_url(url)
    cookie_file = None

    common = [
        'python3', '-m', 'yt_dlp',
        '--no-playlist',
        '--no-warnings',
        '--restrict-filenames',
        '--socket-timeout', '60',
        '--retries', '10',
        '--fragment-retries', '10',
        '--extractor-retries', '5',
        '--retry-sleep', '1:3',
        '--force-ipv4',
        '--impersonate', 'chrome',
        '--js-runtimes', 'node',
        '--merge-output-format', 'mp4',
    ]

    attempts = []

    if is_yt:
        # Primary/current YouTube route: mweb + bgutil PO token provider.
        cookie_file = _youtube_cookie_file()
        yt_common = [
            '--extractor-args',
            'youtube:player-client=mweb;fetch_pot=always',
            '--extractor-args',
            'youtubepot-bgutilhttp:base_url=http://127.0.0.1:4416',
        ]
        cookie_args = ['--cookies', str(cookie_file)] if cookie_file else []

        attempts.append(common + yt_common + cookie_args + ['-f', 'best[ext=mp4]/best'])
        attempts.append(common + yt_common + cookie_args + ['-f', 'bestvideo+bestaudio/best'])
        attempts.append(common + yt_common + cookie_args + ['-f', 'bestvideo*+bestaudio*/best*'])
        attempts.append(common + yt_common + ['-f', 'best[ext=mp4]/best'])
        attempts.append(common + yt_common + ['-f', 'bestvideo+bestaudio/best'])
        attempts.append(common + [
            '--extractor-args', 'youtube:player-client=web_embedded',
            '-f', 'best[ext=mp4]/best',
        ])

    elif is_threads:
        # Threads is NOT a native upstream yt-dlp extractor. The dedicated
        # yt-dlp-threads plugin is installed from requirements.txt. It uses
        # the public crawler-rendered post data, supports both /share/... and
        # /@user/post/... URLs, and extracts the real CDN MP4 URL.
        #
        # Do not pass YouTube cookies here: the Threads extractor intentionally
        # works with public posts and uses its own Googlebot request headers.
        threads_common = common + [
            '--referer', 'https://www.threads.com/',
            '-f', 'best[ext=mp4]/best',
        ]
        attempts.append(threads_common)

    else:
        attempts.append(common + ['-f', 'best[ext=mp4]/best'])
        attempts.append(common + ['-f', 'bestvideo+bestaudio/best'])

    errors = []
    success = False

    try:
        for index, args in enumerate(attempts, 1):
            # Remove leftovers from a failed previous attempt.
            for p in destination.parent.glob(destination.stem + '.*'):
                if p.suffix.lower() in {'.mp4', '.webm', '.mkv', '.mov', '.m4v', '.avi', '.part', '.ytdl'}:
                    try:
                        p.unlink()
                    except Exception:
                        pass

            cmdline = args + ['-o', template, url]
            try:
                cmd(cmdline)
                success = True
                break
            except Exception as e:
                detail = str(e)
                errors.append(f'Attempt {index}: {detail}')
                continue

        if not success:
            detail = '\n\n'.join(errors[-3:])

            if is_threads:
                if 'Unsupported URL' in detail and ('threads.com' in detail or 'threads.net' in detail):
                    raise RuntimeError(
                        'Threads downloader plugin is not loaded in this Space. '
                        'Make sure requirements.txt contains the yt-dlp-threads package and rebuild the Space.\n\n'
                        + detail
                    )
                raise RuntimeError(
                    'Could not download this Threads video. The Threads extractor supports public video posts, '
                    'including /share/... and /@user/post/... links. Private, deleted, login-gated, or image/text-only posts are not downloadable.\n\n'
                    + detail
                )

            if is_yt:
                if 'Sign in to confirm' in detail or 'not a bot' in detail or 'LOGIN_REQUIRED' in detail:
                    raise RuntimeError(
                        'YouTube is still asking the server for browser verification. '
                        'The PO-token provider and automatic format fallbacks were tried. '
                        'Please refresh the YouTube cookies and update YOUTUBE_COOKIES_B64.\n\n'
                        + detail
                    )
                if 'Requested format is not available' in detail or 'Only images are available' in detail:
                    raise RuntimeError(
                        'YouTube returned no downloadable video format for this request. '
                        'The app automatically tried mweb + PO token, multiple format selectors, '
                        'a no-cookie retry, and an embeddable-video fallback.\n\n'
                        + detail
                    )

            raise RuntimeError(
                'Could not download this video link. Make sure the post is public and the link is valid.\n'
                + detail
            )
    finally:
        try:
            cf = BASE / '.youtube-cookies.txt'
            if cf.exists():
                cf.unlink()
        except Exception:
            pass

    candidates = sorted(
        destination.parent.glob(destination.stem + '.*'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [
        p for p in candidates
        if p.suffix.lower() in {'.mp4', '.webm', '.mkv', '.mov', '.m4v', '.avi'}
    ]
    if not candidates:
        raise RuntimeError('The platform did not provide a downloadable video for this link.')
    if candidates[0] != destination:
        shutil.move(str(candidates[0]), str(destination))
    return destination


@app.post('/api/process')
async def process(background_tasks:BackgroundTasks,video:UploadFile=File(None),video_url:str=Form(''),
                  voice:str=Form('Kore'),style:str=Form('natural'),custom_style:str=Form(''),
                  subtitle_mode:str=Form('myanmar'),color_preset:str=Form('original'),
                  brightness:float=Form(0),contrast:float=Form(1),saturation:float=Form(1),
                  subtitle_font_size:int=Form(52),subtitle_scale_x:int=Form(100),subtitle_scale_y:int=Form(100),
                  subtitle_x:float=Form(50),subtitle_y:float=Form(88),subtitle_alignment:int=Form(2),
                  subtitle_color:str=Form('#FFFFFF'),subtitle_outline_width:float=Form(3),
                  subtitle_outline_color:str=Form('#000000'),subtitle_shadow:float=Form(1),
                  subtitle_background:str=Form('true'),subtitle_background_color:str=Form('#000000'),
                  subtitle_background_opacity:float=Form(55),music_track:str=Form('none'),music_volume:float=Form(.12),auto_thumbnail:str=Form('true')):
    if not GEMINI_KEY: raise HTTPException(400,'GEMINI_API_KEY is missing. Add it in Space Settings → Secrets.')
    if not video and not video_url.strip():
        raise HTTPException(400,'Upload a video or paste a YouTube, Facebook, TikTok, or Threads video link.')
    j=uuid.uuid4().hex; inp=UPLOADS/f'{j}.mp4'
    if video_url.strip():
        validate_video_url(video_url)
        jobs[j]={'status':'queued','progress':2,'message':'Downloading video from link…'}
        try:
            download_video_url(video_url,inp)
        except Exception as e:
            jobs[j]={'status':'error','progress':0,'message':str(e)}
            raise HTTPException(400,str(e))
    else:
        ext=Path(video.filename or '').suffix.lower() or '.mp4'
        if ext not in {'.mp4','.mov','.mkv','.webm','.avi','.m4v'}:
            raise HTTPException(400,'Unsupported video format.')
        with inp.with_suffix(ext).open('wb') as f: shutil.copyfileobj(video.file,f)
        inp=inp.with_suffix(ext)
        jobs[j]={'status':'queued','progress':1,'message':'Queued…'}
    background_tasks.add_task(work,j,inp,voice,style,custom_style,subtitle_mode,color_preset,brightness,contrast,saturation,
                              subtitle_font_size,subtitle_scale_x,subtitle_scale_y,subtitle_x,subtitle_y,subtitle_alignment,
                              subtitle_color,subtitle_outline_width,subtitle_outline_color,subtitle_shadow,
                              subtitle_background.lower() in ('true','1','yes','on'),subtitle_background_color,subtitle_background_opacity,music_track,music_volume,auto_thumbnail.strip().lower()=='true')
    return {'job_id':j}

@app.get('/api/jobs/{j}')
def job(j):
    if j not in jobs: raise HTTPException(404,'Job not found.')
    return jobs[j]

@app.get('/api/jobs/{j}/debug')
def job_debug(j):
    if j not in jobs: raise HTTPException(404,'Job not found.')
    w=JOBS/j
    return {
        'job': jobs[j],
        'files': sorted(x.name for x in w.iterdir()) if w.exists() else [],
        'ffmpeg_filters': sorted(ffmpeg_filters()),
    }
@app.get('/api/jobs/{j}/video')
def job_video(j):
    p=JOBS/j/'final_myanmar.mp4'
    if not p.exists(): raise HTTPException(404,'Video not ready.')
    return FileResponse(p,media_type='video/mp4',filename='the-quick-plot-myanmar.mp4')

@app.get('/api/jobs/{j}/thumbnail')
def job_thumbnail(j):
    p=JOBS/j/'thumbnail.jpg'
    if not p.exists(): raise HTTPException(404,'Thumbnail not ready.')
    return FileResponse(p,media_type='image/jpeg',filename='the-quick-plot-thumbnail.jpg')
