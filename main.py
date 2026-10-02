import gradio as gr

from modules.writer import generate_text
from modules.voice import generate_voice
from modules.video import check_video
from modules.downloader import download_video
from modules.translator import translate_text


CSS = """

body {

background:
linear-gradient(
135deg,
#667eea,
#764ba2
);

}


.gradio-container {

max-width:1200px !important;

}

"""


with gr.Blocks(
    css=CSS,
    title="AI Creator Studio"
) as app:


    gr.Markdown(
"""
# ✨ AI Creator Studio

AI Writing • Voice • Video • Translation
"""
)



    with gr.Tab("✍️ AI Writer"):


        prompt = gr.Textbox(
            label="Your Idea",
            lines=5
        )


        mode = gr.Dropdown(
            [
                "YouTube Script",
                "Blog",
                "Social Media Post",
                "Product Description"
            ],
            value="YouTube Script"
        )


        button = gr.Button(
            "Generate ✨"
        )


        output = gr.Textbox(
            lines=15
        )


        button.click(
            generate_text,
            [prompt, mode],
            output
        )



    with gr.Tab("🎙 Voice"):


        text = gr.Textbox(
            lines=5
        )


        btn = gr.Button(
            "Create Voice"
        )


        audio = gr.Audio()


        btn.click(
            generate_voice,
            text,
            audio
        )



    with gr.Tab("🎬 Video Upload"):


        video = gr.Video()


        result = gr.Textbox()


        video.change(
            check_video,
            video,
            result
        )



    with gr.Tab("🔗 Video URL"):


        url = gr.Textbox(
            label="YouTube / TikTok / Facebook URL"
        )


        btn = gr.Button(
            "Download"
        )


        output_video = gr.Video()


        btn.click(
            download_video,
            url,
            output_video
        )



    with gr.Tab("🌐 Translation"):


        txt = gr.Textbox(
            lines=5
        )


        lang = gr.Dropdown(
            [
                "Myanmar",
                "English"
            ]
        )


        btn = gr.Button(
            "Translate"
        )


        result = gr.Textbox(
            lines=10
        )


        btn.click(
            translate_text,
            [txt, lang],
            result
        )



app.launch(
    server_name="0.0.0.0",
    server_port=7860
)
