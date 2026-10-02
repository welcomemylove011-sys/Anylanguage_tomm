import os
import yt_dlp


def download_video(url):

    if not url:
        return None


    try:

        output = "outputs/downloaded_video.mp4"


        options = {

            "format":
            "best",

            "outtmpl":
            output,

            "merge_output_format":
            "mp4"

        }


        with yt_dlp.YoutubeDL(options) as ydl:

            ydl.download(
                [url]
            )


        return output


    except Exception as e:

        return f"Download Error: {e}"
