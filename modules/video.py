import os
from config import OUTPUT_DIR, MAX_VIDEO_SIZE_MB


def check_video(video):

    if video is None:
        return "No video uploaded."


    try:

        size_mb = (
            os.path.getsize(video)
            /
            (1024 * 1024)
        )


        if size_mb > MAX_VIDEO_SIZE_MB:
            return (
                f"Video too large. "
                f"Maximum {MAX_VIDEO_SIZE_MB}MB"
            )


        return (
            "✅ Video uploaded successfully\n"
            f"Size: {round(size_mb,2)} MB"
        )


    except Exception as e:

        return f"Video Error: {e}"
