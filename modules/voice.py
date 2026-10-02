from config import ELEVENLABS_API_KEY
import os


def generate_voice(text):

    if not text:
        return None


    if not ELEVENLABS_API_KEY:
        return "ElevenLabs API key missing."


    try:

        from elevenlabs.client import ElevenLabs


        client = ElevenLabs(
            api_key=ELEVENLABS_API_KEY
        )


        audio = client.text_to_speech.convert(
            voice_id="Rachel",
            model_id="eleven_multilingual_v2",
            text=text
        )


        output = "outputs/voice.mp3"


        with open(output,"wb") as f:

            for chunk in audio:
                f.write(chunk)


        return output


    except Exception as e:

        return f"Voice Error: {e}"
