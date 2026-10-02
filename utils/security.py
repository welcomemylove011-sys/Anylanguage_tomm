import os


def check_api_keys():

    keys = {

        "OPENAI_API_KEY":
        os.getenv("OPENAI_API_KEY"),

        "GEMINI_API_KEY":
        os.getenv("GEMINI_API_KEY"),

        "ELEVENLABS_API_KEY":
        os.getenv("ELEVENLABS_API_KEY")

    }


    return keys
