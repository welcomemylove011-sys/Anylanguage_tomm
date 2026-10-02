from modules.writer import generate_text


def translate_text(text, language):

    if not text:
        return "Enter text first."


    prompt = f"""
Translate this text into {language}.

Keep the meaning natural.

Text:

{text}
"""


    return generate_text(
        prompt,
        "translation"
    )
