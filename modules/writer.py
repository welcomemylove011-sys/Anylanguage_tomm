from openai import OpenAI
from config import OPENAI_API_KEY, GEMINI_API_KEY


def generate_text(prompt, content_type):

    if not prompt:
        return "Please enter your idea."


    # OpenAI
    if OPENAI_API_KEY:

        try:
            client = OpenAI(
                api_key=OPENAI_API_KEY
            )

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "You are an expert content creator."
                    },
                    {
                        "role": "user",
                        "content":
                        f"""
Create a {content_type}.

Topic:
{prompt}
"""
                    }
                ]
            )

            return response.choices[0].message.content


        except Exception as e:
            return f"OpenAI Error: {e}"


    # Gemini Backup

    if GEMINI_API_KEY:

        try:

            import google.generativeai as genai

            genai.configure(
                api_key=GEMINI_API_KEY
            )

            model = genai.GenerativeModel(
                "gemini-1.5-flash"
            )

            result = model.generate_content(
                f"""
Create a {content_type}.

Topic:
{prompt}
"""
            )

            return result.text


        except Exception as e:

            return f"Gemini Error: {e}"


    return "No AI API key found."
