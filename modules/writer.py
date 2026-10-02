from openai import OpenAI
from config import OPENAI_API_KEY, GEMINI_API_KEY


def generate_text(prompt, content_type):

    if not prompt or not prompt.strip():
        return "Please enter your idea."

    request = f"""
Create a {content_type}.

Topic:
{prompt}

Make the result natural, useful, and well structured.
"""

    # OpenAI
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI content creator."
                    },
                    {
                        "role": "user",
                        "content": request
                    }
                ],
                temperature=0.7
            )

            return response.choices[0].message.content or "No response."

        except Exception as e:
            openai_error = str(e)
        else:
            openai_error = ""

    else:
        openai_error = "OPENAI_API_KEY is not available."

    # Gemini fallback
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)

            model = genai.GenerativeModel(
                "gemini-1.5-flash"
            )

            result = model.generate_content(request)

            if result.text:
                return result.text

        except Exception as e:
            return (
                "AI generation failed.\n\n"
                f"OpenAI: {openai_error}\n"
                f"Gemini: {e}"
            )

    return (
        "AI generation failed.\n\n"
        f"OpenAI: {openai_error}\n"
        "Gemini API key is not available."
    )
